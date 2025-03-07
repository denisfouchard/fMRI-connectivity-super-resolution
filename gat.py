from torch.utils.data import Dataset, DataLoader
import torch
import copy
from tqdm import tqdm
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.nn import TransformerConv
import networkx as nx
import torch.nn as nn
from slim import SLIMDataModule
import torch.nn as nn
from MatrixVectorizer import MatrixVectorizer
import pandas as pd
from slim import create_test_dataloader
import numpy as np


def sym_norm_adj(A_tilde):
    """
    Performs symmetric normalization of A_tilde (Adj. matrix with self loops):
      A_norm = D^{-1/2} * A_tilde * D^{-1/2}
    Where D_{ii} = sum of row i in A_tilde.

    A_tilde (N, N): Adj. matrix with self loops
    Returns:
      A_norm : (N, N)
    """

    eps = 1e-5
    d = A_tilde.sum(dim=1) + eps
    D_inv = torch.diag(torch.pow(d, -0.5))
    return D_inv @ A_tilde @ D_inv


def differentiable_corrcoef(A):
    """
    Compute a differentiable correlation matrix for a batch of matrices A.
    Assumes A has shape (batch, num_nodes, num_nodes).
    """
    mean_A = A.mean(dim=-1, keepdim=True)  # Mean over last dimension
    A_centered = A - mean_A  # Center the matrix

    # Compute covariance
    cov = torch.matmul(A_centered, A_centered.transpose(-1, -2))

    # Compute standard deviations
    std = torch.sqrt(
        torch.sum(A_centered**2, dim=-1, keepdim=True) + 1e-6
    )  # Add epsilon for stability

    # Compute correlation matrix
    corr_matrix = cov / (
        std @ std.transpose(-1, -2)
    )  # Outer product of std to normalize

    return corr_matrix


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def train_model(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    train_node_features=None,
    val_node_features=None,
    num_epochs=100,
    lr=0.01,
    validate_every=1,
    patience=10,
    criterion=nn.MSELoss(),
):
    """
    Train the model, validate every 'validate_every' epochs, and pick the
    checkpoint with best validation accuracy.

    Parameters:
    -----------
    model : torch.nn.Module
        The PyTorch model to train.
    train_dataloader : torch.utils.data.DataLoader
        DataLoader for the training set.
    val_dataloader : torch.utils.data.DataLoader
        DataLoader for the validation set.
    num_epochs : int
        Number of training epochs.
    lr : float
        Learning rate for the optimizer.
    validate_every : int
        Validate (and possibly checkpoint) every 'validate_every' epochs.
    patience : int
        Patience for learning rate scheduler.
    criterion : torch.nn.Module
        Loss function.

    Returns:
    --------
    best_loss_history : list
        The training loss history across epochs.
    best_model_state_dict : dict
        The state dictionary of the model achieving the best validation accuracy.
    """

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=patience, cooldown=1, factor=0.1, verbose=True
    )
    train_loss_history = []
    val_loss_history = []
    lr_history = []

    best_val_loss = torch.inf
    best_model_state_dict = None
    val_loss = 0.0
    val_mae = 0.0

    progress_bar = tqdm(range(num_epochs))
    for epoch in progress_bar:
        progress_bar.set_description(f"Epoch {epoch}|{num_epochs}")
        model.train()
        epoch_loss = 0.0

        for i, batch in enumerate(train_dataloader):
            inputs, targets = batch
            # inputs = batch_normalize(inputs)
            inputs = inputs.squeeze(0)
            targets = targets.squeeze(0)
            optimizer.zero_grad()

            X = train_node_features[i] if train_node_features is not None else None

            # Forward pass on training data
            outputs = model.forward(A=inputs, X=X)
            loss = criterion(
                outputs,
                targets.to(model.device),
            )
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            # Record training loss
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_dataloader)
        train_loss_history.append(avg_loss)

        # Validation step
        if (epoch + 1) % validate_every == 0 or (epoch + 1) == num_epochs:
            model.eval()
            val_loss = 0.0
            val_mae = 0.0
            with torch.no_grad():
                for i, batch in enumerate(val_dataloader):
                    inputs, targets = batch
                    inputs = inputs.squeeze(0)
                    targets = targets.squeeze(0)
                    X = val_node_features[i] if val_node_features is not None else None
                    outputs = model(
                        A=inputs,
                        X=X,
                    )

                    val_loss += criterion(
                        outputs,
                        targets.to(model.device),
                    ).item()

                    val_mae += F.l1_loss(outputs, targets.to(model.device)).item()

            val_loss /= len(val_dataloader)
            val_mae /= len(val_dataloader)
            val_loss_history.append(val_loss)
            scheduler.step(val_loss)

            lr = get_lr(optimizer)
            lr_history.append(lr)

            # Check if this is the best f1 score so far
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state_dict = copy.deepcopy(model.state_dict())

            if lr < 1e-5:
                break

        progress_bar.set_postfix(
            {"train_loss": avg_loss, "val_loss": val_loss, "lr": lr, "mae": val_mae}
        )

    # If we have a best model, load it
    if best_model_state_dict is not None:
        model.load_state_dict(best_model_state_dict)

    return train_loss_history, val_loss_history, lr_history, best_model_state_dict


@torch.no_grad()
def evaluate_model(model, dataloader):
    """
    Runs forward pass, calculates binary predictions (threshold=0.5),
    and returns the accuracy score.
    """
    from metrics import evaluation_metrics

    model.eval()

    preds = []
    true = []
    for batch in dataloader:
        inputs, targets = batch
        inputs = inputs.squeeze(0)
        targets = targets.squeeze(0)
        inputs.to(model.device)
        outputs, _, _ = model(inputs)
        preds.append(outputs.detach().cpu().numpy())
        true.append(targets.detach().cpu().numpy())

    batch_metrics = evaluation_metrics(preds, true)

    return batch_metrics


def top_k_eigenvec(A: torch.Tensor, k=10):
    """Compute top-k eigenvectors of the normalized Laplacian for graph positional encoding."""
    D = torch.diag(A.sum(dim=1)).to(A.device)  # Degree matrix
    L = D - A  # Laplacian
    eigvals, eigvecs = torch.linalg.eigh(L)  # Eigen decomposition
    return eigvecs[:, :k]  # Take the top-k eigenvectors


class GraphTransformerBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=4):
        super().__init__()
        self.transformer = TransformerConv(
            in_dim,
            out_dim // num_heads,
            heads=num_heads,
            dropout=0.1,
            edge_dim=1,
            concat=True,
        )
        self.norm = torch.nn.LayerNorm(out_dim)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.transformer(x, edge_index=edge_index, edge_attr=edge_attr)
        return self.norm(F.relu(x))  # Apply normalization & activation


def graph_spectral_upsample(A, new_size=260):
    """Upsample a graph adjacency matrix using Laplacian eigenvector interpolation."""
    # Compute Laplacian eigenvectors of the original graph
    D = torch.diag(A.sum(dim=1))
    L = D - A
    eigvals, eigvecs = torch.linalg.eigh(L)

    # Interpolate to new size
    upsampled_eigvecs = (
        F.interpolate(eigvecs.T.unsqueeze(0), size=new_size, mode="linear").squeeze(0).T
    )
    return torch.mm(upsampled_eigvecs, upsampled_eigvecs.T)  # Reconstruct adjacency


class GraphTransformerUpscaler(nn.Module):
    def __init__(
        self, in_dim=15, hidden_dim=128, num_layers=9, num_heads=4, upsample_size=268
    ):
        super().__init__()
        self.dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Graph Transformer Layers
        self.layers: nn.ModuleList[GraphTransformerBlock] = nn.ModuleList(
            [
                GraphTransformerBlock(hidden_dim, hidden_dim, num_heads)
                for _ in range(num_layers)
            ]
        )

        # Upscaling projection
        self.upsample_proj = nn.Linear(160, upsample_size)

    def build_node_features(self, adjacency: torch.Tensor) -> torch.Tensor:
        # Build node features using topological information

        # Compute degree matrix
        D = torch.diag(torch.sum(adjacency, dim=1)).cpu()

        # Compute Node betweenness centrality
        G = nx.from_numpy_array(adjacency.cpu().numpy())
        betweenness = torch.tensor(
            list(nx.betweenness_centrality(G).values()), dtype=torch.float32
        )

        # Compute Node closeness centrality
        closeness = torch.tensor(
            list(nx.closeness_centrality(G).values()), dtype=torch.float32
        )

        # Compute Node clustering coefficient
        clustering = torch.tensor(list(nx.clustering(G).values()), dtype=torch.float32)

        # Compute eigenvector centrality
        eigenvector = torch.tensor(
            list(nx.eigenvector_centrality(G).values()), dtype=torch.float32
        )

        # Compute Laplacian eigenvecs
        eigvals, eigvecs = torch.linalg.eigh(D - adjacency.cpu())
        laplacian_eigenvecs = eigvecs[:, 1 : self.dim - 3]

        # Concatenate all features
        node_features = torch.stack(
            [betweenness, closeness, clustering, eigenvector], dim=1
        )
        # Include Laplacian eigenvecs
        node_features = torch.cat([node_features, laplacian_eigenvecs], dim=1)

        return node_features

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, A, X):
        A = A.to(self.device)
        X = X.to(self.device)
        edge_index, edge_attr = dense_to_sparse(A)

        # Compute spectral embeddings
        spectral_emb = top_k_eigenvec(A, k=self.hidden_dim - self.dim)
        x = torch.cat([X, spectral_emb], dim=1)  # Concatenate features

        # Upsample adjacency matrix
        edge_attr = edge_attr.unsqueeze(1)

        # Apply Transformer layers
        for layer in self.layers:
            x = layer.forward(x=x, edge_index=edge_index, edge_attr=edge_attr) + x
            A = torch.mm(x, x.T) - torch.eye(x.shape[0]).to(x.device)
            # edge_index, edge_attr = dense_to_sparse(A)
            # edge_attr = edge_attr.unsqueeze(1)

        # Normalize node features
        x_up = self.upsample_proj(x.T).T
        x_up = F.relu(x_up)
        # Project back to original dimension
        A_pred = torch.mm(x_up, x_up.T) - torch.eye(x_up.shape[0]).to(x_up.device)
        # A = graph_spectral_upsample(A, new_size=268)
        # A_pred = differentiable_corrcoef(x_up)
        return A_pred


def criterion(
    A_true,
    A_pred,
):
    # Compute laplacian loss
    A_pred_ = A_pred - torch.eye(A_pred.shape[0]).to(A_pred.device)
    A_true_ = A_true - torch.eye(A_true.shape[0]).to(A_true.device)

    loss = F.mse_loss(A_pred_, A_true_) + 1e-3 * F.l1_loss(A_pred_, A_true_)
    return loss


@torch.no_grad()
def predict(model, dataloader, X_test=None):
    model.eval()

    preds = []
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    progress_bar.set_description("Predicting...")
    for i, batch in progress_bar:

        inputs = batch.squeeze(0)
        inputs.to(model.device)
        X = X_test[i] if X_test is not None else None
        outputs = model(inputs, X=X)
        preds.append(outputs.detach().cpu().numpy())

    # Vectorize matrices
    preds_v = [MatrixVectorizer.vectorize(p) for p in preds]
    preds_v = np.array(preds_v)

    # Submission format
    submission_df = pd.DataFrame(
        {"ID": range(1, len(preds_v.flatten()) + 1), "Predicted": preds_v.flatten()}
    )
    submission_df.to_csv("outputs/gat/submission.csv", index=False)
    return torch.tensor(np.array(preds))


if __name__ == "__main__":
    import os

    # Clear CUDA cache
    torch.cuda.empty_cache()

    data_module = SLIMDataModule(data_dir="./data", batch_size=1)
    train_dataloader = data_module.train_dataloader()
    # Get first batch
    batch = next(iter(train_dataloader))

    print(torch.diag(batch[0]).mean())

    # Define the model
    in_dim = batch[0].shape[1]
    out_dim = batch[1].shape[1]
    dim = 15
    model = GraphTransformerUpscaler(
        in_dim=dim, hidden_dim=32, num_layers=7, num_heads=2
    )
    model.to(torch.device("cuda:1"))
    # model.load_state_dict(torch.load("outputs/gat/graph_transformer_15.pth"))

    if not os.path.exists("data/test_node_features_15.pt"):
        print("Computing train node features...")
        train_node_features = [
            model.build_node_features(A[0].squeeze(0))
            for A in tqdm(data_module.train_dataloader())
        ]

        print("Computing val node features...")
        val_node_features = [
            model.build_node_features(A[0].squeeze(0))
            for A in tqdm(data_module.val_dataloader())
        ]

        from slim import create_test_dataloader

        test_dataloader = create_test_dataloader(data_dir="./data", batch_size=1)
        X_val = [
            model.build_node_features(A[0].squeeze(0)) for A in tqdm(test_dataloader)
        ]

    else:
        print("Loading node features...")
        train_node_features = torch.load("data/train_node_features_15.pt")
        val_node_features = torch.load("data/val_node_features_15.pt")
        X_val = torch.load("data/test_node_features_15.pt")

    train_losses, val_losses, lr, best_model_dict = train_model(
        model=model,
        train_dataloader=data_module.train_dataloader(),
        val_dataloader=data_module.val_dataloader(),
        train_node_features=train_node_features,
        val_node_features=val_node_features,
        num_epochs=200,
        lr=0.001,
        validate_every=1,
        patience=2,
        criterion=criterion,
    )

    # Save the model
    torch.save(model.state_dict(), "outputs/gat/graph_transformer_sq_15.pth")

    # Evaluate the model
    print("Evaluating model...")
    test_dataloader = create_test_dataloader(data_dir="./data", batch_size=1)
    preds = predict(model, test_dataloader, X_test=X_val)
