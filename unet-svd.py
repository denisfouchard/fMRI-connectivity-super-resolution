from torch.utils.data import DataLoader
import pandas as pd
import torch
import copy
from tqdm import tqdm

# from metrics import evaluation_metrics

from slim import SLIMDataModule
import torch.nn as nn
import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.nn as nn
import numpy as np
import networkx as nx


def symmetric_normalize(A_tilde):
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


def batch_normalize(batch):
    batch_n = torch.zeros_like(batch)
    for i, A in enumerate(batch):
        batch_n[i] = symmetric_normalize(A + torch.eye(n=A.shape[0]))
    return batch_n


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
    criterion=None,
    intermediate_losses=False,
    skip=False,
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
        optimizer, mode="min", patience=patience
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
            outputs, A_hist, A_recon_hist = model.forward(A=inputs, X=X, skip=skip)
            loss = criterion(
                outputs,
                targets.to(model.device),
                A_hist,
                A_recon_hist,
                intermediate_losses=intermediate_losses,
            )
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
                    targets = targets.to(model.device)
                    inputs = inputs.squeeze(0)
                    targets = targets.squeeze(0)
                    X = val_node_features[i] if val_node_features is not None else None
                    outputs, A_hist, A_recon_hist = model(A=inputs, X=X, skip=skip)

                    val_loss += criterion(
                        outputs,
                        targets,
                        A_hist,
                        A_recon_hist,
                        intermediate_losses,
                    ).item()

                    A = outputs - torch.diag(torch.diag(outputs)).to(model.device)
                    A_true = targets - torch.diag(torch.diag(targets)).to(model.device)

                    val_mae += F.l1_loss(A, A_true).item()

            val_loss /= len(val_dataloader)
            val_mae /= len(val_dataloader)
            val_loss_history.append(val_loss)
            scheduler.step(val_loss)

            lr = get_lr(optimizer)
            lr_history.append(lr)

            # Check if this is the best f1 score so far
            if val_loss > best_val_loss:
                best_val_loss = val_loss
                best_model_state_dict = copy.deepcopy(model.state_dict())

            if lr < 1e-5:
                break

        progress_bar.set_postfix(
            {"train_loss": avg_loss, "val_loss": val_loss, "lr": lr, "val_mae": val_mae}
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


# putting it all together - code taken from https://github.com/HongyangGao/Graph-U-Nets/tree/master


def reconstruct_adjacency(X, threshold=0.2):
    """
    Reconstruct adjacency from node embeddings while preserving fMRI-like structure.

    Args:
        X (torch.Tensor): Node embeddings of shape [num_nodes, hidden_dim]
        threshold (float): Value below which edges are removed for sparsity

    Returns:
        adj (torch.Tensor): Reconstructed weighted adjacency matrix
    """
    # Compute differenciable Pearson correlation
    adj = torch.matmul(X, X.T)
    return adj


class GraphUpsampler(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        n_nodes,
        m_nodes,
        act,
        drop_p,
    ):
        """
        Args:
        - in_dim: Input node feature dimension
        - hidden_dim: Hidden dimension for message passing
        - num_iterations: Number of iterative updates
        - upsample_factor: Factor by which to increase node count
        """
        super(GraphUpsampler, self).__init__()
        self.n_nodes = n_nodes
        self.m_nodes = m_nodes

        # MLP for new node generation
        self.upsample_mlp = nn.Linear(n_nodes, m_nodes, bias=False)

    def forward(self, X, A, refine=False):
        """
        Args:
        - x: Node features [num_nodes, in_dim]
        - adj_matrix: Initial adjacency matrix [num_nodes, num_nodes]

        Returns:
        - Upsampled adjacency matrix [self.m_nodes, self.m_nodes]
        - Upsampled node features [new_num_nodes, in_dim]
        """

        # Generate new nodes by transforming existing ones
        X_upsampled = torch.relu(self.upsample_mlp(X.T).T)  # [num_nodes, in_dim]
        # Concatenate old and new nodes

        A_upsampled = reconstruct_adjacency(X=X_upsampled)
        return A_upsampled


class GraphUnet(nn.Module):

    def __init__(self, ks, n_nodes, m_nodes, dim, act, drop_p):
        super(GraphUnet, self).__init__()
        self.ks = ks
        self.dim = dim

        self.down_gcns = nn.ModuleList()
        self.up_gcns = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.unpools = nn.ModuleList()
        self.upsampler = GraphUpsampler(
            in_dim=dim,
            hidden_dim=dim,
            n_nodes=n_nodes,
            m_nodes=m_nodes,
            act=act,
            drop_p=drop_p,
        )
        self.l_n = len(ks)
        for k in ks:
            # out_dim = dim
            out_dim = int(dim / k)
            self.down_gcns.append(GIN(dim, out_dim, act, drop_p))
            self.up_gcns.append(GIN(out_dim, dim, act, drop_p))
            self.pools.append(Pool(k, out_dim, drop_p))
            self.unpools.append(Unpool(dim, dim, drop_p))
            dim = out_dim

        self.up_gcns = self.up_gcns[::-1]
        # self.node_features = nn.Parameter(torch.randn(n_nodes, dim))
        self.bottom_gcn = GCN(dim, dim, act, drop_p)

    @property
    def device(self):
        return next(self.parameters()).device

    def build_node_features(self, adjacency: torch.Tensor, dim: int) -> torch.Tensor:
        # Perform SVD on the adjacency matrix
        U, S, _ = torch.svd(adjacency)
        U = U[:, :dim]
        S = S[:dim]
        U = U * torch.sqrt(S)
        return U

    def forward(
        self, A: torch.Tensor, skip: bool = False, threshold: float = -1, X=None
    ):
        # Process A
        if threshold > 0:
            A = torch.where(A > threshold, A, torch.zeros_like(A))
        A = A + torch.eye(A.shape[0])
        A = symmetric_normalize(A)
        A = A.to(self.device)

        if X is None:
            X = self.build_node_features(A, self.dim).to(self.device)
        else:
            X = X.to(self.device)

        A_history = []
        A_recon_history = []
        indices_list = []
        down_outs = []
        if skip:
            org_X = X.clone()
        for i in range(self.l_n):
            X = self.down_gcns[i](A, X)
            A_history.append(A)
            down_outs.append(X)
            A, X, idx = self.pools[i](A, X)
            indices_list.append(idx)

        X = self.bottom_gcn(A, X)
        for i in range(self.l_n):
            up_idx = self.l_n - i - 1
            A, idx = A_history[up_idx], indices_list[up_idx]
            A, X = self.unpools[i](A, X, down_outs[up_idx], idx)
            X = self.up_gcns[i](A, X)

            A_recon = reconstruct_adjacency(X)
            A_recon_history.append(A_recon)
            if skip:

                X = X.add(down_outs[up_idx])

        if skip:
            X = X.add(org_X)

        A_upsampled = self.upsampler.forward(X, A)

        return A_upsampled, A_history, A_recon_history


class GCN(nn.Module):

    def __init__(self, in_dim, out_dim, act, p):
        super(GCN, self).__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        self.act = act
        self.drop = nn.Dropout(p=p) if p > 0.0 else nn.Identity()

    def forward(self, A, X):
        X = self.drop(X)  # they have added dropout
        X = torch.matmul(A, X)
        X = self.proj(X)
        X = self.act(X)
        return X


class GIN(nn.Module):
    def __init__(self, in_dim, out_dim, act, p):
        super(GIN, self).__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        self.act = act
        self.drop = nn.Dropout(p=p) if p > 0.0 else nn.Identity()
        self.eps = nn.Parameter(torch.Tensor([0.01]), requires_grad=True)

    def forward(self, A, X):
        X = self.drop(X)  # they have added dropout
        X = torch.matmul(A + self.eps * torch.eye(A.shape[0]).to(X.device), X)
        X = self.proj(X)
        X = self.act(X)
        return X


class GIN2(nn.Module):
    def __init__(self, in_dim, out_dim, act, eps=0, train_eps=True):
        super(GIN, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)
        )
        self.eps = nn.Parameter(torch.Tensor([eps])) if train_eps else eps

    def forward(self, A, X):
        out = torch.matmul(A, X)
        out = (1 + self.eps) * X + out
        out = self.mlp(out)
        return out


class Pool(nn.Module):

    def __init__(self, k, in_dim, p):
        super(Pool, self).__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()  # added dropout here

    def forward(self, g, h):
        Z = self.drop(h)
        weights = self.proj(Z).squeeze()
        scores = self.sigmoid(weights)
        return top_k_graph(scores, g, h, self.k)


class Unpool(nn.Module):

    def __init__(self, *args):
        super(Unpool, self).__init__()

    def forward(self, A, X, pre_h, idx):
        new_h = X.new_zeros([A.shape[0], X.shape[1]])
        new_h[idx] = X
        return A, new_h


def top_k_graph(scores, A, X, k):
    num_nodes = A.shape[0]
    values, idx = torch.topk(
        scores, max(2, int(k * num_nodes))
    )  # make sure k works based on number of current nodes
    X_pooled = X[idx, :]
    values = torch.unsqueeze(values, -1)
    X_pooled = torch.mul(X_pooled, values)
    # A_treshold = torch.where(A > 0.10, torch.ones_like(A), torch.zeros_like(A))
    # A_pooled = A_treshold.bool().float()
    # A_pooled = (
    #     torch.matmul(A_pooled, A_pooled).bool().float()
    # )  # second power to reduce chance of isolated nodes
    A_pooled = A[idx, :]
    A_pooled = A_pooled[:, idx]
    A_pooled = symmetric_normalize(A_pooled)
    return A_pooled, X_pooled, idx


def symmetric_normalize(A_tilde):
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


from MatrixVectorizer import MatrixVectorizer
import pandas as pd


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
        outputs, _, _ = model(inputs, X=X)
        preds.append(outputs.detach().cpu().numpy())

    # Vectorize matrices
    preds = [MatrixVectorizer.vectorize(p) for p in preds]
    preds = np.array(preds)

    # Submission format
    print(preds.shape)
    submission_df = pd.DataFrame(
        {"ID": range(1, len(preds.flatten()) + 1), "Predicted": preds.flatten()}
    )
    submission_df.to_csv("outputs/unet/submission.csv", index=False)


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
        preds.append(outputs[0].detach().cpu().numpy())

    # Vectorize matrices
    preds_v = [MatrixVectorizer.vectorize(p) for p in preds]
    preds_v = np.array(preds_v)

    # Submission format
    submission_df = pd.DataFrame(
        {"ID": range(1, len(preds_v.flatten()) + 1), "Predicted": preds_v.flatten()}
    )
    submission_df.to_csv("outputs/gat/submission.csv", index=False)
    return torch.tensor(preds)


def loss(
    A_true, A_pred, A_hist=None, A_recon_hist=None, intermediate_losses: bool = True
):
    # Remove diagonal from A_true and A_pred
    A_true_ = A_true - torch.diag(torch.diag(A_true))
    A_pred_ = A_pred - torch.diag(torch.diag(A_pred))
    loss = F.mse_loss(A_true_, A_pred_)
    loss += 0.2 * F.l1_loss(A_true_, A_pred_)

    if intermediate_losses:
        i = 1
        for A, A_recon in zip(A_hist, A_recon_hist[::-1]):
            A_ = A - torch.diag(torch.diag(A))
            A_recon_ = A_recon - torch.diag(torch.diag(A_recon))
            loss += F.mse_loss(A_, A_recon_)
            i += 1
    return loss


if __name__ == "__main__":
    import os

    data_module = SLIMDataModule(data_dir="./data", batch_size=1)
    train_dataloader = data_module.train_dataloader()
    # Get first batch
    batch = next(iter(train_dataloader))

    from slim import create_test_dataloader

    test_dataloader = create_test_dataloader(data_dir="./data", batch_size=1)

    # Define the model, loss function, and optimizer

    # Clear CUDA cache
    torch.cuda.empty_cache()

    # Define the model
    in_dim = batch[0].shape[1]
    out_dim = batch[1].shape[1]
    dim = 30
    model = GraphUnet(
        ks=[
            0.75,
            0.75,
            0.75,
        ],
        n_nodes=in_dim,
        m_nodes=out_dim,
        dim=dim,
        act=torch.relu,
        drop_p=0.1,
    )
    model.to(torch.device("cuda:2"))
    # model.load_state_dict(torch.load("./outputs/unet/unet-26-02.pth"))

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

        X_val = [
            model.build_node_features(A[0].squeeze(0)) for A in tqdm(test_dataloader)
        ]

    else:
        print("Loading node features...")
        train_node_features = torch.load("data/train_node_features_15.pt")
        val_node_features = torch.load("data/val_node_features_15.pt")
        X_val = torch.load("data/test_node_features_15.pt")

    train_losses, val_losses, lr, _ = train_model(
        model=model,
        train_dataloader=data_module.train_dataloader(),
        val_dataloader=data_module.val_dataloader(),
        # train_node_features=train_node_features,
        # val_node_features=val_node_features,
        num_epochs=200,
        lr=0.01,
        validate_every=1,
        patience=2,
        criterion=loss,
        intermediate_losses=True,
        skip=False,
    )

    predict(model, test_dataloader, X_test=X_val)
