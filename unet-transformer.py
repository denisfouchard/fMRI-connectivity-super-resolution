from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import copy
from tqdm import tqdm
from torch_geometric.nn import GATConv, TransformerConv, GINConv
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.nn as nn
import numpy as np
import networkx as nx
from slim import SLIMDataModule
import torch.nn as nn
from sklearn.model_selection import KFold
from slim import create_test_dataloader
from torch.utils.data import DataLoader, Subset

from utils.evaluation import print_metrics


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
        optimizer, mode="min", patience=patience, threshold=1e-2, factor=0.1
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

            # Forward pass on training data
            outputs, A_hist, A_recon_hist = model.forward(A=inputs, skip=skip)
            loss = criterion(
                outputs,
                targets.to(model.device),
                A_hist,
                A_recon_hist,
                intermediate_losses=intermediate_losses,
            )
            loss.backward()

            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
                    targets = targets.to(model.device)
                    targets = targets.squeeze(0)
                    outputs, A_hist, A_recon_hist = model(A=inputs,  skip=skip)

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


def reconstruct_adjacency(X, threshold=0.2):
    """
    Reconstruct adjacency from node embeddings while preserving fMRI-like structure.

    Args:
        X (torch.Tensor): Node embeddings of shape [num_nodes, hidden_dim]
        threshold (float): Value below which edges are removed for sparsity

    Returns:
        adj (torch.Tensor): Reconstructed weighted adjacency matrix
    """
    X_norm = X
    # Compute cosine similarity matrix
    adj = F.relu((X_norm @ X_norm.T))  # Values in range [-1, 1]

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
        num_iterations=3,
    ):
        """
        Args:
        - in_dim: Input node feature dimension
        - hidden_dim: Hidden dimension for message passing
        - num_iterations: Number of iterative updates
        - upsample_factor: Factor by which to increase node count
        """
        super(GraphUpsampler, self).__init__()
        self.num_iterations = num_iterations
        self.n_nodes = n_nodes
        self.m_nodes = m_nodes

        # MLP for new node generation
        self.upsample_mlp = nn.Linear(n_nodes, m_nodes)

    def forward(self, X, A,):
        """
        Args:
        - x: Node features [num_nodes, in_dim]
        - adj_matrix: Initial adjacency matrix [num_nodes, num_nodes]

        Returns:
        - Upsampled adjacency matrix [self.m_nodes, self.m_nodes]
        - Upsampled node features [new_num_nodes, in_dim]
        """

        # Generate new nodes by transforming existing ones
        X_upsampled = self.upsample_mlp(X.T).T  # [num_nodes, in_dim]
        X_upsampled = F.softmax(X_upsampled)
        # Concatenate old and new nodes

        A_upsampled = reconstruct_adjacency(X=X_upsampled)

        return A_upsampled


class GraphUnet(nn.Module):

    def __init__(self, ks, n_nodes, m_nodes, dim, act, drop_p, heads=4):
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
            out_dim = int(dim / k)
            # out_dim = dim
            self.down_gcns.append(GT(dim, out_dim, act, drop_p, heads))
            self.up_gcns.append(GT(out_dim, dim, act, drop_p, heads))
            self.pools.append(Pool(k, out_dim, drop_p))
            self.unpools.append(Unpool(dim, dim, drop_p))
            dim = out_dim

        self.up_gcns = self.up_gcns[::-1]
        # self.node_features = nn.Parameter(torch.randn(n_nodes, dim))
        self.bottom_gcn = GT(dim, dim, act, drop_p)

    @property
    def device(self):
        return next(self.parameters()).device

    def build_batch_features(
        self, batch: list[torch.Tensor], n_jobs: int = 1
    ) -> torch.Tensor:
        # Build batch features using topological information
        from joblib import Parallel, delayed

        # Use the build_node_features function to build features for each graph in the batch
        features = Parallel(n_jobs=n_jobs)(
            delayed(self.build_node_features)(adjacency) for adjacency in batch
        )
        return torch.stack(features, dim=0)

    def build_node_features(self, adjacency: torch.Tensor, dim: int) -> torch.Tensor:
        # Perform SVD on the adjacency matrix
        U, S, _ = torch.svd(adjacency)
        U = U[:, :dim]
        return U


    def forward(
        self, A: torch.Tensor, skip: bool = False, threshold: float = -1, X=None
    ):

        A_ = A + torch.eye(A.shape[0])
        A = symmetric_normalize(A)
        A_ = A_.to(self.device)

        if X is None:
            X = self.build_node_features(A_, self.dim).to(self.device)
     

        A_history = []
        A_recon_history = []
        indices_list = []
        down_outs = []
        if skip:
            org_X = X.clone()
        for i in range(self.l_n):
            X = self.down_gcns[i](A_, X)
            A_history.append(A_)
            down_outs.append(X)
            A_, X, idx = self.pools[i](A_, X)
            indices_list.append(idx)

        X = self.bottom_gcn(A_, X)
        for i in range(self.l_n):
            up_idx = self.l_n - i - 1
            A_, idx = A_history[up_idx], indices_list[up_idx]
            A_, X = self.unpools[i](A_, X, down_outs[up_idx], idx)
            X = self.up_gcns[i](A_, X)

            A_recon = reconstruct_adjacency(X)
            A_recon_history.append(A_recon)
            if skip:

                X = X.add(down_outs[up_idx])

        if skip:
            X = X.add(org_X)

        A_upsampled = self.upsampler.forward(X, A_)

        return A_upsampled, A_history, A_recon_history


class GT(nn.Module):

    def __init__(self, in_dim, out_dim, act, p, heads=2):
        super(GT, self).__init__()
        self.act = act
        # self.gat = TransformerConv(
        #     in_dim, out_dim // heads, heads=heads, dropout=p, edge_dim=1, concat=True
        # )
        self.gat = GATConv(in_dim, out_dim // heads, heads=heads, dropout=p, concat=True)

    def forward(self, A, X):
        edge_index, edge_attr = dense_to_sparse(A)
        edge_attr = edge_attr.unsqueeze(1)
        X = self.gat(X, edge_index, edge_attr)
        X = self.act(X)
        return X


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


def loss(
    A_true, A_pred, A_hist=None, A_recon_hist=None, intermediate_losses: bool = True
):
    # Remove diagonal from A_true and A_pred
    A_true_ = A_true - torch.diag(torch.diag(A_true))
    A_pred_ = A_pred - torch.diag(torch.diag(A_pred))
    loss = F.mse_loss(A_true_, A_pred_)

    if intermediate_losses:
        i = 1
        for A, A_recon in zip(A_hist, A_recon_hist[::-1]):
            A_ = A - torch.diag(torch.diag(A))
            A_recon_ = A_recon - torch.diag(torch.diag(A_recon))
            loss += F.mse_loss(A_, A_recon_)
            i += 1
    return loss


if __name__ == "__main__":

    full_dataset = SLIMDataModule(data_dir="./data", batch_size=1).full_dataset

    
    # Initialize 3-fold cross validation
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    
    all_test_predictions = []
    all_test_ground_truths = []
    
    # Perform 3-fold cross-validation
    for fold, (train_idx, val_idx) in enumerate(kf.split(full_dataset)):
        print(f"Training fold {fold+1}/3...")
        
        # Clear CUDA cache between folds
        torch.cuda.empty_cache()
        
        # Create train and validation dataloaders for this fold
        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)
        
        train_dataloader = DataLoader(train_subset, batch_size=1, shuffle=True)
        val_dataloader = DataLoader(val_subset, batch_size=1, shuffle=False)
        
        model = GraphUnet(
            ks=[0.5, 0.5, 0.5],
            n_nodes=160,
            m_nodes=268,
            dim=16,
            act=torch.relu,
            drop_p=0.01,
        )
        model.to(torch.device("cuda:1"))

        train_losses, val_losses, lr, _ = train_model(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            num_epochs=100,
            lr=0.01,
            validate_every=1,
            patience=10,
            criterion=loss,
            intermediate_losses=True,
            skip=False,
        )

        model.eval()
        model.eval()
        gt_adj = []
        pred_adj = []
        with torch.no_grad():
            for inputs, targets in val_dataloader:
                inputs = inputs.squeeze(0)
                targets = targets.squeeze(0)

                # Forward pass on training data
                outputs, _, _ = model.forward(A=inputs, X=None)

                gt_adj.append(targets.detach().cpu().numpy())
                pred_adj.append(outputs.detach().cpu().numpy())

        print_metrics(gt_adj, pred_adj)
  