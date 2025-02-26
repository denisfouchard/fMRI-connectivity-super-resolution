# putting it all together - code taken from https://github.com/HongyangGao/Graph-U-Nets/tree/master

import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.nn as nn
import numpy as np


def reconstruct_adjacency(X, threshold=0.15):
    """
    Reconstruct adjacency from node embeddings while preserving fMRI-like structure.

    Args:
        X (torch.Tensor): Node embeddings of shape [num_nodes, hidden_dim]
        threshold (float): Value below which edges are removed for sparsity

    Returns:
        adj (torch.Tensor): Reconstructed weighted adjacency matrix
    """
    # Normalize embeddings to unit length (cosine similarity instead of raw dot product)
    X_norm = F.normalize(X, p=2, dim=1)  # [num_nodes, hidden_dim]
    # Compute cosine similarity matrix
    adj = X_norm @ X_norm.T  # Values in range [-1, 1]

    # adj = torch.sigmoid(adj)

    # Apply sparsification: Keep only values above threshold
    adj = torch.where(adj > threshold, adj, torch.zeros_like(adj))

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

        # Message passing layers
        self.conv1 = GCN(in_dim, hidden_dim, act, drop_p)
        self.conv2 = GCN(hidden_dim, hidden_dim, act, drop_p)

        # MLP for new node generation
        self.upsample_mlp = nn.Linear(n_nodes, m_nodes - n_nodes)

    def forward(self, X, A):
        """
        Args:
        - x: Node features [num_nodes, in_dim]
        - adj_matrix: Initial adjacency matrix [num_nodes, num_nodes]

        Returns:
        - Upsampled adjacency matrix [self.m_nodes, self.m_nodes]
        - Upsampled node features [new_num_nodes, in_dim]
        """

        # Generate new nodes by transforming existing ones
        new_nodes = torch.sigmoid(self.upsample_mlp(X.T).T)  # [num_nodes, in_dim]
        # Concatenate old and new nodes
        X_upsampled = torch.cat([X, new_nodes], dim=0)

        A_upsampled = reconstruct_adjacency(X=X_upsampled)

        # print("Mean : ", A_upsampled.mean().item(), "Std :", A_upsampled.std().item())

        # Message passing to refine embeddings
        for _ in range(self.num_iterations):
            X_upsampled = self.conv1(A_upsampled, X_upsampled)
            X_upsampled = F.relu(X_upsampled)
            A_upsampled = reconstruct_adjacency(X_upsampled)

            X_upsampled = self.conv2(A_upsampled, X_upsampled)
            X_upsampled = F.relu(X_upsampled)

            # Reconstruct adjacency with updated embeddings
            A_upsampled = reconstruct_adjacency(A_upsampled)

        return A_upsampled


class GraphUnet(nn.Module):

    def __init__(self, ks, n_nodes, m_nodes, dim, act, drop_p):
        super(GraphUnet, self).__init__()
        self.ks = ks
        self.dim = dim
        self.bottom_gcn = GCN(dim, dim, act, drop_p)
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
        for i in range(self.l_n):
            self.down_gcns.append(GCN(dim, dim, act, drop_p))
            self.up_gcns.append(GCN(dim, dim, act, drop_p))
            self.pools.append(Pool(ks[i], dim, drop_p))
            self.unpools.append(Unpool(dim, dim, drop_p))

        self.node_features = nn.Parameter(torch.randn(n_nodes, dim))

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, A: torch.Tensor):
        # Process A
        A = A.squeeze(0)
        A = torch.where(A > 0.2, A, torch.zeros_like(A))
        A = A + torch.eye(A.shape[0])
        A = symmetric_normalize(A)
        A = A.to(self.device)

        X = self.node_features
        adj_ms = []
        indices_list = []
        down_outs = []
        org_A = torch.tensor(A)
        org_X = torch.tensor(X)
        for i in range(self.l_n):
            X = self.down_gcns[i](A, X)
            adj_ms.append(A)
            down_outs.append(X)
            A, X, idx = self.pools[i](A, X)
            indices_list.append(idx)

        X = self.bottom_gcn(A, X)
        for i in range(self.l_n):
            up_idx = self.l_n - i - 1
            A, idx = adj_ms[up_idx], indices_list[up_idx]
            A, X = self.unpools[i](A, X, down_outs[up_idx], idx)
            X = self.up_gcns[i](A, X)
            # X = X.add(down_outs[up_idx])

        # X = X.add(org_X)

        A_upsampled = self.upsampler.forward(X, org_A)
        return A_upsampled


class GCN(nn.Module):

    def __init__(self, in_dim, out_dim, act, p):
        super(GCN, self).__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.act = act
        self.drop = nn.Dropout(p=p) if p > 0.0 else nn.Identity()

    def forward(self, g, h):
        h = self.drop(h)  # they have added dropout
        h = torch.matmul(g, h)
        h = self.proj(h)
        h = self.act(h)
        return h


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
    A_pooled = A.bool().float()
    A_pooled = (
        torch.matmul(A_pooled, A_pooled).bool().float()
    )  # second power to reduce chance of isolated nodes
    A_pooled = A_pooled[idx, :]
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
