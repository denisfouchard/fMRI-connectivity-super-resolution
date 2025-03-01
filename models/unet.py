# putting it all together - code taken from https://github.com/HongyangGao/Graph-U-Nets/tree/master

import torch
import torch.nn.functional as F
import torch.nn as nn
import networkx as nx


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
    adj = F.relu(X_norm @ X_norm.T)  # Values in range [-1, 1]

    # adj = torch.sigmoid(adj)

    # Apply sparsification: Keep only values above threshold
    # adj = torch.where(adj > threshold, adj, torch.zeros_like(adj))

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
        self.upsample_mlp = nn.Linear(n_nodes, m_nodes)

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
        X_upsampled = torch.sigmoid(self.upsample_mlp(X.T).T)  # [num_nodes, in_dim]
        # Concatenate old and new nodes
        # X_upsampled = torch.cat([X, new_nodes], dim=0)

        A_upsampled = reconstruct_adjacency(X=X_upsampled)

        # print("Mean : ", A_upsampled.mean().item(), "Std :", A_upsampled.std().item())

        # Message passing to refine embeddings
        if refine:
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
            self.down_gcns.append(GCN(dim, out_dim, act, drop_p))
            self.up_gcns.append(GCN(out_dim, dim, act, drop_p))
            self.pools.append(Pool(k, out_dim, drop_p))
            self.unpools.append(Unpool(dim, dim, drop_p))
            dim = out_dim

        self.up_gcns = self.up_gcns[::-1]
        # self.node_features = nn.Parameter(torch.randn(n_nodes, dim))
        self.bottom_gcn = GCN(dim, dim, act, drop_p)

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
            X = torch.randn(A.shape[0], self.dim, device=self.device)
        else:
            X = X.to(self.device)

        A_history = []
        A_recon_history = []
        indices_list = []
        down_outs = []
        org_A = A.clone()
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

        A_upsampled = self.upsampler.forward(X, org_A)

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
    # A_pooled = A.bool().float()
    # A_pooled = (
    #    torch.matmul(A_pooled, A_pooled).bool().float()
    # )  # second power to reduce chance of isolated nodes
    A_pooled = A[idx, :]
    A_pooled = A_pooled[:, idx]
    # Peform thresholding
    # A_pooled = torch.where(
    #    A_pooled > 0.0, torch.ones_like(A_pooled), torch.zeros_like(A_pooled)
    # )
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
