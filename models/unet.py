from typing import Callable, List, Union

import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, TopKPooling
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import OptTensor, PairTensor
from torch_geometric.utils import (
    add_self_loops,
    remove_self_loops,
    to_torch_csr_tensor,
)
from torch_geometric.utils.repeat import repeat

from torch_geometric.utils import dense_to_sparse, to_dense_adj


class GraphUpsampler(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        n_nodes,
        m_nodes,
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
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # MLP for new node generation
        self.upsample_mlp = nn.Linear(n_nodes, m_nodes)

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
        new_nodes = self.upsample_mlp(X.T).T  # [num_nodes, in_dim]

        # Concatenate old and new nodes
        X_upsampled = torch.cat([X, new_nodes], dim=0)

        # Expand adjacency matrix (initialize new connections)
        A_upsampled = torch.zeros(self.m_nodes, self.m_nodes, device=X.device)
        A_upsampled[: self.n_nodes, : self.n_nodes] = A  # Copy old structure

        # Connect new nodes to their source nodes
        A_upsampled[self.n_nodes :, : self.n_nodes] = A  # Link new nodes to originals
        A_upsampled[: self.n_nodes, self.n_nodes :] = A.T  # Symmetrize
        A_upsampled[self.n_nodes :, self.n_nodes :] = torch.sigmoid(
            torch.matmul(new_nodes, new_nodes.T)
        )  # Self-connections

        # Convert to edge list
        edge_index, edge_weight = dense_to_sparse(A_upsampled)

        # Message passing to refine embeddings
        for _ in range(self.num_iterations):
            X_upsampled = self.conv1(X_upsampled, edge_index, edge_weight)
            X_upsampled = F.relu(X_upsampled)
            A = torch.sigmoid(X_upsampled @ X_upsampled.T)
            edge_index, edge_weight = dense_to_sparse(A)

            X_upsampled = self.conv2(X_upsampled, edge_index, edge_weight)
            X_upsampled = F.relu(X_upsampled)

            # Reconstruct adjacency with updated embeddings
            A_upsampled = torch.sigmoid(torch.matmul(X_upsampled, X_upsampled.T))

        return A_upsampled


class GraphUNet(torch.nn.Module):
    r"""The Graph U-Net model from the `"Graph U-Nets"
    <https://arxiv.org/abs/1905.05178>`_ paper which implements a U-Net like
    architecture with graph pooling and unpooling operations.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        out_channels (int): Size of each output sample.
        depth (int): The depth of the U-Net architecture.
        pool_ratios (float or [float], optional): Graph pooling ratio for each
            depth. (default: :obj:`0.5`)
        sum_res (bool, optional): If set to :obj:`False`, will use
            concatenation for integration of skip connections instead
            summation. (default: :obj:`True`)
        act (torch.nn.functional, optional): The nonlinearity to use.
            (default: :obj:`torch.nn.functional.relu`)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        in_size: int,
        out_size: int,
        depth: int,
        pool_ratios: Union[float, List[float]] = 0.5,
        sum_res: bool = True,
        act: Union[str, Callable] = "relu",
        init_features: str = "ones",
    ):
        super().__init__()
        assert depth >= 1
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.depth = depth
        self.pool_ratios = repeat(pool_ratios, depth)
        self.act = activation_resolver(act)
        self.sum_res = sum_res
        self.init_features = init_features
        channels = hidden_channels

        self.down_convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.down_convs.append(GCNConv(in_channels, channels, improved=True))
        for i in range(depth):
            self.pools.append(TopKPooling(channels, self.pool_ratios[i]))
            self.down_convs.append(GCNConv(channels, channels, improved=True))

        in_channels = channels if sum_res else 2 * channels

        self.up_convs = torch.nn.ModuleList()
        for i in range(depth - 1):
            self.up_convs.append(GCNConv(in_channels, channels, improved=True))
        self.up_convs.append(GCNConv(in_channels, out_channels, improved=True))

        self.upsampler = GraphUpsampler(
            in_dim=hidden_channels,
            hidden_dim=hidden_channels,
            n_nodes=in_size,
            m_nodes=out_size,
        )
        self.reset_parameters()

    @property
    def device(self):
        return next(self.parameters()).device

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for conv in self.down_convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        for conv in self.up_convs:
            conv.reset_parameters()

    def forward(
        self,
        A: Tensor,
        batch: OptTensor = None,
    ) -> Tensor:
        """"""  # noqa: D419
        A = A.squeeze(0)
        A = A.to(self.device)
        print("A shape", A.shape)
        edge_index, edge_weight = dense_to_sparse(A)
        print("edge shape", edge_index.shape)

        if self.init_features == "ones":
            X = torch.ones(A.shape[0], A.shape[1], self.in_channels, device=self.device)

        if batch is None:
            batch = A.new_zeros(X.size(0))

        X = self.down_convs[0](X, edge_index, edge_weight)
        X = self.act(X)

        xs = [X]
        edge_indices = [A]
        edge_weights = [edge_weight]
        perms = []

        for i in range(1, self.depth + 1):
            edge_index, edge_weight = self.augment_adj(
                edge_index, edge_weight, X.size(0)
            )
            X, edge_index, edge_weight, batch, perm, _ = self.pools[i - 1](
                X, edge_index, edge_weight, batch
            )

            X = self.down_convs[i](X, edge_index, edge_weight)
            X = self.act(X)

            if i < self.depth:
                xs += [X]
                edge_indices += [A]
                edge_weights += [edge_weight]
            perms += [perm]

        for i in range(self.depth):
            j = self.depth - 1 - i

            res = xs[j]
            A = edge_indices[j]
            edge_weight = edge_weights[j]
            perm = perms[j]

            up = torch.zeros_like(res)
            up[perm] = X
            X = res + up if self.sum_res else torch.cat((res, up), dim=-1)

            X = self.up_convs[i](X, edge_index, edge_weight)
            X = self.act(X) if i < self.depth - 1 else X

        A_upsampled = self.upsampler.forward(X, A)
        return A_upsampled

    def augment_adj(
        self, edge_index: Tensor, edge_weight: Tensor, num_nodes: int
    ) -> PairTensor:
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight = add_self_loops(
            edge_index, edge_weight, num_nodes=num_nodes
        )
        adj = to_dense_adj(edge_index=edge_index, edge_attr=edge_weight)

        edge_index, edge_weight = dense_to_sparse(adj @ adj)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"{self.hidden_channels}, {self.out_channels}, "
            f"depth={self.depth}, pool_ratios={self.pool_ratios})"
        )
