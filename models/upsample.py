from torch_geometric.nn.conv import SAGEConv, GCNConv
import torch.nn as nn
import torch

num_nodes = 1000  # Adjust based on your dataset
embedding_dim = 128  # Dimension of node embeddings

# Trainable node embeddings
node_embeddings = nn.Embedding(num_nodes, embedding_dim)


class GCNLayer(nn.Module):
    """
    A single layer of a Graph Convolutional Network (GCN).
    """

    def __init__(self, input_dim, output_dim, use_nonlinearity=True):
        super(GCNLayer, self).__init__()
        self.use_nonlinearity = use_nonlinearity
        self.Omega = nn.Parameter(
            torch.randn(input_dim, output_dim)
            * torch.sqrt(torch.tensor(2.0) / (input_dim + output_dim))
        )
        self.beta = nn.Parameter(torch.zeros(output_dim))

    def forward(self, A_normalized, H_k):
        agg = torch.matmul(A_normalized, H_k)  # local agg
        H_k_next = torch.matmul(agg, self.Omega) + self.beta
        return nn.functional.relu(H_k_next) if self.use_nonlinearity else H_k_next


# GraphSAGE model
class GraphSAGE(nn.Module):
    def __init__(self, hidden_channels, out_size, n_layers: int = 2):
        super().__init__()
        self.in_channels = hidden_channels
        self.out_size = out_size
        self.conv = nn.ModuleList()
        for _ in range(n_layers - 1):
            self.conv.append(
                GCNLayer(input_dim=hidden_channels, output_dim=hidden_channels)
            )
        self.conv.append(
            GCNLayer(
                input_dim=hidden_channels,
                output_dim=hidden_channels,
                use_nonlinearity=False,
            )
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, A):
        A = A.to(self.device)
        X = torch.ones(
            A.shape[0],
            A.shape[1],
            self.in_channels,
            dtype=torch.float32,
            device=self.device,
        )
        for layer in self.conv:
            X = layer(A, X)
        X = X.permute(0, 2, 1)
        X = torch.nn.functional.interpolate(
            X, size=(self.out_size,), mode="linear"
        ).squeeze(0)
        X = X.permute(0, 2, 1)
        A_pred = torch.zeros(
            A.shape[0],
            self.out_size,
            self.out_size,
            dtype=torch.float32,
            device=self.device,
        )
        for i, x in enumerate(X):
            A_pred[i] = torch.sigmoid(x @ x.T)
        A_pred = A_pred * (
            A_pred > 0.2
        )  # Thresholding to preserve sparse brain connectivity
        return A_pred
