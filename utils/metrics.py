from utils.matrix_vectorizer import MatrixVectorizer

from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
from scipy.spatial.distance import jensenshannon
import networkx as nx
import numpy as np
import torch
from tqdm import tqdm
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import Data



def betweenness_centrality(adj, num_iter=10):
    """
    Approximate betweenness centrality using matrix powers.
    """
    # Add self-loops to avoid zero rows
    adj = adj + torch.eye(adj.size(0), device=adj.device)
    
    # Matrix powers to approximate shortest paths
    dist = torch.linalg.matrix_power(adj, num_iter)
    
    # Normalize and compute centrality
    centrality = dist.sum(dim=1) / dist.sum()
    return centrality.to(adj.device)

def eigenvector_centrality(adj, num_iter=100):
    """
    Eigenvector centrality via power iteration.
    """
    # Add self-loops to ensure connectivity
    adj = adj + torch.eye(adj.size(0), device=adj.device)
    
    # Initialize eigenvector
    x = torch.ones(adj.size(0), 1, device=adj.device)
    
    for _ in range(num_iter):
        x = adj @ x
        x = x / x.norm()  # Normalize
    
    return x.squeeze().to(adj.device)

def pagerank(adj, alpha=0.85, num_iter=100):
    """
    Differentiable PageRank.
    """
    # Row-normalize the adjacency matrix
    adj = adj / adj.sum(dim=1, keepdim=True).clamp(min=1e-9)
    
    # Initialize ranks
    teleport = torch.ones(adj.size(0), device=adj.device) / adj.size(0)
    rank = teleport.clone()
    
    # Power iteration
    for _ in range(num_iter):
        rank = alpha * (adj.T @ rank) + (1 - alpha) * teleport
        
    return rank.to(adj.device)


def get_adj(data):
    return to_dense_adj(data.edge_index, edge_attr=data.edge_attr, batch=data.batch)


class GSRLoss:
    def __init__(self, device):
        self.criterion = torch.nn.L1Loss()
        self.device = device
    
    def __call__(self, input_adj, target_adj):
        loss = torch.zeros(1).to(self.device)
        input_adj = input_adj.to(self.device)
        target_adj = target_adj.to(self.device)
        k = 4
        for i in range(input_adj.shape[0]):
            loss += (1/k) * self.criterion(betweenness_centrality(input_adj[i]),betweenness_centrality(target_adj[i])).to(self.device)
            loss += (1/k) * self.criterion(eigenvector_centrality(input_adj[i]),eigenvector_centrality(target_adj[i])).to(self.device)
            loss += (1/k) * self.criterion(pagerank(input_adj[i]),pagerank(target_adj[i])).to(self.device)
            loss += (1/k) * self.criterion(input_adj[i],target_adj[i]).to(self.device)
        return loss/input_adj.shape[0]


@torch.no_grad()
def evaluate_model(model, dataloader):
    """
    Runs forward pass, calculates binary predictions (threshold=0.5),
    and returns the accuracy score.
    """
    model.eval()

    preds = []
    true = []
    for (batch,target_batch) in dataloader:
        batch = batch.to(model.device)
        targets = to_dense_adj(target_batch.edge_index, edge_attr=target_batch.edge_attr, batch=target_batch.batch)

        outputs = model(batch)
        
        outputs = outputs.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        
        for i in range(len(outputs)):
            preds.append(outputs[i])
            true.append(targets[i])

    preds = np.array(preds)
    true = np.array(true)

    return np.mean(np.abs(preds-true))

