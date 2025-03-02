import torch
import pytorch_lightning as pl
import pandas as pd
import os
import numpy as np
import time
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from typing import List, Tuple
from tqdm import tqdm

from utils.matrix_vectorizer import MatrixVectorizer


def calculate_topological_metrics(adj_matrix):
    num_nodes = adj_matrix.size(0)
    metrics = []

    # Degree (weighted sum of edges per node)
    degree = torch.sum(adj_matrix, dim=1)

    # Strength (same as degree for weighted graphs)
    strength = degree.clone()

    # Clustering Coefficient (local triangle clustering approximation)
    triangles = torch.diagonal(torch.matmul(adj_matrix, torch.matmul(adj_matrix, adj_matrix)))
    possible_triangles = degree * (degree - 1)
    clustering_coeff = torch.where(possible_triangles > 0, triangles / possible_triangles, torch.zeros_like(triangles))

    # Average Neighbor Degree (weighted neighbor degree average)
    neighbor_degrees = torch.matmul(adj_matrix, degree.unsqueeze(1)).squeeze(1)
    neighbor_counts = torch.sum((adj_matrix > 0).float(), dim=1)
    avg_neighbor_degree = torch.where(neighbor_counts > 0, neighbor_degrees / neighbor_counts, torch.zeros_like(neighbor_degrees))

    # Combine metrics into 2D tensor (num_nodes x 4)
    metrics = torch.stack([degree, strength, clustering_coeff, avg_neighbor_degree], dim=1)

    return metrics

def create_graph(adjacency_matrix) -> Data:
    """
    Convert an adjacency matrix to a PyG Data object.

    Parameters:
    -----------
    adjacency_matrix : torch.Tensor
        The input adjacency matrix
    target_adjacency_matrix : torch.Tensor
        The target adjacency matrix
    node_features : torch.Tensor, optional
        Node feature matrix

    Returns:
    --------
    data : torch_geometric.data.Data
        The PyG Data object containing the graph
    """

    # Get indices where adjacency_matrix > 0 using torch.where
    edge_indices = torch.where(adjacency_matrix > 0)

    # Create edge_index tensor directly from the indices
    edge_index = torch.stack([edge_indices[0], edge_indices[1]], dim=0)

    # Get edge attributes directly from adjacency matrix at those indices
    edge_attr = adjacency_matrix[edge_indices]

    # Set node features if provided, otherwise use ones
    x = calculate_topological_metrics(adjacency_matrix)

    # Create Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=adjacency_matrix.shape[0]
    )

    return data

def csv_to_tensor(file_path):
    start_time = time.time()
    df = pd.read_csv(file_path)
    tensor = torch.tensor(df.values, dtype=torch.float32)
    end_time = time.time()
    print(f"Time taken to load {file_path}: {end_time - start_time} seconds")
    return tensor


class UpscaledGraphDataLoader:
    def __init__(self, input_graphs, output_graphs, batch_size):
        assert len(input_graphs)==len(output_graphs)
        self.input_graphs = input_graphs
        self.output_graphs = output_graphs
        self.batch_size = batch_size
        self.indices = list(range(len(input_graphs))) # List of indices to shuffle
        self.shuffle()

    def shuffle(self):
        np.random.shuffle(self.indices)  # Shuffle the dataset indices

    def __iter__(self):
        for i in range(0, len(self.input_graphs), self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]  # Get batch indices
            input_graphs = Batch.from_data_list([self.input_graphs[idx] for idx in batch_indices])
            output_graphs = Batch.from_data_list([self.output_graphs[idx] for idx in batch_indices])
            yield input_graphs, output_graphs
        self.shuffle()

    def __len__(self):
        return len(self.input_graphs)//self.batch_size


class GraphDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for handling graph-structured data.

    This module manages loading, processing, and batching of graph data from high-resolution
    and low-resolution adjacency matrices. It supports automatic caching of processed tensors,
    training/validation splitting, and conversion between vector and matrix representations.

    Parameters
    ----------
    data_dir : str
        Directory containing the input data files (hr_train.csv, lr_train.csv, etc.)
    batch_size : int, default=32
        Number of samples per batch
    p_val : float, default=0.2
        Proportion of data to use for validation
    num_workers : int, default=10
        Number of subprocesses for data loading
    """
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        p_val: float = 0.2,
        num_workers: int = 10,
    ):
        super().__init__()
        self.num_workers = num_workers
        self.hr_train = csv_to_tensor(os.path.join(data_dir, "hr_train.csv"))
        self.lr_train = csv_to_tensor(os.path.join(data_dir, "lr_train.csv"))
        self.lr_test = csv_to_tensor(os.path.join(data_dir, "lr_test.csv"))

        self.batch_size = batch_size
        self.p_val = p_val

        # Shuffle and split the training data into training and validation sets
        num_train = len(self.lr_train)
        indices = torch.randperm(num_train)
        split = int(num_train * (1 - self.p_val))
        train_indices, val_indices = indices[:split], indices[split:]

        self.lr_train, self.lr_val = (
            self.lr_train[train_indices],
            self.lr_train[val_indices],
        )
        self.hr_train, self.hr_val = (
            self.hr_train[train_indices],
            self.hr_train[val_indices],
        )

        self.train_dataset = self._get_data(lr_data=self.lr_train, hr_data=self.hr_train)
        self.val_dataset = self._get_data(lr_data=self.lr_val, hr_data=self.hr_val)
        self.test_dataset = self._get_test_data(lr_data=self.lr_test)

    @staticmethod
    def _get_data(lr_data, hr_data, is_vector=True) -> Tuple[List[Data],List[Data]]:
        assert lr_data.shape[0] == hr_data.shape[0]
        lr_size = 160
        hr_size = 268

        lr_graphs = []
        hr_graphs = []
        if is_vector:
            progress_bar = tqdm(range(lr_data.shape[0]))
            progress_bar.set_description(f"Converting vectors to graphs")
            for i in progress_bar:
                lr_m = MatrixVectorizer.anti_vectorize(
                    vector=lr_data[i], matrix_size=lr_size
                )
                hr_m = MatrixVectorizer.anti_vectorize(
                    vector=hr_data[i], matrix_size=hr_size
                )

                lr_graph = create_graph(lr_m)
                hr_graph = create_graph(hr_m)

                lr_graphs.append(lr_graph)
                hr_graphs.append(hr_graph)
        return lr_graphs, hr_graphs

    @staticmethod
    def _get_test_data(lr_data, is_vector=True) -> List[Data]:
        lr_size = 160

        lr_graphs = []
        if is_vector:
            progress_bar = tqdm(range(lr_data.shape[0]))
            progress_bar.set_description(f"Converting vectors to graphs")
            for i in progress_bar:
                lr_m = MatrixVectorizer.anti_vectorize(
                    vector=lr_data[i], matrix_size=lr_size
                )
                lr_graph = create_graph(lr_m)
                lr_graphs.append(lr_graph)
        return lr_graphs

    def train_dataloader(self) -> UpscaledGraphDataLoader:
        return UpscaledGraphDataLoader(self.train_dataset[0], self.train_dataset[1], batch_size=self.batch_size)

    def val_dataloader(self) -> UpscaledGraphDataLoader:
        return UpscaledGraphDataLoader(self.val_dataset[0], self.val_dataset[1], batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


@torch.no_grad()
def save_prediction(model, dataloader, output_file):
    model.eval()

    preds = []
    for batch in dataloader:
        batch = batch.to(model.device)
        outputs = model(batch)
        preds.append(outputs.detach().cpu().numpy())

    # Vectorize matrices
    preds = [MatrixVectorizer.vectorize(p) for p in preds]
    preds = np.array(preds)

    # Submission format
    print(preds.shape)
    submission_df = pd.DataFrame(
        {"ID": range(1, len(preds.flatten()) + 1), "Predicted": preds.flatten()}
    )
    submission_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    data_module = GraphDataModule(data_dir="./data")
