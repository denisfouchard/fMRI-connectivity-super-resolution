import torch
import pytorch_lightning as pl
import pandas as pd
import os
import numpy as np
import time
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from typing import List
from tqdm import tqdm

from utils.matrix_vectorizer import MatrixVectorizer


def create_graph(adjacency_matrix, target_adjacency_matrix, node_features=None) -> Data:
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
    x = node_features if node_features is not None else torch.ones((adjacency_matrix.shape[0], 1))

    # Create Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=target_adjacency_matrix,
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
        hr_train_path = os.path.join(data_dir, "hr_train.pt")
        lr_train_path = os.path.join(data_dir, "lr_train.pt")
        is_init = os.path.exists(hr_train_path) and os.path.exists(lr_train_path)
        if is_init:
            print("Loading data from disk")
            self.hr_train = torch.load(hr_train_path)
            self.lr_train = torch.load(lr_train_path)
        else:
            self.hr_train = csv_to_tensor(os.path.join(data_dir, "hr_train.csv"))
            self.lr_train = csv_to_tensor(os.path.join(data_dir, "lr_train.csv"))
            # Save the tensors to disk

        self.batch_size = batch_size
        self.p_val = p_val

        if not is_init:
            torch.save(self.lr_train, lr_train_path)
            torch.save(self.hr_train, hr_train_path)

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


    def _get_data(self, lr_data, hr_data, is_vector=True) -> List[Data]:
        assert lr_data.shape[0] == hr_data.shape[0]
        lr_size = 160
        hr_size = 268

        data = []
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

                graph = create_graph(lr_m, hr_m)

                data.append(graph)
        return data


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


if __name__ == "__main__":
    data_module = GraphDataModule(data_dir="./data")
