import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import pandas as pd
import os
from MatrixVectorizer import MatrixVectorizer
import numpy as np


def csv_to_tensor(file_path):
    df = pd.read_csv(file_path)
    tensor = torch.tensor(df.values, dtype=torch.float32)
    return tensor


class SLIMDataset(Dataset):
    def __init__(self, lr_data, hr_data):
        assert lr_data.shape[0] == hr_data.shape[0]
        lr_mat = []
        hr_mat = []

        lr_size = int(np.sqrt(lr_data.shape[1]))
        hr_size = int(np.sqrt(hr_data.shape[1]))

        for i in range(lr_data.shape[0]):
            lr_m = MatrixVectorizer.anti_vectorize(
                vector=lr_data[i], matrix_size=lr_size
            )
            hr_m = MatrixVectorizer.anti_vectorize(
                vector=hr_data[i], matrix_size=hr_size
            )
            lr_mat.append(lr_m)
            hr_mat.append(hr_m)

        self.lr_data = torch.tensor(np.array(lr_mat), dtype=torch.float32)
        self.hr_data = torch.tensor(np.array(hr_mat), dtype=torch.float32)

    def __len__(self):
        return len(self.lr_data)

    def __getitem__(self, idx):
        return self.lr_data[idx], self.hr_data[idx]


class SLIMDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        p_val: float = 0.2,
        test: bool = False,
    ):
        super().__init__()
        hr_train_path = os.path.join(data_dir, "hr_train.pt")
        lr_train_path = os.path.join(data_dir, "lr_train.pt")
        if os.path.exists(hr_train_path) and os.path.exists(lr_train_path):
            print("Lol")
            self.hr_train = torch.load(hr_train_path)
            self.lr_train = torch.load(lr_train_path)
        else:
            self.hr_train = csv_to_tensor(os.path.join(data_dir, "hr_train.csv"))
            self.lr_train = csv_to_tensor(os.path.join(data_dir, "lr_train.csv"))

        if test:
            lr_test_path = os.path.join(data_dir, "lr_test.pt")
            hr_test_path = os.path.join(data_dir, "hr_test.pt")
            if os.path.exists(lr_test_path) and os.path.exists(hr_test_path):
                self.lr_test = torch.load(lr_test_path)
                self.hr_test = torch.load(hr_test_path)
            else:
                self.lr_test = csv_to_tensor(os.path.join(data_dir, "lr_test.csv"))
                self.hr_test = csv_to_tensor(os.path.join(data_dir, "hr_test.csv"))

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

        self.train_dataset = SLIMDataset(lr_data=self.lr_train, hr_data=self.hr_train)
        self.val_dataset = SLIMDataset(lr_data=self.lr_val, hr_data=self.hr_val)
        if test:
            self.test_dataset = SLIMDataset(lr_data=self.lr_test, hr_data=self.hr_test)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


if __name__ == "__main__":
    data_module = SLIMDataModule(data_dir="./data")
