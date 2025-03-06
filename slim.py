import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import pandas as pd
import os
from MatrixVectorizer import MatrixVectorizer
import numpy as np
import time
from tqdm import tqdm


def csv_to_tensor(file_path):
    start_time = time.time()
    df = pd.read_csv(file_path)
    tensor = torch.tensor(df.values, dtype=torch.float32)
    end_time = time.time()
    print(f"Time taken to load {file_path}: {end_time - start_time} seconds")
    return tensor


class SLIMDataset(Dataset):
    def __init__(self, lr_data, hr_data):
        assert lr_data.shape[0] == hr_data.shape[0]
        assert len(lr_data.shape) == 3
        assert len(hr_data.shape) == 3
        self.lr_data = lr_data
        self.hr_data = hr_data

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
        is_init = os.path.exists(hr_train_path) and os.path.exists(lr_train_path)
        if is_init:
            print("Loading data from disk")
            self.hr_train = torch.load(hr_train_path)
            self.lr_train = torch.load(lr_train_path)
        else:
            hr_train = csv_to_tensor(os.path.join(data_dir, "hr_train.csv"))
            lr_train = csv_to_tensor(os.path.join(data_dir, "lr_train.csv"))
            lr_size = 160
            hr_size = 268

            lr_mat = []
            hr_mat = []

            # Convert vectors to matrices
            print("Converting LR vectors to matrices")
            for i in tqdm(range(lr_train.shape[0])):
                lr_m = MatrixVectorizer.anti_vectorize(
                    vector=lr_train[i], matrix_size=lr_size
                )
                lr_mat.append(lr_m)

            print("Converting HR vectors to matrices")
            for i in tqdm(range(hr_train.shape[0])):
                hr_m = MatrixVectorizer.anti_vectorize(
                    vector=hr_train[i], matrix_size=hr_size
                )
                hr_mat.append(hr_m)

            self.lr_train = torch.tensor(np.array(lr_mat), dtype=torch.float32)
            self.hr_train = torch.tensor(np.array(hr_mat), dtype=torch.float32)

            # Save the tensors to disk
            print("Saving data to disk")
            torch.save(self.lr_train, lr_train_path)
            torch.save(self.hr_train, hr_train_path)

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

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=10
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=10)


class SLIMTestDataset(Dataset):
    def __init__(self, lr_data):
        self.lr_data = lr_data

    def __len__(self):
        return len(self.lr_data)

    def __getitem__(self, idx):
        return self.lr_data[idx]


def create_test_dataloader(data_dir, batch_size=32):
    lr_test_path = os.path.join(data_dir, "lr_test.pt")
    if os.path.exists(lr_test_path):
        lr_test = torch.load(lr_test_path)
    else:
        lr_test = csv_to_tensor(os.path.join(data_dir, "lr_test.csv"))

        # Convert vectors to matrices
        lr_mat = []
        lr_size = 160

        print("Converting LR vectors to matrices")
        for i in tqdm(range(lr_test.shape[0])):
            lr_m = MatrixVectorizer.anti_vectorize(
                vector=lr_test[i], matrix_size=lr_size
            )
            lr_mat.append(lr_m)

        lr_test = torch.tensor(np.array(lr_mat), dtype=torch.float32)

        # Save the tensor to disk
        torch.save(lr_test, lr_test_path)
        print("Saved LR test data to disk")

    test_dataset = SLIMTestDataset(lr_data=lr_test)
    return DataLoader(test_dataset, batch_size=batch_size)


if __name__ == "__main__":
    data_module = SLIMDataModule(data_dir="./data")
