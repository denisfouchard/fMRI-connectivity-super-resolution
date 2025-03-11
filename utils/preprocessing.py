import torch
import numpy as np
import os
import scipy.io 
import pandas as pd
from utils.matrix_vectorizer import MatrixVectorizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pad_HR_adj(label, split):
    label = label.cpu().numpy()  # Move to CPU before NumPy operations
    label = np.pad(label, ((split, split), (split, split)), mode="constant")
    np.fill_diagonal(label, 1)
    return torch.from_numpy(label).float().to(device)  # Move back to CUDA if needed

def normalize_adj_torch(mx):
    # mx = mx.to_dense()
    rowsum = mx.sum(1)
    r_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
    mx = torch.matmul(mx, r_mat_inv_sqrt)
    mx = torch.transpose(mx, 0, 1)
    mx = torch.matmul(mx, r_mat_inv_sqrt)
    return mx

def unpad(data, split):
  
  idx_0 = data.shape[0]-split
  idx_1 = data.shape[1]-split
  # print(idx_0,idx_1)
  train = data[split:idx_0, split:idx_1]
  return train

def extract_data(csv_path, is_hr=True):
    """
    Loads brain connectivity matrices from a CSV file and preprocesses them.

    Parameters:
    -----------
    csv_path : str
        Path to the CSV file containing adjacency matrices (vectorized format).
    is_hr : bool, optional
        If True, the function loads a high-resolution (HR) matrix.
        If False, it loads a low-resolution (LR) matrix.
    
    Returns:
    --------
    numpy.ndarray
        Preprocessed adjacency matrices in (N, size, size) format.
    """
    # Load CSV file into a DataFrame
    df = pd.read_csv(csv_path)

    # Drop the first column (indexing or ID column if present)
    if df.columns[0] == "Unnamed: 0" or df.columns[0] == "ID":
        df = df.iloc[:, 1:]

    # Convert DataFrame to NumPy array
    data = df.to_numpy()

    # Replace NaN values with 0 (as per the dataset preprocessing rules)
    data = np.nan_to_num(data, nan=0.0)

    # Define matrix size based on HR or LR
    matrix_size = 268 if is_hr else 160

    # Convert vectorized adjacency matrices into 2D matrices
    num_samples = data.shape[0]
    adjacency_matrices = np.zeros((num_samples, matrix_size, matrix_size))

    for i in range(num_samples):
        adjacency_matrices[i] = MatrixVectorizer.anti_vectorize(data[i], matrix_size)

    return adjacency_matrices

def load_data():
    """
    Loads the LR and HR training data and test data from CSV files.

    Returns:
    --------
    subjects_adj : np.ndarray
        Low-resolution (LR) adjacency matrices, shape (N, 160, 160)
    subjects_labels : np.ndarray
        High-resolution (HR) adjacency matrices, shape (N, 268, 268)
    test_adj : np.ndarray
        Low-resolution (LR) test matrices, shape (M, 160, 160)
    """
    # Load LR and HR training data
    subjects_adj = extract_data("data/lr_train.csv", is_hr=False)  # Shape (167, 160, 160)
    subjects_labels = extract_data("data/hr_train.csv", is_hr=True)  # Shape (167, 268, 268)

    # Load LR test data (no ground truth available)
    test_adj = extract_data("data/lr_test.csv", is_hr=False)  # Shape (112, 160, 160)

    return subjects_adj, subjects_labels, test_adj

def data():
    """
    Wrapper function to load all data in the correct format.

    Returns:
    --------
    subjects_adj : np.ndarray
        Training low-resolution (LR) matrices
    subjects_labels : np.ndarray
        Training high-resolution (HR) matrices
    test_adj : np.ndarray
        Test low-resolution (LR) matrices
    """
    subjects_adj, subjects_labels, test_adj = load_data()
    return subjects_adj, subjects_labels, test_adj
