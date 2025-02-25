import pandas as pd
import torch
import os


def csv_to_tensor(file_path):
    df = pd.read_csv(file_path)
    tensor = torch.tensor(df.values)
    return tensor


data_folder = "/Users/df/Documents/Imperial/DGL/DGL-Group-Project/data"
csv_files = [f for f in os.listdir(data_folder) if f.endswith(".csv")]

for csv_file in csv_files:
    file_path = os.path.join(data_folder, csv_file)
    tensor = csv_to_tensor(file_path)
    tensor_file_path = file_path.replace(".csv", ".pt")
    torch.save(tensor, tensor_file_path)
