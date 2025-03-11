import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils.initializations import *
from utils.preprocessing import normalize_adj_torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GSRLayer(nn.Module):
    def __init__(self, hr_dim, lr_dim):
        super(GSRLayer, self).__init__()
        self.hr_dim = hr_dim  # e.g., 268
        self.lr_dim = lr_dim  # e.g., 160

        # Initialize weights with shape (hr_dim, lr_dim)
        self.weights = torch.nn.Parameter(torch.randn(hr_dim, lr_dim))

    def forward(self, A, X):
        # A: LR adjacency matrix (160x160)
        # X: feature matrix from GraphUnet, shape (lr_dim, feature_dim) e.g., (160, d)
        lr = A
        f = X

        # Eigen-decomposition of the LR adjacency matrix
        eig_val_lr, U_lr = torch.linalg.eigh(lr, UPLO='U')  # U_lr shape: (160,160)

        # Create an identity matrix of size lr_dim (160x160)
        eye_mat = torch.eye(self.lr_dim).to(device)

        # Create s_d by stacking parts of the identity to map from LR to HR
        # Original: s_d = cat(eye_mat, eye_mat[:(hr_dim - lr_dim)], 0)
        s_d = torch.cat((eye_mat, eye_mat[:(self.hr_dim - self.lr_dim)]), 0)  # shape: (268, 160)

        # Multiply weights with the transpose of s_d to project to HR space
        a = torch.matmul(self.weights, s_d.T)  # (268, 160) x (160, 268) -> (268, 268)

        # Pad U_lr to shape (hr_dim, hr_dim) = (268,268)
        pad_size_row = self.hr_dim - self.lr_dim  # 268-160 = 108
        pad_size_col = self.hr_dim - self.lr_dim  # 108
        U_lr_padded = torch.nn.functional.pad(U_lr, (0, pad_size_col, 0, pad_size_row))  # (268, 268)

        # Multiply with the transpose of the padded U_lr
        b = torch.matmul(a, torch.t(U_lr_padded))  # (268,268) x (268,268) -> (268,268)

        # Pad feature matrix f from (160, feature_dim) to (268, feature_dim)
        if f.shape[0] < self.hr_dim:
            f_padded = torch.nn.functional.pad(f, (0, 0, 0, self.hr_dim - f.shape[0]))
        else:
            f_padded = f

        # Multiply b with the padded feature matrix
        f_d = torch.matmul(b, f_padded)  # (268,268) x (268, feature_dim) -> (268, feature_dim)
        f_d = torch.abs(f_d)
        self.f_d = f_d.fill_diagonal_(1)

        adj = normalize_adj_torch(self.f_d)
        X_out = torch.mm(adj, adj.t())
        X_out = (X_out + X_out.t()) / 2
        idx = torch.eye(self.hr_dim, dtype=bool).to(device)
        X_out[idx] = 1
        return adj, torch.abs(X_out)


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    #160x320 320x320 =  160x320
    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = torch.nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        # input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        # output = self.act(output)
        return output