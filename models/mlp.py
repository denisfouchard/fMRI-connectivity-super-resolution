import torch
import torch.nn as nn
import numpy as np


class SuperResMLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super().__init__()
        h = int(np.sqrt(output_size))
        self.layers = nn.ModuleList(
            [
                nn.Flatten(),
                nn.Linear(in_features=input_size, out_features=hidden_dim),
                nn.BatchNorm1d(num_features=hidden_dim),
                nn.Dropout(p=0.1),
                nn.ReLU(),
            ]
        )
        for _ in range(n_layers - 1):
            self.layers.append(
                nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
            )
            self.layers.append(nn.BatchNorm1d(num_features=hidden_dim))
            self.layers.append(nn.Dropout(p=0.1))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(in_features=hidden_dim, out_features=output_size))
        self.layers.append(nn.Unflatten(dim=1, unflattened_size=(h, h)))

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, samples: torch.Tensor):
        # Flatten the input if it's a 2D matrix
        x = samples.to(self.device)
        for layer in self.layers:
            x = layer(x)

        return x
