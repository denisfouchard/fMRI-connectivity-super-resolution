from torch_geometric.loader import DataLoader
import pandas as pd
import torch
import copy
from tqdm import tqdm
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj


# Assuming metrics.py is adapted for PyG data format
# from metrics import evaluation_metrics


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def train_model(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    num_epochs=100,
    lr=0.01,
    validate_every=1,
    optimizer=None,
    lr_scheduler=None,
    criterion=None,
    skip=False,
    device="cpu",
):
    """
    Train the model, validate every 'validate_every' epochs, and pick the
    checkpoint with best validation accuracy.

    Parameters:
    -----------
    model : torch.nn.Module
        The PyTorch Geometric model to train.
    train_dataloader : torch_geometric.loader.DataLoader
        DataLoader for the training set.
    val_dataloader : torch_geometric.loader.DataLoader
        DataLoader for the validation set.
    num_epochs : int
        Number of training epochs.
    lr : float
        Learning rate for the optimizer.
    validate_every : int
        Validate (and possibly checkpoint) every 'validate_every' epochs.
    optimizer
        Optimizer used to train the model.
    lr_scheduler :
        Learning rate scheduler used to train the model.
    criterion : torch.nn.Module
        Loss function.

    Returns:
    --------
    best_loss_history : list
        The training loss history across epochs.
    best_model_state_dict : dict
        The state dictionary of the model achieving the best validation accuracy.
    """

    model.train()
    if not optimizer:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    if not lr_scheduler:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=10
        )
    train_loss_history = []
    val_loss_history = []
    lr_history = []

    best_val_loss = float('inf')
    best_model_state_dict = None
    val_loss = 0.0

    progress_bar = tqdm(range(num_epochs))
    for epoch in progress_bar:
        model.train()
        epoch_loss = 0.0

        for (batch,target_batch) in train_dataloader:
            batch = batch.to(model.device)
            target_batch = target_batch.to(model.device)
            optimizer.zero_grad()

            # Forward pass on training data
            outputs = model(batch)

            # Assuming y contains the target adjacency information
            targets = to_dense_adj(target_batch.edge_index, edge_attr=target_batch.edge_attr, batch=target_batch.batch)

            loss = criterion(
                outputs,
                targets
            )
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Record training loss
            epoch_loss += loss.item()
            progress_bar.set_description(f"Epoch loss {loss.item()}")

        avg_loss = epoch_loss / len(train_dataloader)
        train_loss_history.append(avg_loss)

        # Validation step
        if (epoch + 1) % validate_every == 0 or (epoch + 1) == num_epochs:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for (batch,target_batch) in val_dataloader:
                    batch = batch.to(model.device)
                    target_batch = target_batch.to(model.device)

                    outputs = model(batch)

                    targets = to_dense_adj(target_batch.edge_index, edge_attr=target_batch.edge_attr, batch=target_batch.batch)

                    val_loss += criterion(
                        outputs,
                        targets
                    ).item()

            val_loss /= len(val_dataloader)
            val_loss_history.append(val_loss)
            lr_scheduler.step(val_loss)

            lr = get_lr(optimizer)
            lr_history.append(lr)

            # Check if this is the best validation loss so far
            # Note: changed from > to < since we're minimizing loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state_dict = copy.deepcopy(model.state_dict())

            if lr < 1e-5:
                break

        progress_bar.set_postfix(
            {"train_loss": avg_loss, "val_loss": val_loss, "lr": lr}
        )

    # If we have a best model, load it
    if best_model_state_dict is not None:
        model.load_state_dict(best_model_state_dict)

    return train_loss_history, val_loss_history, lr_history, best_model_state_dict

