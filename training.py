import torch
import copy
from tqdm import tqdm
from metrics import evaluation_metrics


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def train_model(
    model,
    train_dataloader,
    val_dataloader,
    num_epochs=100,
    lr=0.01,
    validate_every=1,
    patience=10,
    criterion=None,
):
    """
    Train the model, validate every 'validate_every' epochs, and pick the
    checkpoint with best validation accuracy.

    Parameters:
    -----------
    model : torch.nn.Module
        The PyTorch model to train.
    train_dataloader : torch.utils.data.DataLoader
        DataLoader for the training set.
    val_dataloader : torch.utils.data.DataLoader
        DataLoader for the validation set.
    num_epochs : int
        Number of training epochs.
    lr : float
        Learning rate for the optimizer.
    validate_every : int
        Validate (and possibly checkpoint) every 'validate_every' epochs.
    patience : int
        Patience for learning rate scheduler.
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=patience
    )
    train_loss_history = []
    val_loss_history = []

    best_val_loss = torch.inf
    best_model_state_dict = None
    val_loss = 0.0

    progress_bar = tqdm(range(num_epochs))
    for epoch in progress_bar:
        progress_bar.set_description(f"Epoch {epoch}|{num_epochs}")
        model.train()
        epoch_loss = 0.0

        for batch in train_dataloader:
            inputs, targets = batch
            optimizer.zero_grad()

            # Forward pass on training data
            outputs = model(inputs)
            loss = criterion(outputs, targets.to(model.device))
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Record training loss
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_dataloader)
        train_loss_history.append(avg_loss)

        # Validation step
        if (epoch + 1) % validate_every == 0 or (epoch + 1) == num_epochs:
            model.eval()
            val_loss = 0.0
            for batch in val_dataloader:
                inputs, targets = batch
                outputs = model(inputs)

                val_loss += criterion(outputs, targets.to(model.device)).item()

            val_loss /= len(val_dataloader)
            val_loss_history.append(val_loss)
            scheduler.step(val_loss)

            lr = get_lr(optimizer)

            # Check if this is the best f1 score so far
            if val_loss > best_val_loss:
                best_val_loss = val_loss
                best_model_state_dict = copy.deepcopy(model.state_dict())

            if lr < 1e-5:
                break

        progress_bar.set_postfix({"train_loss": avg_loss, "val_loss": val_loss})

    # If we have a best model, load it
    if best_model_state_dict is not None:
        model.load_state_dict(best_model_state_dict)

    return train_loss_history, val_loss_history, best_model_state_dict


@torch.no_grad()
def evaluate_model(model, dataloader, criterion):
    """
    Runs forward pass, calculates binary predictions (threshold=0.5),
    and returns the accuracy score.
    """
    model.eval()
    val_loss = 0.0
    eval_metrics = {
        "mae": 0,
        "pcc": 0,
        "js_dis": 0,
        "avg_mae_bc": 0,
        "avg_mae_ec": 0,
        "avg_mae_pc": 0,
    }

    for batch in dataloader:
        inputs, targets = batch
        outputs = model(inputs)

        val_loss += criterion(outputs.to(model.device)).item()
        batch_metrics = evaluation_metrics(
            outputs.detach().numpy(), targets.detach().numpy()
        )

        for k, v in batch_metrics.items():
            eval_metrics[k] += v

    val_loss /= len(dataloader)
    for v in eval_metrics.values():
        v /= len(dataloader)
    return val_loss, eval_metrics
