"""Scaffold for PyTorch training loop utilities."""
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np

from tqdm import tqdm


def make_loader(X, y, batch_size: int, shuffle: bool) -> DataLoader:
    assert y.ndim == 2 and y.shape[1] == 1, 'y must be selected as df[["target"]], with shape (n_samples, 1)'
    X_tensor = torch.from_numpy(X.to_numpy(dtype=np.float32).copy())
    y_tensor = torch.from_numpy(y.to_numpy(dtype=np.float32).copy())

    return DataLoader(
        TensorDataset(X_tensor, y_tensor),
        batch_size=batch_size,
        shuffle=shuffle,
    )

def train_one_epoch(
    model,
    optimizer,
    loss_fn,
    train_loader,
    epoch,
    epochs
) -> float:
    """
    Returns: The avarage train loss over all batches
    """
    progress = tqdm(
        train_loader,
        desc=f"train {epoch:03d}/{epochs:03d}",
        unit="batch",
        leave=False,
        position=1,
        dynamic_ncols=True,
        mininterval=0.2,
    )

    model.train()
    running_loss = 0
    sample_cnt = 0
    for X, y in train_loader:
        # Basically we want to start each batch as if the previous parameters where some random
        # start parameters and we don't know anyhting yet
        optimizer.zero_grad()
        pred = model(X)
        
        loss = loss_fn(pred, y)
        
        loss.backward()
        optimizer.step()

        batch_size = X.size(0)
        sample_cnt += batch_size
        running_loss += loss.item() * batch_size # multiplying because on last iteration batch size can differ
        
    return running_loss / sample_cnt


def validate_one_epoch(
    model,
    loss_fn,
    val_loader,
    epoch,
    epochs
) -> float:
    progress = tqdm(
        val_loader,
        desc=f"val {epoch:03d}/{epochs:03d}",
        unit="batch",
        leave=False,
        position=1,
        dynamic_ncols=True,
        mininterval=0.2,
    )
    model.eval()
    running_loss = 0
    sample_cnt = 0
    with torch.no_grad():
        for X, y in progress:
                pred = model(X)
                loss = loss_fn(pred, y)
                
                batch_size = X.size(0)
                sample_cnt += batch_size
                running_loss += loss.item() * batch_size
    return running_loss / sample_cnt 


def train_once(
    model,
    optimizer,
    loss_fn,
    train_loader,
    val_loader,
    epochs,
    run_name
) -> dict[str, list[float]]:

    epoch_progress = tqdm(
        range(1, epochs + 1),
        desc=run_name,
        unit="epoch",
        position=0,
        dynamic_ncols=True,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
    }
    for epoch in epoch_progress:
        train_loss = train_one_epoch(model, optimizer, loss_fn, train_loader, epoch, epochs)
        val_loss = validate_one_epoch(model, loss_fn, val_loader, epoch, epochs)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
    return history