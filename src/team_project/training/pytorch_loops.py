"""Scaffold for PyTorch training loop utilities."""
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np


def make_loader(X, y, batch_size: int, shuffle: bool) -> DataLoader:
    assert y.ndim == 2 and y.shape[1] == 1, 'y must be selected as df[["target"]], with shape (n_samples, 1)'
    X_tensor = torch.from_numpy(X.to_numpy(dtype=np.float32))
    y_tensor = torch.from_numpy(y.to_numpy(dtype=np.float32))

    return DataLoader(
        TensorDataset(X_tensor, y_tensor),
        batch_size=batch_size,
        shuffle=shuffle,
    )

def train_one_epoch(
    model,
    optimizer,
    loss_fn,
    train_loader
) -> float:
    """
    Returns: The avarage train loss over all batches
    """

    model.train()
    running_loss = 0
    sample_cnt = 0
    for X, y in train_loader:
        # Basically we want to start each batch as if the previous parameters where some random
        # start parameters and we don't know anyhting yet
        optimizer.zero_grad()
        pred = model(X)
        
        loss = loss_fn(pred, y)

        batch_size = X.size(0)
        sample_cnt += batch_size
        running_loss += loss.item() * batch_size # multiplying because on last iteration batch size can differ
        
    return running_loss / sample_cnt


def validate_one_epoch(
    model,
    loss_fn,
    val_loader 
) -> float:
    model.eval()
    running_loss = 0
    sample_cnt = 0
    with torch.no_grad():
        for X, y in val_loader:
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
    epochs
) -> dict[str, list[float]]:
    history = {
        "train_loss": [],
        "val_loss": [],
    }
    for _ in range(0, epochs):
        train_loss = train_one_epoch(model, optimizer, loss_fn, train_loader)
        val_loss = validate_one_epoch(model, loss_fn, val_loader)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
    return history