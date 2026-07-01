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

def train_once():
    #TODO
    
def train_multiple(): 
    #TODO
    
def validate():
    #TODO