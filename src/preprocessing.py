import numpy as np
from torch.utils.data import DataLoader, Subset
import torch

def normalize_features(features: np.ndarray, epsilon=1e-5) -> np.ndarray:
    """
    Normalizes the provided scores
    """

    mean = np.mean(features, axis=0)
    # print(mean.shape)
    var = np.mean((features - mean) ** 2, axis=0)
    # print(var.shape)
    std = np.sqrt(var + epsilon)

    return (features - mean) / std


def create_data_loader(y: np.ndarray, X: np.ndarray, batch_size: int) -> DataLoader:
    sequence = list(range(y.shape[0]))
    np.random.shuffle(sequence)
    num_batches = len(y.shape[0]) // batch_size

    subsets = [Subset(torch.stack([y, X], dim=0), sequence[i * batch_size: (i + 1) * batch_size]) for i in range(num_batches)]
    train_loader = [DataLoader(sub, batch_size=batch_size) for sub in subsets]  # Create multiple batches, each with BS number of samples

    return train_loader