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


def _convert_subset_to_tensor(subset: Subset) -> torch.tensor:
    """
    Converts the provided subset to a tensor
"""
    feature_vecs = []

    for v in subset:
        if v.nelement() == 0:
            continue
        feature_vecs.append(v)
    
    if len(feature_vecs) == 1:
        return feature_vecs[0]
    
    # if feature_vecs[0].shape != feature_vecs[1].shape:
    #     print(feature_vecs)
    # print(feature_vecs[0].size())
    # print(feature_vecs[1].size())
    # print('###')
    return torch.stack(feature_vecs)
    
  
def create_data_loader(y: np.ndarray, X: np.ndarray, batch_size: int) -> DataLoader:
    sequence = list(range(y.shape[0]))
    np.random.shuffle(sequence)
    num_batches = y.shape[0] // batch_size

    subsets = [_convert_subset_to_tensor(Subset(X, sequence[i * batch_size: (i + 1) * batch_size])) for i in range(num_batches )]
    y_subsets = [_convert_subset_to_tensor(Subset(y, sequence[i * batch_size: (i + 1) * batch_size])) for i in range(num_batches - 1)]

    subsets = list(filter(lambda x: x!= None, subsets))
    y_subsets = list(filter(lambda x: x!= None, y_subsets))
    # print(_convert_subset_to_tensor(y_subsets[0]))
    # print(_convert_subset_to_tensor(subsets[0]).shape)
    train_loader = [(sub_x, sub_y) for sub_x, sub_y in zip(subsets, y_subsets)]  # Create multiple batches, each with BS number of samples
    # print(train_loader)
    return train_loader