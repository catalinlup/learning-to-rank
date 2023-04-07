import torch
import numpy as np
from utils import get_torch_device
import pickle


# TARGET_FOLDER = '../data/train/GroupedQbQMQ2008'
# OUTPUT_FOLDER = '../data/train/PairwiseMQ2008'


# batches = pickle.load(open(f"{TARGET_FOLDER}/batches.pickle", 'rb'))
# batches_test = pickle.load(open(f"{TARGET_FOLDER}/batches_test.pickle", 'rb'))



def pairwise_comb_mat(X):
    combs = torch.combinations(torch.arange(X.shape[0]))
    v1 = X[combs[:, 0]]
    v2 = X[combs[:, 1]]
    return torch.concat([v1, v2], dim=-1)

def pairwise_comb_mat_reverse(X):
    combs = torch.combinations(torch.arange(X.shape[0]))
    v1 = X[combs[:, 0]]
    v2 = X[combs[:, 1]]
    return torch.concat([v2, v1], dim=-1)


def pairwise_comb_full(X):
    x1 = pairwise_comb_mat(X)
    x2 = pairwise_comb_mat(X)

    return torch.concat([x1, x2], dim = 0)


def create_score(y):
    combs = torch.combinations(y)
    v1 = combs[:, 0] > combs[:, 1]
    v2 = combs[:, 0] == combs[:, 1]
    v3 = combs[:, 0] < combs[:, 1]
    return 1.0 * v1 + 0.5 * v2 + 0.0 * v3


def create_score_reverse(y):
    combs = torch.combinations(y)
    v1 = combs[:, 0] > combs[:, 1]
    v2 = combs[:, 0] == combs[:, 1]
    v3 = combs[:, 0] < combs[:, 1]
    return 1.0 * v3 + 0.5 * v2 + 0.0 * v1


def create_score_full(y):
    y1 = create_score(y)
    y2 = create_score_reverse(y)
    return torch.concat([y1, y2])


def upsample_pairwise_dataset(X: torch.tensor, y: torch.tensor):
    """
    Upsamples the pairwise dataset such that the number if instances with probability 0
    and the number of instances with probability 1 to be equal to the number of instances
    with probability 0.5
    """

    labels = y * 10
    labels = labels.type(torch.IntTensor)

    index_at_0 = torch.nonzero(labels == 0).T[0]
    # print(index_at_0)
    index_at_5 = torch.nonzero(labels == 5).T[0]
    # print(index_at_5)
    index_at_10 = torch.nonzero(labels == 10).T[0]
    # print(index_at_10)

    # perform upsampling
    
    # count the number of instances that have the index at 5
    cnt_5 = max(index_at_5.shape[0], index_at_0.shape[0], index_at_10.shape[0])

    # sample cnt_5 items with the score of 0 and cnt_5 items with the score of 10
    print(index_at_0.shape)
    print(index_at_5.shape)
    print(index_at_10.shape)

    if index_at_0.shape[0] == 0 or index_at_5.shape[0] == 0 or index_at_10.shape[0] == 0:
        return X, y

    samples_at_0 = np.random.choice(index_at_0, size=cnt_5)
    samples_at_5 = np.random.choice(index_at_5, size=cnt_5)
    samples_at_10 = np.random.choice(index_at_10, size=cnt_5)

    # print(samples_at_0)

    
    X_upsampled = torch.concat([X[samples_at_5], X[samples_at_0], X[samples_at_10]], dim=0)
    y_upsampled = torch.concat([y[samples_at_5], y[samples_at_0], y[samples_at_10]])

    # print(y)
    # print(y_upsampled)

    # reshufle the upsampled dataset
    reshuffled_indices = np.arange(y_upsampled.shape[0]).astype(np.int32)
    np.random.shuffle(reshuffled_indices)

    # print(reshuffled_indices)
    # print(X_upsampled)

    X_upsampled = X_upsampled[reshuffled_indices]
    y_upsampled = y_upsampled[reshuffled_indices]

    return X_upsampled, y_upsampled


def compute_pairwise_dataset(X, y):
    X_full = pairwise_comb_full(X)
    y_full = create_score_full(y)

    return upsample_pairwise_dataset(X_full, y_full)


