from neural_nets.SimpleNet import SimpleNet
from neural_nets.RankNet import RankNet
import numpy as np
import torch
from functools import cmp_to_key
# Contain functions that rank a batch of documents in a query based on a model


def simple_ranker(net: SimpleNet, X: np.ndarray):
    """
    Ranks a batch of documents using simple net
    """
    print(X.shape)
    X = torch.from_numpy(X).type(torch.FloatTensor)
    predicted = net(X)

    print(predicted.shape)
    print(predicted)

    return predicted.detach().numpy().T[0]


def _approx_ndcg_ranker_helper(i, predicted):
    return 1 + np.sum(predicted[i] < predicted)

def approx_ndcg_ranker(net: SimpleNet, X: np.ndarray):
    X = torch.from_numpy(X).type(torch.FloatTensor)
    predicted = net(X)
    predicted = predicted.detach().numpy().T
    # print(predicted.shape)
    ranks = list(np.zeros_like(predicted))

    for i in range(len(ranks)):
        ranks[i] = _approx_ndcg_ranker_helper(i, predicted)

    print(ranks)
    return np.array(ranks)


def pairwise_ranker(net: RankNet, X: np.ndarray):
    """
    Ranks a batch of documents using rank net
    """

    # create an array of document scores
    doc_indices = list(np.arange(0, X.shape[0]))

    # assign the document score to the documnets based on the order
    # returned by ranknet

    def comparator(i, j):
        feature_vector = np.concatenate([X[i], X[j]])
        feature_vector = torch.from_numpy(feature_vector).type(torch.FloatTensor).unsqueeze(dim=0)
        output = net(feature_vector)
        # print(output.item())

        return output.item() - 0.5

    # the first index corresponds to the lowest rank
    sorted_doc_indices = sorted(doc_indices, key=cmp_to_key(comparator))

    ranks = list(np.zeros(len(sorted_doc_indices)))

    # assign ranks in the order of the sorted indices
    for rank, index in enumerate(sorted_doc_indices):
        ranks[index] = rank + 1

    return ranks
