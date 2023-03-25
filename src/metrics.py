import numpy as np

def dsg(predicted_ranks: np.ndarray, actual_ranks: np.ndarray):
    """
    Computes the dsg score for the predicted ranks.
    """

    sorted_indexes = np.flip(np.argsort(predicted_ranks))
    sorted_actual_ranks = actual_ranks[sorted_indexes]

    rank_vector = np.power(2, sorted_actual_ranks) - 1
    log_vector = np.log2(np.arange(2, rank_vector.size + 2))

    return np.sum(rank_vector / log_vector)


def idsg(actual_ranks: np.ndarray):
    """
    Compute the ideal discounted cumulative gain
    """
    sorted_actual_ranks = np.flip(np.sort(actual_ranks))

    rank_vector = np.power(2, sorted_actual_ranks) - 1
    log_vector = np.log2(np.arange(2, rank_vector.size + 2))

    return np.sum(rank_vector / log_vector)


def ndsg(predicted_ranks: np.ndarray, actual_ranks: np.ndarray):
    """
    Computes the normalized discounted gain
    """

    actual_ranks += 1


    return dsg(predicted_ranks, actual_ranks) / idsg(actual_ranks)