import numpy as np
from utils import DEFAULT_EPS

def dsg(predicted_ranks: np.ndarray, actual_ranks: np.ndarray, k):
    """
    Computes the dsg score for the predicted ranks.

    Args:
        predicted_ranks: Predicted ranks numpy array.
        actual_ranks: True ranks numpy array.
    Returns:
        DSG score
    """

    sorted_indexes = np.flip(np.argsort(predicted_ranks))
    sorted_actual_ranks = actual_ranks[sorted_indexes][:k]

    rank_vector = np.power(2, sorted_actual_ranks) - 1
    log_vector = np.log2(np.arange(2, rank_vector.size + 2))

    return np.sum(rank_vector / log_vector)


def idsg(actual_ranks: np.ndarray, k):
    """
    Compute the ideal discounted cumulative gain.

    Args:
        actual_ranks: True ranks numpy array.
    Returns:
        Ideal DSG score
    """
    sorted_actual_ranks = np.flip(np.sort(actual_ranks))[:k]

    rank_vector = np.power(2, sorted_actual_ranks) - 1
    log_vector = np.log2(np.arange(2, rank_vector.size + 2))

    return np.sum(rank_vector / log_vector)


def ndsg(predicted_ranks: np.ndarray, actual_ranks: np.ndarray, k=None):
    """
    Computes the normalized discounted gain.

    Args:
        predicted_ranks: Predicted ranks numpy array.
        actual_ranks: True ranks numpy array.
    Returns:
        Normalized DSG score
    """

    if k == None:
        try:
            k = predicted_ranks.shape[0]
        except:
            k = len(predicted_ranks)


    return dsg(predicted_ranks, actual_ranks + 1, k) / idsg(actual_ranks + 1, k)




def precision_at_k(y_pred: np.ndarray, y_true: int, k):
    """
    Score is precision @ k
    Relevance is binary (nonzero is relevant).

    Traceback (most recent call last):
        File "<stdin>", line 1, in ?
    ValueError: Relevance score length < k

    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    sorted_indexes = np.flip(np.argsort(y_pred))
    sorted_actual_ranks = y_true[sorted_indexes]

    print('Sorted', sorted_actual_ranks)

    return np.mean(sorted_actual_ranks[:k] > 0)







    # assert k >= 1
    # r = np.asarray(r)[:k] != 0
    # if r.size != k:
    #     raise ValueError('Relevance score length < k')

    # return np.mean(r)


def average_precision(y_pred: np.ndarray, y_true: np.ndarray):
    """
    Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).

    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Average precision
    """
    sorted_indexes = np.flip(np.argsort(y_pred))
    sorted_actual_ranks = y_true[sorted_indexes]

    precisions = np.array([precision_at_k(y_pred, y_true, k) for k in range(1, sorted_actual_ranks.shape[0] + 1)])

    if np.sum(sorted_actual_ranks > 0) < DEFAULT_EPS:
        return 0.0

    avg_prec = np.sum((sorted_actual_ranks > 0) * precisions) / np.sum(sorted_actual_ranks > 0)
    # print('Avg prec', avg_prec)
    return avg_prec
    


# def mean_average_precision(rs: np.ndarray):
#     """
#     Score is mean average precision
#     Relevance is binary (nonzero is relevant).

#     Args:
#         rs: Iterator of relevance scores (list or numpy) in rank order
#             (first element is the first item)
#     Returns:
#         Mean average precision
#     """
#     return np.mean([average_precision(r) for r in rs])


# def dcg_at_k(r: np.ndarray, k: int, method=0):
#     """Score is discounted cumulative gain (dcg)
#     Relevance is positive real values.  Can use binary
#     as the previous methods.

#     Args:
#         r: Relevance scores (list or numpy) in rank order
#             (first element is the first item)
#         k: Number of results to consider
#         method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
#                 If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
#     Returns:
#         Discounted cumulative gain
#     """
#     r = np.asfarray(r)[:k]
#     if r.size:
#         if method == 0:
#             return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
#         elif method == 1:
#             return np.sum(r / np.log2(np.arange(2, r.size + 2)))
#         else:
#             raise ValueError('method must be 0 or 1.')
#     return 0.


# def ndcg_at_k(r: np.ndarray, k: int, method=0):
#     """
#     Score is normalized discounted cumulative gain (ndcg)
#     Relevance is positive real values.  Can use binary
#     as the previous methods.

#     Args:
#         r: Relevance scores (list or numpy) in rank order
#             (first element is the first item)
#         k: Number of results to consider
#         method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
#                 If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
#     Returns:
#         Normalized discounted cumulative gain
#     """
#     dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
#     if not dcg_max:
#         return 0.
#     return dcg_at_k(r, k, method) / dcg_max
