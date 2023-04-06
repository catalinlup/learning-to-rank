import numpy as np


def dsg(predicted_ranks: np.ndarray, actual_ranks: np.ndarray):
    """
    Computes the dsg score for the predicted ranks.

    Args:
        predicted_ranks: Predicted ranks numpy array.
        actual_ranks: True ranks numpy array.
    Returns:
        DSG score
    """

    sorted_indexes = np.flip(np.argsort(predicted_ranks))
    sorted_actual_ranks = actual_ranks[sorted_indexes]

    rank_vector = np.power(2, sorted_actual_ranks) - 1
    log_vector = np.log2(np.arange(2, rank_vector.size + 2))

    return np.sum(rank_vector / log_vector)


def idsg(actual_ranks: np.ndarray):
    """
    Compute the ideal discounted cumulative gain.

    Args:
        actual_ranks: True ranks numpy array.
    Returns:
        Ideal DSG score
    """
    sorted_actual_ranks = np.flip(np.sort(actual_ranks))

    rank_vector = np.power(2, sorted_actual_ranks) - 1
    log_vector = np.log2(np.arange(2, rank_vector.size + 2))

    return np.sum(rank_vector / log_vector)


def ndsg(predicted_ranks: np.ndarray, actual_ranks: np.ndarray):
    """
    Computes the normalized discounted gain.

    Args:
        predicted_ranks: Predicted ranks numpy array.
        actual_ranks: True ranks numpy array.
    Returns:
        Normalized DSG score
    """

    actual_ranks += 1

    return dsg(predicted_ranks, actual_ranks) / idsg(actual_ranks)


def mse(predicted_ranks: np.ndarray, actual_ranks: np.ndarray):
    """
    Compute the Mean Squared Error.

    Args:
        predicted_ranks: Predicted ranks numpy array.
        actual_ranks: True ranks numpy array.
    Returns:
        MSE score
    """
    output_errors = np.average((actual_ranks - predicted_ranks) ** 2, axis=0)

    return np.average(output_errors)


def mape(predicted_ranks: np.ndarray, actual_ranks: np.ndarray):
    """
    Mean absolute percentage error (MAPE).

    Args:
        predicted_ranks: Predicted ranks numpy array.
        actual_ranks: True ranks numpy array.
    Returns:
        MAPE score
    """
    epsilon = np.finfo(np.float64).eps
    map_error = np.abs(predicted_ranks - actual_ranks) / np.maximum(np.abs(actual_ranks), epsilon)
    output_errors = np.average(map_error, axis=0)

    return np.average(output_errors)


def mrr(rs: np.ndarray):
    """
    Score is reciprocal of the rank of the first relevant item
    First element is 'rank 1'.  Relevance is binary (nonzero is relevant).

    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean reciprocal rank
    """
    rs = (np.asarray(r).nonzero()[0] for r in rs)

    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])


def r_precision(r: np.ndarray):
    """
    Score is precision after all relevant documents have been retrieved
    Relevance is binary (nonzero is relevant).

    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        R Precision
    """
    r = np.asarray(r) != 0
    z = r.nonzero()[0]
    if not z.size:
        return 0.

    return np.mean(r[:z[-1] + 1])


def precision_at_k(r: np.ndarray, k: int):
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
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')

    return np.mean(r)


def average_precision(r: np.ndarray):
    """
    Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).

    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Average precision
    """
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.

    return np.mean(out)


def mean_average_precision(rs: np.ndarray):
    """
    Score is mean average precision
    Relevance is binary (nonzero is relevant).

    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r) for r in rs])


def dcg_at_k(r: np.ndarray, k: int, method=0):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.

    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r: np.ndarray, k: int, method=0):
    """
    Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.

    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max
