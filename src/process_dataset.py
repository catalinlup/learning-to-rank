import numpy as np
from data_loaders import get_dataset, get_pairwise_dataset
import pickle


def save_dataset(qids, X, y, folder):
    """
    Save the dataset in the provided folder.
    """
    pickle.dump(qids, open(f'{folder}/qids.pickle', 'wb'))
    pickle.dump(y, open(f'{folder}/y.pickle', 'wb'))
    pickle.dump(X, open(f'{folder}/X.pickle', 'wb'))


def generate_random_mask(size, ratio_positives):
    """
    Generates a random binary mask of size 'size' having ratio_positives * size 1s
    """
    mask = np.zeros(size)
    mask[:int(ratio_positives * size)] = 1
    np.random.shuffle(mask)
    mask = mask.astype(bool)

    return mask


def split_dataset(qids, X, y, evaluation_size=0.1):
    """
    Splits the dataset into train and test based on the questions.
    Keyword arguments:
    qids -- the array containing the query ids
    X -- the feature matrix
    y -- the label (rank) matrix
    """

    qids_train = []
    qids_evaluation = []
    X_train = []
    X_evaluation = []
    y_train = []
    y_evaluation = []

    # get the evaluation and train qids
    unique_qids = np.unique(qids)
    random_mask = generate_random_mask(unique_qids.size, evaluation_size)
    evaluation_qids = set(unique_qids[random_mask])

    for i, qid in enumerate(qids):
        if qid in evaluation_qids:

            qids_evaluation.append(qid)
            X_evaluation.append(X[i])
            y_evaluation.append(y[i])

        else:
            qids_train.append(qid)
            X_train.append(X[i])
            y_train.append(y[i])
    
    qids_evaluation = np.array(qids_evaluation)
    qids_train = np.array(qids_train)

    X_evaluation = np.stack(X_evaluation)
    X_train = np.stack(X_train)

    y_evaluation = np.stack(y_evaluation)
    y_train = np.stack(y_train)

    return qids_train, y_train, X_train, qids_evaluation, y_evaluation, X_evaluation


def upsample_pairwise_dataset(qids: np.ndarray, X: np.ndarray, y: np.ndarray):
    """
    Upsamples the pairwise dataset such that the number if instances with probability 0
    and the number of instances with probability 1 to be equal to the number of instances
    with probability 0.5
    """

    labels = y * 10
    labels = labels.astype(np.int32)

    index_at_0 = np.argwhere(labels == 0).T[0]
    index_at_5 = np.argwhere(labels == 5).T[0]
    index_at_10 = np.argwhere(labels == 10).T[0]

    # perform upsampling
    
    # count the number of instances that have the index at 5
    cnt_5 = index_at_5.size

    # sample cnt_5 items with the score of 0 and cnt_5 items with the score of 10
    samples_at_0 = np.random.choice(index_at_0, size=cnt_5)
    samples_at_10 = np.random.choice(index_at_10, size=cnt_5)

    qids_upsampled = np.array(list(qids[index_at_5]) + list(qids[samples_at_0]) + list(qids[samples_at_10]))
    X_upsampled = np.stack(list(X[index_at_5]) + list(X[samples_at_0]) + list(X[samples_at_10]))
    y_upsampled = np.array(list(y[index_at_5]) + list(y[samples_at_0]) + list(y[samples_at_10]))

    # reshufle the upsampled dataset
    reshuffled_indices = np.arange(qids_upsampled.size).astype(np.int32)
    np.random.shuffle(reshuffled_indices)

    qids_upsampled = qids_upsampled[reshuffled_indices]
    X_upsampled = X_upsampled[reshuffled_indices]
    y_upsampled = y_upsampled[reshuffled_indices]

    return qids_upsampled, y_upsampled, X_upsampled


# Split the MQ2008 dataset
def process_MQ2008():
    """
    Process the MQ2008 dataset
    """
    qids, y, X = get_dataset('../data/MQ2008/min.txt')

    qids_train, y_train, X_train, qids_evaluation, y_evaluation, X_evaluation = split_dataset(qids, X, y)
    save_dataset(qids_train, X_train, y_train, '../data/train/MQ2008')
    save_dataset(qids_evaluation, X_evaluation, y_evaluation, '../data/evaluation/MQ2008')


# Split the MQ2008 pairwise dataset
def process_MQ2008_Pairwise():
    """
    Process the MQ2008 pairwise dataset
    """

    qids, y, X = get_pairwise_dataset('../data/MQ2008/min.txt')
    qids_train, y_train, X_train, qids_evaluation, y_evaluation, X_evaluation = split_dataset(qids, X, y)

    # save the evaluation dataset
    save_dataset(qids_evaluation, X_evaluation, y_evaluation, '../data/evaluation/PairwiseMQ2008')

    # upsample the train dataset
    qids_upsampled, y_upsampled, X_upsampled = upsample_pairwise_dataset(qids_train, X_train, y_train)

    # save the upsampled dataset
    save_dataset(qids_upsampled, X_upsampled, y_upsampled, '../data/train/PairwiseMQ2008')


process_MQ2008()
process_MQ2008_Pairwise()
