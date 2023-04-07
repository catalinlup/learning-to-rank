import numpy as np
from data_loaders import get_dataset, get_pairwise_dataset
import pickle
import os

def save_dataset(qids, X, y, folder):
    """
    Save the dataset in the provided folder.
    """

    if not os.path.exists(folder):
        os.mkdir(folder)

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

def split_dataset_qbq(qids, X, y, evaluation_size=0.1):
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


def upsample_dataset(qids: np.ndarray, X: np.ndarray, y: np.ndarray):
    """
    Upsamples the normal dataset.
    """
    labels = y.astype(np.int32)

    index_at_0 = np.argwhere(labels == 1).T[0]
    index_at_5 = np.argwhere(labels == 0).T[0]
    index_at_10 = np.argwhere(labels == 2).T[0]



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


def group_by_qids_qbq(qids, y, X):
    unique_quids = np.unique(qids)

    y_grouped = []
    X_grouped = []
    qids_grouped = []

    for qid in unique_quids:
        mask = (qids == qid)

        y_g = y[mask]
        X_g = X[mask]


        y_grouped.append(y_g)
        X_grouped.append(X_g)
        qids_grouped.append(qid)

    qids_grouped = np.array(qids_grouped)
    return qids_grouped, y_grouped, X_grouped

def group_by_qids(qids, y, X):
    """
    Groups the dataset by query id
    """
    unique_quids = np.unique(qids)
    y_grouped = []
    X_grouped = []
    qids_grouped = []

    for qid in unique_quids:
        mask = (qids == qid)

        y_g = y[mask]
        X_g = X[mask]

        if y_g.shape[0] != 8:
            continue

        y_grouped.append(y_g)
        X_grouped.append(X_g)
        qids_grouped.append(qid)

    # print(y_grouped[100].shape)
    y_grouped = np.stack(y_grouped)
    X_grouped = np.stack(X_grouped)
    qids_grouped = np.array(qids_grouped)
    return qids_grouped, y_grouped, X_grouped

# Split the MQ2008 dataset
def process_MQ2008():
    """
    Process the MQ2008 dataset
    """
    qids, y, X = get_dataset('../data/MQ2008/min.txt')

    qids_train, y_train, X_train, qids_evaluation, y_evaluation, X_evaluation = split_dataset(qids, X, y)

 
    save_dataset(qids_train, y_train, X_train, '../data/train/MQ2008')
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


def process_MQ2008_Grouped():
    """
    Process the MQ2008 pairwise dataset grouped by query
    """

    qids, y, X = get_dataset('../data/MQ2008/min.txt')


    qids_grouped, y_grouped, X_grouped = group_by_qids(qids, y, X)

    qids_train, y_train, X_train, qids_evaluation, y_evaluation, X_evaluation = split_dataset(qids_grouped, X_grouped, y_grouped)

    # save the evaluation dataset
    save_dataset(qids_evaluation, X_evaluation, y_evaluation, '../data/evaluation/GroupedMQ2008')

    # upsample the train dataset
    # save the upsampled dataset
    save_dataset(qids_train, X_train, y_train, '../data/train/GroupedMQ2008')



def process_MQ2008_Grouped_QbQ():
    """
    Process the MQ2008 pairwise dataset grouped by query
    """

    qids, y, X = get_dataset('../data/MQ2008/min.txt')
    qids_grouped, y_grouped, X_grouped = group_by_qids_qbq(qids, y, X)



    qids_train, y_train, X_train, qids_evaluation, y_evaluation, X_evaluation = split_dataset_qbq(qids_grouped, X_grouped, y_grouped)


    save_dataset(qids_evaluation, X_evaluation, y_evaluation, '../data/evaluation/GroupedQbQMQ2008')


    save_dataset(qids_train, X_train, y_train, '../data/train/GroupedQbQMQ2008')





def load_folded_dataset(folder_path, fold_paths=[], get_data=lambda x: []):
    """
    Load the folded dataset
    """

    qids_train_folds = []
    y_train_folds = []
    X_train_folds = []

    qids_evaluation_folds = []
    y_evaluation_folds = []
    X_evaluation_folds = []

    for fold in fold_paths:
        qids_train_fold, y_train_fold, X_train_fold = get_data(os.path.join(folder_path, fold, 'train.txt'))
        qids_test_fold, y_test_fold, X_test_fold = get_data(os.path.join(folder_path, fold, 'test.txt'))
        qids_vali_fold, y_vali_fold, X_vali_fold = get_data(os.path.join(folder_path, fold, 'vali.txt'))

        qids_train_folds.append(qids_train_fold)
        qids_train_folds.append(qids_test_fold)
        qids_evaluation_folds.append(qids_vali_fold)

        y_train_folds.append(y_train_fold)
        y_train_folds.append(y_test_fold)
        y_evaluation_folds.append(y_vali_fold)

        X_train_folds.append(X_train_fold)
        X_train_folds.append(X_test_fold)
        X_evaluation_folds.append(X_vali_fold)

    # print(qids_train_folds)

    return np.concatenate(qids_train_folds), np.concatenate(y_train_folds), np.concatenate(X_train_folds), np.concatenate(qids_evaluation_folds), np.concatenate(y_evaluation_folds), np.concatenate(X_evaluation_folds)




def process_MSLR_pairwise(dataset_name, folder_path, folds = []):
    """
    Processes in pairwise format
    """
    qids_train, y_train, X_train, qids_evaluation, y_evaluation, X_evaluation = load_folded_dataset(folder_path, folds, get_pairwise_dataset)

    save_dataset(qids_evaluation, X_evaluation, y_evaluation, f'../data/evaluation/{dataset_name}')

    qids_upsampled, y_upsampled, X_upsampled = upsample_pairwise_dataset(qids_train, X_train, y_train)
    save_dataset(qids_upsampled, y_upsampled, X_upsampled, f'../data/train/{dataset_name}')


def process_MSLR_grouped(dataset_name, folder_path, folds = []):
    """
    Process MSLR in grouped format
    """
    qids_train, y_train, X_train, qids_evaluation, y_evaluation, X_evaluation = load_folded_dataset(folder_path, folds, get_dataset)

    qids_train_grouped, y_train_grouped, X_train_grouped = group_by_qids(qids_train, y_train, X_train)
    qids_evaluation_grouped, y_evaluation_grouped, X_evaluation_grouped = group_by_qids(qids_evaluation, y_evaluation, X_evaluation)

    save_dataset(qids_evaluation_grouped, X_evaluation_grouped, y_evaluation_grouped, f'../data/evaluation/{dataset_name}')
    save_dataset(qids_train_grouped, X_train_grouped, y_train_grouped, f'../data/train/{dataset_name}')


def process_MSLR_grouped_qbq(dataset_name, folder_path, folds=[]):
    """
    Process MSLR in grouped qbq format
    """

    qids_train, y_train, X_train, qids_evaluation, y_evaluation, X_evaluation = load_folded_dataset(folder_path, folds, get_dataset)

    qids_train_grouped, y_train_grouped, X_train_grouped = group_by_qids_qbq(qids_train, y_train, X_train)
    qids_evaluation_grouped, y_evaluation_grouped, X_evaluation_grouped = group_by_qids_qbq(qids_evaluation, y_evaluation, X_evaluation)

    save_dataset(qids_evaluation_grouped, X_evaluation_grouped, y_evaluation_grouped, f'../data/evaluation/{dataset_name}')
    save_dataset(qids_train_grouped, X_train_grouped, y_train_grouped, f'../data/train/{dataset_name}')

def process_MSLR10K_Pairwise():
    """
    Processes the MSLR10K dataset
    """
    process_MSLR_pairwise("PairwiseMSLR10K", "../data/MSLR-WEB10K", folds=['Fold1', 'Fold2', 'Fold3', 'Fold4', 'Fold5'])

def process_MSLR10K_Grouped():
    """
    Processes the MSLR10K grouped dataset
    """
    process_MSLR_grouped("GroupedMSLR10K", "../data/MSLR-WEB10K", folds=['Fold1', 'Fold2', 'Fold3', 'Fold4', 'Fold5'])

def process_MSLR10K_Grouped_QbQ():
    """
    Processes the MSLR10K grouped query by query dataset
    """
    process_MSLR_grouped_qbq("GroupedQbQMSLR10K", "../data/MSLR-WEB10K", folds=['Fold1', 'Fold2', 'Fold3', 'Fold4', 'Fold5'])



def process_MSLR30K_Pairwise():
    process_MSLR_pairwise("PairwiseMSLR30K", "../data/MSLR-WEB30K", folds=['Fold1', 'Fold2', 'Fold3', 'Fold4', 'Fold5'])

def process_MSLR30K_Grouped():
    process_MSLR_grouped("GroupedMSLR30K", "../data/MSLR-WEB30K", folds=['Fold1', 'Fold2', 'Fold3', 'Fold4', 'Fold5'])

def process_MSLR30K_Grouped_QbQ():
    process_MSLR_grouped_qbq("GroupedQbQMSLR30K", "../data/MSLR-WEB30K", folds=['Fold1', 'Fold2', 'Fold3', 'Fold4', 'Fold5'])


# process_MQ2008()
# process_MQ2008_Pairwise()
# process_MQ2008_Grouped()
# process_MQ2008_Grouped_QbQ()
# process_MSLR10K_Pairwise()
process_MSLR10K_Grouped_QbQ()
