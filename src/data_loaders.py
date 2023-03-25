import numpy as np
import pickle
import random

def process_line(line: str) -> np.ndarray:

    line_without_comment = line.split('#')[0]
    line_without_comment = line_without_comment.strip()
    features = line_without_comment.split(sep=' ')
    score = float(features[0])
    qid = int(features[1].split(':')[1].strip())
    processed_features = list(map(lambda x: float(x.split(':')[1].strip()), features[2:]))


    return qid, score, np.array(processed_features)



def process_dataset(dataset: str):
    qids = []
    scores = []
    features = []

    lines = dataset.splitlines()

    for line in lines:
        qid, score, feature_vec = process_line(line)

        qids.append(qid)
        scores.append(score)
        features.append(feature_vec)

    return np.array(qids), np.array(scores), np.stack(features)


def get_dataset(path: str):
    with open(path, 'r') as file:
        return process_dataset(file.read())




def group_data_by_query_id(qids, scores, features):
    data_by_query = {}

    for i, qid in enumerate(qids):
        if qid not in data_by_query.keys():
            data_by_query[qid] = list()
        
        data_by_query[qid].append((scores[i], features[i]))
    
    return data_by_query

def compute_pairwise_dataset_for_query(qid, data_by_query, score_equal_drop_prob=0.85):
    score_features_list = data_by_query[qid]
    

    pairwise_features = []
    target_probabilities = []

    for i in range(len(score_features_list)):
        for j in range(len(score_features_list)):

            if i == j:
                continue

            score_i, features_i = score_features_list[i][0], score_features_list[i][1]
            score_j, features_j = score_features_list[j][0], score_features_list[j][1]

            # if score_i == score_j:
            #     rnd = random.random()

            #     if rnd < score_equal_drop_prob:
            #         continue

            combined_feature = np.concatenate([features_i, features_j])
            target_probability = 1.0 if score_i > score_j else (0.5 if score_i == score_j else 0.0)

            pairwise_features.append(combined_feature)
            target_probabilities.append(target_probability)

    return pairwise_features, target_probabilities
    

def get_pairwise_dataset(path: str):
    qids, scores, features = get_dataset(path)

    # group dataset by query id
    data_by_query = group_data_by_query_id(qids, scores, features)

    unique_qids = list(set(list(qids)))

    pairwise_qids = []
    pairwise_target_probabilities = []
    pairwise_features = []

    for i, qid in enumerate(unique_qids):
        print(f'{i} / {len(unique_qids)}')
        f, p = compute_pairwise_dataset_for_query(qid, data_by_query)


        pairwise_qids += [qid] * len(p)
        pairwise_target_probabilities += p
        pairwise_features += f

    
    return np.array(pairwise_qids), np.array(pairwise_target_probabilities), np.stack(pairwise_features)


def load_dataset(folder):
    """
    Load the the pairwise training dataset used in ranknet training.
    """
    qids = pickle.load(open(f'{folder}/qids.pickle', 'rb'))
    y = pickle.load(open(f'{folder}/y.pickle', 'rb'))
    X = pickle.load(open(f'{folder}/X.pickle', 'rb'))

    return qids, y, X



