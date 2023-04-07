from data_loaders import load_dataset, group_data_by_query_id
from preprocessing import normalize_features, create_data_loader
from sklearn.model_selection import train_test_split
from neural_nets.SimpleNet import SimpleNet
from neural_nets.RankNet import RankNet
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from experiments import EXPERIMENTS
import sys
from rankers import simple_ranker, pairwise_ranker
from metrics import ndsg
import json

experiment = EXPERIMENTS[sys.argv[1]]
EVALUATION_FOLDER = sys.argv[2]


# load the experiment batches

def load_experiment_batches():
    evaluation_folder = f"../data/evaluation/{EVALUATION_FOLDER}"
    qids, y, X = load_dataset(evaluation_folder)
    X_normalized = [normalize_features(x) for x in X]
    # group the retrieved documents by query
    return group_data_by_query_id(qids, y, X_normalized)


data = load_experiment_batches()

model = experiment['model']
model.load_state_dict(torch.load(f'../models/{experiment["model_name"]}'))
model.eval()

ranker = experiment['ranker']

metrics_acc = dict()

for metric_name in experiment['metrics'].keys():
    metrics_acc[metric_name] = []


# compute the metrics

for qid in data.keys():
    batch = data[qid]
    scores = []
    features = []

    for score, feature in batch:
        scores.append(score)
        features.append(feature)

    scores = scores[0]
    features = np.stack(features)

    if np.sum(scores) < 1:
        continue
    
 
    predicted_ranks = ranker(model, features)
    print('Predicted ranks', predicted_ranks)
    print('Scores', scores)

    for metric_name in experiment['metrics'].keys():
        metrics_acc[metric_name].append(experiment['metrics'][metric_name](predicted_ranks, scores))


# compute the mean of each of the metrics
# print(metrics_acc)
metric_mean = dict()

for metric_name in metrics_acc.keys():
    metric_mean[metric_name] = np.mean(np.array(metrics_acc[metric_name]))

json.dump(metric_mean, open(f'../evaluation/{experiment["experiment_name"]}.json', 'w'))
