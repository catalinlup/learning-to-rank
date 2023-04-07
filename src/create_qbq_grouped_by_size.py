from data_loaders import load_dataset
from preprocessing import normalize_features, create_data_loader
from sklearn.model_selection import train_test_split
from neural_nets.SimpleNet import SimpleNet
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from experiments import EXPERIMENTS
import sys
import pickle
from utils import get_torch_device, generate_random_mask

TARGET_DATA_SET=sys.argv[1]
FILTER=int(sys.argv[2])

device = get_torch_device()


def convert_np_to_tensor(arr):
    return torch.from_numpy(arr).type(torch.FloatTensor).to(device)

qids, y_full, X = load_dataset(f'../data/train/{TARGET_DATA_SET}')
X_normalized = [normalize_features(x) for x in X]
test_qids = generate_random_mask(len(qids), 0.33)

batches = {}
batches_test = {}

print(y_full[0].shape[0])

for i, qid in enumerate(qids):
    y = y_full[i]
    X_c = X_normalized[i]

    if test_qids[i]:
        cbatches = batches_test
    else:
        cbatches = batches

    shape = y.shape[0]

    if shape not in cbatches.keys():
        cbatches[shape] = dict()
        cbatches[shape]['qid'] = []
        cbatches[shape]['y'] = []
        cbatches[shape]['X'] = []

    cbatches[shape]['qid'].append(qid)
    cbatches[shape]['y'].append(y)
    cbatches[shape]['X'].append(X_c)


def batches_to_tensor(batches):

    new_batches = dict()
    for shape in batches.keys():
        if len(batches[shape]['qid']) < FILTER:
            continue

        new_batches[shape] = dict()
        new_batches[shape]['qid'] = convert_np_to_tensor(np.array(batches[shape]['qid']))
        new_batches[shape]['y'] = convert_np_to_tensor(np.stack(batches[shape]['y']))
        new_batches[shape]['X'] = convert_np_to_tensor(np.stack(batches[shape]['X']))

    return new_batches

batches = batches_to_tensor(batches)
batches_test = batches_to_tensor(batches_test)





pickle.dump(batches, open(f'../data/train/{TARGET_DATA_SET}/batches.pickle', 'wb'))
pickle.dump(batches_test, open(f'../data/train/{TARGET_DATA_SET}/batches_test.pickle', 'wb'))
