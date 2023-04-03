from data_loaders import get_pairwise_dataset
from preprocessing import normalize_features, create_data_loader
from sklearn.model_selection import train_test_split
from neural_nets.SimpleNet import SimpleNet
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle


qids, y, X = get_pairwise_dataset('../data/MQ2008/min.txt')

pickle.dump(qids, open('../data/PairwiseMQ2008/qids.pickle', 'wb'))
pickle.dump(y, open('../data/PairwiseMQ2008/target_p.pickle', 'wb'))
pickle.dump(X, open('../data/PairwiseMQ2008/features.pickle', 'wb'))

# print(qids)
# print(y)
# print(X)
