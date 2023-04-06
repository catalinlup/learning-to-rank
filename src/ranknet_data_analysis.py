from data_loaders import load_dataset
from preprocessing import normalize_features, create_data_loader
from sklearn.model_selection import train_test_split
from neural_nets.RankNet import RankNet
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from loss_functions import ranknet_loss

qids, y, X = load_dataset('../data/train/PairwiseMQ2008')

y = y * 10
y = y.astype(np.int32)
cnt = {0: 0, 5: 0, 10: 0}
for x in y:
    cnt[x] += 1

print(cnt)
sm = cnt[0] + cnt[5] + cnt[10]

sm_cnt = {0: cnt[0] / sm, 5: cnt[5] / sm, 10: cnt[10] / sm}
print(sm_cnt)
