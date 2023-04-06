import numpy as np
from data_loaders import get_dataset, get_pairwise_dataset
import pickle

qids, y, X = get_dataset('../data/MQ2008/min.txt')
unique_quids = np.unique(qids)

y_grouped = []
X_grouped = []
qids_grouped = []

shapes = []

for qid in unique_quids:
    mask = (qids == qid)

    y_g = y[mask]
    X_g = X[mask]

    shapes.append(y_g.shape[0])

    y_grouped.append(y_g)
    X_grouped.append(X_g)
    qids_grouped.append(qid)


unique, counts = np.unique(shapes, return_counts=True)

print(np.asarray((unique, counts)).T)