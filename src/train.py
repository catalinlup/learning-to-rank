from data_loaders import get_dataset
from preprocessing import normalize_features
from sklearn.model_selection import train_test_split
from neural_nets.SimpleNet import SimpleNet
import torch
import torch.optim as optim

qids, y, X = get_dataset('../data/MQ2008/min.txt')
X_normalized = normalize_features(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=21)

# convert the data to tensors
X_train = torch.tensor(X_train)
X_test = torch.tensor(X_test)
y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)

# define the model and params
net = SimpleNet(47, 5)

optimizer = optim.Adam(net.parameters(), lr=1e-3)
num_epochs = 20
batch_size = 64

