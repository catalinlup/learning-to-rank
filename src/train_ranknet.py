import sys

from data_loaders import get_pairwise_dataset
from experiments import EXPERIMENTS
from preprocessing import normalize_features, create_data_loader
from sklearn.model_selection import train_test_split
from neural_nets.RankNet import RankNet
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from loss_functions import ranknet_loss


experiment = EXPERIMENTS[sys.argv[1]]
qids, y, X = get_pairwise_dataset(experiment['train_folder'])
X_normalized = normalize_features(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=21)

# convert the data to tensors
X_train = torch.from_numpy(X_train).type(torch.FloatTensor)
X_test = torch.from_numpy(X_test).type(torch.FloatTensor)
y_train = torch.from_numpy(y_train).type(torch.FloatTensor)
y_test = torch.from_numpy(y_test).type(torch.FloatTensor)

# define the model and params
net = RankNet(46, 5)

optimizer = optim.Adam(net.parameters(), lr=1e-4)
num_epochs = 50
batch_size = 64

loss_fn = nn.BCELoss()

loader = create_data_loader(y_train, X_train, batch_size)

losses = []
test_losses = []

# print(X_test.shape)

for epoch in range(num_epochs):

    with torch.no_grad():
        predicted_test = net(X_test)
        test_loss = loss_fn(predicted_test, y_test).item()
        test_losses.append(test_loss)

    running_loss = []

    for input, y in loader:
        optimizer.zero_grad()

        predicted = net(input)

        # print(predicted)
        
        loss = loss_fn(predicted, y)
        loss.backward()

        optimizer.step()

        running_loss.append(loss.item())
    
    mean_loss = np.mean(running_loss)
    losses.append(mean_loss)

    print(f'Epoch {epoch} - Train Loss: {mean_loss} | Test Loss: {test_loss}')


# Plot the learning curve
plt.plot(losses[1:], label='Train Loss')
plt.plot(test_losses[1:], label='Test Loss')
plt.title('Ranknet Learning curves')
plt.ylabel('Ranknet Loss')
plt.xlabel('#Epoch')
plt.legend()
plt.savefig('../plots/learning_curve_ranknet.png')
