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

experiment = EXPERIMENTS[sys.argv[1]]


qids, y_full, X = load_dataset(experiment['train_folder'])
X_normalized = normalize_features(X)

X_train, X_test, y_train, y_test = train_test_split(X, y_full, test_size=0.33, random_state=21)

# convert the data to tensors
X_train = torch.from_numpy(X_train).type(torch.FloatTensor)
X_test = torch.from_numpy(X_test).type(torch.FloatTensor)
y_train = torch.from_numpy(y_train).type(torch.FloatTensor)
y_test = torch.from_numpy(y_test).type(torch.FloatTensor)
X_normalized = torch.from_numpy(X_normalized).type(torch.FloatTensor)
y_full = torch.from_numpy(y_full).type(torch.FloatTensor)

# define the model and params
net = experiment['model']

optimizer = optim.Adam(net.parameters(), lr=experiment['lr'])
num_epochs = experiment['num_epochs']
batch_size = experiment['batch_size']

loss_fn = experiment['loss_fn']


def train_test_loop():
    loader = create_data_loader(y_train, X_train, batch_size)

    losses = []
    test_losses = []

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
    plt.title('Learning curves')
    plt.ylabel(experiment['plot_y_label_name'])
    plt.xlabel('#Epoch')
    plt.legend()
    plt.savefig(f'../plots/{experiment["plot_name"]}')


def train_loop():
    loader = create_data_loader(y_full, X_normalized, batch_size)

    losses = []

    for epoch in range(num_epochs):

        running_loss = []

        for input, y in loader:
            optimizer.zero_grad()

            predicted = net(input)
            
            loss = loss_fn(predicted, y)
            loss.backward()

            optimizer.step()

            running_loss.append(loss.item())
        
        mean_loss = np.mean(running_loss)
        losses.append(mean_loss)

        print(f'Epoch {epoch} - Full Train Loss: {mean_loss}')

    torch.save(net.state_dict(), f'../models/{experiment["model_name"]}')


if sys.argv[2] == 'full_train':
    train_loop()

elif sys.argv[2] == 'train_test':
    train_test_loop()
