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
from utils import get_torch_device


device = get_torch_device()

experiment = EXPERIMENTS[sys.argv[1]]


qids, y_full, X = load_dataset(experiment['train_folder'])
X_normalized = [normalize_features(x) for x in X]

# print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y_full, test_size=0.33, random_state=21)

# convert the data to tensors
X_train = [torch.from_numpy(x).type(torch.FloatTensor).to(device) for x in X_train]
X_test = [torch.from_numpy(x).type(torch.FloatTensor).to(device) for x in X_test]
y_train = [torch.from_numpy(x).type(torch.FloatTensor).to(device) for x in y_train]
y_test = [torch.from_numpy(x).type(torch.FloatTensor).to(device) for x in y_test]
X_normalized = [torch.from_numpy(x).type(torch.FloatTensor).to(device) for x in X_normalized]
y_full = [torch.from_numpy(x).type(torch.FloatTensor).to(device) for x in y_full]

# define the model and params
net = experiment['model']
net = net.to(device)

optimizer = optim.Adam(net.parameters(), lr=experiment['lr'])

if 'weight_decay' in experiment.keys():
    print(f'Initialized Adam with weight decay of {experiment["weight_decay"]}')
    optimizer = optim.Adam(net.parameters(), lr=experiment['lr'], weight_decay=experiment['weight_decay'])

num_epochs = experiment['num_epochs']
batch_size = experiment['batch_size']

loss_fn = experiment['loss_fn']


def train_test_loop():

    losses = []
    test_losses = []

    for epoch in range(num_epochs):

        with torch.no_grad():
            # print(X_test.shape)
            test_loss = []
            for X_test_1, y_test_1 in zip(X_test, y_test):
                X_test_1 = X_test_1.unsqueeze(dim = 0)
                y_test_1 = y_test_1.unsqueeze(dim = 0)
                predicted_test = net(X_test_1)
                test_loss_1 = loss_fn(predicted_test, y_test_1).item()
                test_loss.append(test_loss_1)

            test_loss = np.mean(np.array(test_loss))
            test_losses.append(test_loss)

        running_loss = []

        for input, y in zip(X_train, y_train):
            
            # print(input.shape)
            # print(y.shape)
            # print('Hi')
            input = input.unsqueeze(dim = 0)
            y = y.unsqueeze(dim = 0)
           
            optimizer.zero_grad()

            predicted = net(input)

            # predicted = predicted.unsqueeze(dim = 0)
            # print(predicted.shape)


           
            loss = loss_fn(predicted, y)
            # print(loss)
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
    # loader = create_data_loader(y_full, X_normalized, batch_size)

    losses = []

    for epoch in range(num_epochs):

        running_loss = []

        for input, y in zip(X_normalized, y_full):
            
            input = input.unsqueeze(dim = 0)
            y = y.unsqueeze(dim = 0)

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
