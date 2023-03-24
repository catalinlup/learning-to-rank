from data_loaders import get_dataset
from preprocessing import normalize_features, create_data_loader
from sklearn.model_selection import train_test_split
from neural_nets.SimpleNet import SimpleNet
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


device = 'cuda' if torch.cuda.is_available() else 'cpu'

qids, y, X = get_dataset('../data/MQ2008/min.txt')
X_normalized = normalize_features(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=21)

# convert the data to tensors
X_train = torch.from_numpy(X_train).type(torch.FloatTensor)
X_test = torch.from_numpy(X_test).type(torch.FloatTensor)
y_train = torch.from_numpy(y_train).type(torch.FloatTensor)
y_test = torch.from_numpy(y_test).type(torch.FloatTensor)

# define the model and params
net = SimpleNet(46, 5)

optimizer = optim.Adam(net.parameters(), lr=1e-3)
num_epochs = 100
batch_size = 64

loss_fn = nn.MSELoss()

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

    # print(f'Epoch {epoch} - Loss: {mean_loss}')

    # Compute the test loss


  

    print(f'Epoch {epoch} - Train Loss: {mean_loss} | Test Loss: {test_loss}')



# Plot the learning curve
plt.plot(losses[1:], label='Train Loss')
plt.plot(test_losses[1:], label='Test Loss')
plt.title('Learning curves')
plt.ylabel('MSE Loss')
plt.xlabel('#Epoch')
plt.legend()
plt.savefig('../plots/learning_curve.png')




