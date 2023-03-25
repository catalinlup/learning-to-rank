import torch
import torch.nn as nn

class RankNet(nn.Module):
    """
    Defines a simple neural network for regression purposes
    """

    def __init__(self, input_size, hidden_size) -> None:
        super().__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        split = x.split(self.input_size, dim=-1)
        x_first = split[0]
        x_second = split[1]

        # print(x_first.shape)

        x_first = self.fc1(x_first)
        x_second = self.fc1(x_second)

        x_first = self.relu(x_first)
        x_second = self.relu(x_second)


        x_first = self.fc2(x_first)
        x_second = self.fc2(x_second)

        return self.sigmoid(x_first - x_second).squeeze(dim=1)
    
