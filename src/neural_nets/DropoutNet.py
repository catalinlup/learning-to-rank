import torch
import torch.nn as nn


class DropoutNet(nn.Module):
    """
    Defines a simple neural network for regression purposes
    """

    def __init__(self, input_size, hidden_size, p) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.do1 = nn.Dropout(p[0])
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.do2 = nn.Dropout(p[1])

    def forward(self, x):
        # print(x.shape)
        x = self.fc1(x)
        x = self.do1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.do2(x)

        return x
