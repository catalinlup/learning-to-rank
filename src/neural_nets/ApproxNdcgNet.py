import torch
import torch.nn as nn
from neural_nets.SimpleNet import SimpleNet
from loss_functions import approxNDCGLoss


class ApproxNdcgNet(nn.Module):
    """
    Defines a simple neural network for regression purposes
    """

    def __init__(self, input_size, hidden_size) -> None:
        super().__init__()
        self.simple_net = SimpleNet(input_size, hidden_size)


    def forward(self, x):
        """
        Shape of x - (b, d, f)
        """

        if len(x.shape) < 3:
            x = x.unsqueeze(dim = 0)

        b = x.shape[0]
        d = x.shape[1]
        f = x.shape[2]

        x = torch.reshape(x, (b * d, f))
        x = self.simple_net(x)
        x = torch.reshape(x, (b, d))

        return x
