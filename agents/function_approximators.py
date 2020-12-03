import torch.nn as nn


class MLP(nn.Module):
    """
    A simple MLP.
    """

    def __init__(self, layer_sizes):
        """
        layer_sizes[0] is input, layer_sizes[-1] is output
        Args:
            layer_sizes:
        """
        super().__init__()
        self.fc = nn.Sequential()
        for i in range(0, len(layer_sizes) - 1):
            self.fc.add_module(f'layer_{i}', nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            self.fc.add_module(f'activation_{i}', nn.ReLU())

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))
