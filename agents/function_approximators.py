import torch.nn as nn

class MLP(nn.Module):
    """
    A simple MLP built from an experiment config file that defines the hidden layer sizes,
    activation, loss, and optimizer.
    TODO add lr scheduler
    """

    def __init__(self, config):
        super().__init__()
        hidden_layer_sizes = config['hidden_layers']
        self.layers = nn.Sequential()
        self.layers.add_module('input', nn.Linear(config['num_states'], hidden_layer_sizes[0]))

        for i in range(1, len(hidden_layer_sizes)):
            self.layers.add_module(f'hidden_{i}', nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i + 1]))
            self.layers.add_module(f'activation_{i}', nn.ReLU())

        self.layers.add_module('output', nn.Linear(config['num_actions'], hidden_layer_sizes[-1]))

    def forward(self, x):
        return self.layers(x)

