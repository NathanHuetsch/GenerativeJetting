import torch
import torch.nn as nn


class Classifier(nn.Module):
    """
    Simple Conditional Resnet class to build from a params dict
    """
    def __init__(self, param):
        super().__init__()
        # Read in the network specifications from the params
        self.param = param
        self.dim = self.param["dim"]

        self.n_layers = param["n_layers"]
        self.intermediate_dim = self.param["intermediate_dim"]

        self.dropout = self.param.get("dropout", None)
        self.normalization = self.param.get("normalization", None)
        self.activation = self.param.get("activation", "SiLU")

        # Build the Resnet blocks
        layers = [nn.Linear(self.dim, self.intermediate_dim)]
        if self.normalization is not None:
            layers.append(getattr(nn, self.normalization)())
        if self.dropout is not None:
            layers.append(nn.Dropout(p=self.dropout))
        layers.append(getattr(nn, self.activation)())

        for i in range(1, self.n_layers):
            layers.append(nn.Linear(self.intermediate_dim, self.intermediate_dim))
            if self.normalization is not None:
                layers.append(getattr(nn, self.normalization)())
            if self.dropout is not None:
                layers.append(nn.Dropout(p=self.dropout))
            layers.append(getattr(nn, self.activation)())

        layers.append(nn.Linear(self.intermediate_dim, 1))
        layers.append(nn.Sigmoid())

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
