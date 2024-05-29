import torch
import torch.nn as nn
from Source.Networks.vblinear import VBLinear
import math

class MLP(nn.Module):
    """
    Simple Conditional Resnet class to build from a params dict
    """
    def __init__(self, param):
        super().__init__()
        # Read in the network specifications from the params
        self.param = param

        self.dim_x = self.param["dim_x"]
        self.out_dim = self.param.get("out_dim", self.dim_x)

        # Use GaussianFourierProjection for the time if specified
        self.encode_t = self.param.get("encode_t", False)
        if self.encode_t:
            self.encode_t_dim = self.param.get("encode_t_dim", 64)
            self.embed_t = nn.Sequential(GaussianFourierProjection(embed_dim=self.encode_t_dim,
                                                                 scale=self.encode_t_scale),
                                       nn.Linear(self.encode_t_dim, self.encode_t_dim))
        else:
            self.encode_t_dim = 1
            self.embed_t = nn.Identity()

        # Use a linear layer to embed x if specified
        self.encode_x = self.param.get("encode_x", False)
        if self.encode_x:
            self.encode_x_dim = self.param.get("encode_x_dim", 64)
            self.embed_x = nn.Linear(self.dim, self.encode_x_dim)
        else:
            self.encode_x_dim = self.dim_x
            self.embed_x = nn.Identity()

        self.dim_c = self.param.get("dim_c", 0)
        self.conditional = self.param.get("conditional", False)
        # Use a linear layer to embed c if specified
        self.encode_c = self.param.get("encode_c", False)
        if self.encode_c:
            self.encode_c_dim = self.param.get("encode_c_dim", 64)
            self.embed_c = nn.Linear(self.dim_c, self.encode_c_dim)
        else:
            self.encode_c_dim = self.dim_c
            self.embed_c = nn.Identity()

        # get the network architecture parameters
        self.layers = self.param["layers"]
        self.intermediate_dim = self.param["intermediate_dim"]
        self.dropout = self.param.get("dropout", None)
        self.normalization = self.param.get("normalization", None)

        # bayesian network or not
        self.bayesian = self.param.get("bayesian", False)
        self.bayesian_layers = []

        # Build the network
        self.net = self.build_net()

    def build_net(self):
        """
        Method to build the Resnet blocks with the defined specifications
        """
        layers = []

        first_layer = nn.Linear(self.encode_x_dim + self.encode_c_dim + self.encode_t_dim, self.intermediate_dim)
        layers.append(first_layer)
        layers.append(nn.SiLU())

        for _ in range(1, self.layers - 1):
            linear = nn.Linear(self.intermediate_dim, self.intermediate_dim)
            layers.append(linear)
            if self.normalization is not None:
                layers.append(getattr(nn, self.normalization)(self.intermediate_dim))
            if self.dropout is not None:
                layers.append(nn.Dropout(p=self.dropout))
            layers.append(nn.SiLU())

        last_layer_class = VBLinear if self.bayesian else nn.Linear
        last_layer = last_layer_class(self.intermediate_dim, self.out_dim)
        layers.append(last_layer)

        if self.bayesian:
            self.bayesian_layers.append(last_layer)

        return nn.Sequential(*layers)

    def forward(self, x, t, condition=None):

        self.kl = 0

        t_embed = self.embed_t(t)
        x_embed = self.embed_x(x)

        if condition is not None:
            c_embed = self.embed_c(condition)
            nn_in = torch.cat([x_embed, c_embed, t_embed], 1)
        else:
            nn_in = torch.cat([x_embed, t_embed], 1)

        nn_out = self.net(nn_in)

        for bay_layer in self.bayesian_layers:
            self.kl += bay_layer.KL()

        return nn_out


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x * self.W * 2 * torch.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=1)
