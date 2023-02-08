import torch
import torch.nn as nn
from Source.Networks.vblinear import VBLinear


class Resnet(nn.Module):
    """
    Simple Conditional Resnet class to build from a params dict
    """
    def __init__(self, param):
        super().__init__()
        # Read in the network specifications from the params
        self.param = param
        self.n_blocks = param["n_blocks"]
        self.intermediate_dim = self.param["intermediate_dim"]
        self.dim = self.param["dim"]
        self.n_con = self.param["n_con"]
        self.layers_per_block = self.param["layers_per_block"]
        self.dropout = self.param.get("dropout", None)
        self.normalization = self.param.get("normalization", None)
        self.activation = self.param.get("activation", "SiLU")
        self.encode_t = self.param.get("encode_t", False)
        self.conditional = self.param.get("conditional", False)
        self.embed_condition = self.param.get("embed_condition",False)
        self.bayesian = self.param.get("bayesian", False)
        self.kl = 0
        self.bayesian_layers = []
        self.prior_prec = self.param.get("prior_prec", 1.0)
        self.map = False

        # Use GaussianFourierProjection for the time if specified
        if self.encode_t:
            self.encode_t_scale = self.param.get("encode_t_scale", 30)
            self.encode_t_dim = self.param.get("encode_t_dim", 4)
            self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=self.encode_t_dim,
                                                                 scale=self.encode_t_scale),
                                       nn.Linear(self.encode_t_dim, self.encode_t_dim))
        else:
            self.encode_t_dim = 1
        if self.embed_condition:
            self.embed_c = nn.Sequential(nn.Linear(self.n_con+self.encode_t_dim,self.encode_c_dim),
                                         nn.Linear(self.encode_c_dim,self.encode_c_dim))
            self.encode_t_dim = 0
        else:
            self.encode_c_dim = self.n_con
        # Build the Resnet blocks
        self.blocks = nn.ModuleList([
            self.make_block()
            for _ in range(self.n_blocks)])

        if self.bayesian:
            for block in self.blocks:
                block[-1].mu_w.data *= 0
                block[-1].bias.data *= 0
                #block[-1].logsig2_w.data *= 10**(-5)
        else:
            # Initialize the weights in the last layer of each block as 0
            for block in self.blocks:
                block[-1].weight.data *= 0
                block[-1].bias.data *= 0

    def make_block(self):
        """
        Method to build the Resnet blocks with the defined specifications
        """
        if self.bayesian:
            bays_layer = VBLinear(self.dim + self.encode_c_dim + self.encode_t_dim, self.intermediate_dim,
                                  prior_prec=self.prior_prec,_map=self.map)
            layers = [bays_layer, nn.SiLU()]
            self.bayesian_layers.append(bays_layer)

            for _ in range(1, self.layers_per_block - 1):
                bays_layer = VBLinear(self.intermediate_dim, self.intermediate_dim,prior_prec=self.prior_prec,
                                      _map=self.map)
                layers.append(bays_layer)
                self.bayesian_layers.append(bays_layer)
                if self.normalization is not None:
                    layers.append(getattr(nn, self.normalization)())
                if self.dropout is not None:
                    layers.append(nn.Dropout(p=self.dropout))
                layers.append(getattr(nn, self.activation)())
            bays_layer = VBLinear(self.intermediate_dim, self.dim,prior_prec=self.prior_prec,_map=self.map)
            layers.append(bays_layer)
            self.bayesian_layers.append(bays_layer)

        else:
            layers = [nn.Linear(self.dim + self.encode_c_dim + self.encode_t_dim, self.intermediate_dim), nn.SiLU()]

            for _ in range(1, self.layers_per_block-1):
                layers.append(nn.Linear(self.intermediate_dim, self.intermediate_dim))
                if self.normalization is not None:
                    layers.append(getattr(nn, self.normalization)())
                if self.dropout is not None:
                    layers.append(nn.Dropout(p=self.dropout))
                layers.append(getattr(nn, self.activation)())
            layers.append(nn.Linear(self.intermediate_dim, self.dim))

        return nn.Sequential(*layers)

    def forward(self, x, t, condition=None):
        """
        forward method of our Resnet
        """
        if self.encode_t:
            t = self.embed(t)

        if self.conditional:
            add_input = torch.cat([t, condition], 1)
            if self.embed_condition:
                add_input = self.embed_c(add_input)
        else:
            add_input = t

        for block in self.blocks[:-1]:
            x = x + block(torch.cat([x, add_input], 1))
        x = self.blocks[-1](torch.cat([x, add_input], 1))

        for bay_layer in self.bayesian_layers:
            self.kl += bay_layer.KL()

        return x


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
