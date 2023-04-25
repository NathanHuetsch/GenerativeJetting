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
        self.out_dim = self.param.get("out_dim", self.dim)
        #self.block_out_dim = self.param.get("out_dim", self.out_dim)
        self.n_con = self.param["n_con"]
        self.layers_per_block = self.param["layers_per_block"]
        self.dropout = self.param.get("dropout", None)
        self.normalization = self.param.get("normalization", None)
        self.activation = self.param.get("activation", "SiLU")
        self.encode_t = self.param.get("encode_t", False)
        self.encode_x = self.param.get("encode_x", False)
        if self.encode_x:
            self.block_out_dim = self.encode_x_dim = self.param.get("encode_x_dim", 64)
            self.add_final = True
        else:
            self.block_out_dim = self.out_dim
            self.add_final = False
        self.conditional = self.param.get("conditional", False)
        self.embed_condition = self.param.get("embed_condition",False)
        self.bayesian = self.param.get("bayesian", False)
        self.bayesian_layers = []
        self.deter_layers = []
        self.prior_prec = self.param.get("prior_prec", 1.0)
        self.map = False

        # Use GaussianFourierProjection for the time if specified
        if self.encode_t:
            self.encode_t_scale = self.param.get("encode_t_scale", 30)
            self.encode_t_dim = self.param.get("encode_t_dim", 64)
            self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=self.encode_t_dim,
                                                                 scale=self.encode_t_scale),
                                       nn.Linear(self.encode_t_dim, self.encode_t_dim))
        else:
            self.encode_t_dim = 1

        if self.embed_condition:
            self.encode_c_dim = self.param.get("encode_c_dim", 64)
            self.embed_c = nn.Linear(self.n_con, self.encode_c_dim)
        else:
            self.encode_c_dim = self.n_con

        if self.encode_x:

            self.embed_x = nn.Linear(self.dim, self.encode_x_dim)
        else:
            self.encode_x_dim = self.dim
        # Build the Resnet blocks
        self.blocks = nn.ModuleList([
            self.make_block()
            for _ in range(self.n_blocks)])

        if self.add_final:
            if self.bayesian == 0:
                self.final = nn.Linear(self.block_out_dim, self.out_dim)
                self.deter_layers.append(self.final)
            else:
                self.final = VBLinear(self.block_out_dim, self.out_dim, prior_prec=self.prior_prec)
                self.bayesian_layers.append(self.final)

        if self.bayesian > 1:
            for block in self.blocks:
                block[-1].mu_w.data *= 0
                block[-1].bias.data *= 0
                #block[-1].logsig2_w.data *= 10**(-5)
        #else:
        #    # Initialize the weights in the last layer of each block as 0
        #    for block in self.blocks:
        #        block[-1].weight.data *= 0
        #        block[-1].bias.data *= 0

    def make_block(self):
        """
        Method to build the Resnet blocks with the defined specifications
        """
        if self.bayesian > 1:
            bays_layer = VBLinear(self.encode_x_dim + self.encode_c_dim + self.encode_t_dim, self.intermediate_dim,
                                  prior_prec=self.prior_prec)
            layers = [bays_layer, nn.SiLU()]
            self.bayesian_layers.append(bays_layer)

            for _ in range(1, self.layers_per_block - 1):
                bays_layer = VBLinear(self.intermediate_dim, self.intermediate_dim,prior_prec=self.prior_prec)
                layers.append(bays_layer)
                self.bayesian_layers.append(bays_layer)
                if self.normalization is not None:
                    layers.append(getattr(nn, self.normalization)(self.intermediate_dim))
                if self.dropout is not None:
                    layers.append(nn.Dropout(p=self.dropout))
                layers.append(getattr(nn, self.activation)())
            bays_layer = VBLinear(self.intermediate_dim, self.block_out_dim, prior_prec=self.prior_prec)
            layers.append(bays_layer)
            self.bayesian_layers.append(bays_layer)

        else:
            linear = nn.Linear(self.encode_x_dim + self.encode_c_dim + self.encode_t_dim, self.intermediate_dim)
            self.deter_layers.append(linear)
            layers = [linear, nn.SiLU()]

            for _ in range(1, self.layers_per_block-1):
                linear = nn.Linear(self.intermediate_dim, self.intermediate_dim)
                layers.append(linear)
                self.deter_layers.append(linear)
                if self.normalization is not None:
                    layers.append(getattr(nn, self.normalization)(self.intermediate_dim))
                if self.dropout is not None:
                    layers.append(nn.Dropout(p=self.dropout))
                layers.append(getattr(nn, self.activation)())

            linear = nn.Linear(self.intermediate_dim, self.block_out_dim)
            layers.append(linear)
            self.deter_layers.append(linear)

        return nn.Sequential(*layers)

    def forward(self, x, t, condition=None):
        """
        forward method of our Resnet
        """
        self.kl = 0
        if self.encode_t:
            t = self.embed(t)

        if self.conditional:
            if self.embed_condition:
                condition = self.embed_c(condition)
            add_input = torch.cat([t, condition], 1)
        else:
            add_input = t

        if self.encode_x:
            x = self.embed_x(x)

        for bay_layer in self.bayesian_layers:
            bay_layer.map = self.map

        for block in self.blocks[:-1]:
            x = x + block(torch.cat([x, add_input], 1))
        x = self.blocks[-1](torch.cat([x, add_input], 1))

        if self.add_final:
            x = nn.SiLU()(x)
            x = self.final(x)

        for bay_layer in self.bayesian_layers:
            self.kl += bay_layer.KL()

        for deter_layer in self.deter_layers:
            self.kl += (deter_layer.weight.pow(2)/self.prior_prec**2).sum()

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
