import torch
import torch.nn as nn


class AttentionNet(nn.Module):
    """
    AttentionNet. Not finished
    TODO: Finish implementation
    """
    def __init__(self, param):
        super().__init__()
        # Read in the network specifications from the params
        self.param = param
        self.n_blocks = param["n_blocks"]
        self.intermediate_dim = self.param["intermediate_dim"]
        self.dim = self.param["dim"]
        self.layers_per_block = self.param["layers_per_block"]
        self.dropout = self.param.get("dropout", None)
        self.normalization = self.param.get("normalization", None)
        self.activation = self.param.get("activation", "SiLU")

        self.encode_t = self.param.get("encode_t", False)

        # Use GaussianFourierProjection for the time if specified
        if self.encode_t:
            self.encode_t_scale = self.param.get("encode_t_scale", 30)
            self.encode_t_dim = self.param.get("encode_t_dim", 4)
            self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=self.encode_t_dim,
                                                                 scale=self.encode_t_scale),
                                       nn.Linear(self.encode_t_dim, self.encode_t_dim))
        else:
            self.encode_t_dim = 1

        # Build the blocks
        self.blocks = nn.ModuleList([
            self.make_attention_block()
            for _ in range(self.n_blocks)])
        # Initialize the weights in the last layer of each block as 0 (except for the last block)
        for block in self.blocks: #[:-1]:
            block[-1].weight.data *= 0
            block[-1].bias.data *= 0
        # Initialize the weights in the last layer of the last block to be small
        #self.blocks[-1][-1].weight.data *= 0.02
        #self.blocks[-1][-1].bias.data *= 0.02

    def make_attention_block(self):

        layers = [nn.Linear(self.dim + self.encode_t_dim, 3*(self.dim + self.encode_t_dim)),
                  nn.MultiheadAttention(embed_dim=self.dim + self.encode_t_dim, num_heads=1)]
        if self.normalization is not None:
            layers.append(getattr(nn, self.normalization)())
        layers.append(nn.Linear(self.self.dim + self.encode_t_dim, self.intermediate_dim))
        layers.append(getattr(nn, self.activation)())
        layers.append(nn.Linear(self.intermediate_dim, self.dim))
        if self.dropout is not None:
            layers.append(nn.Dropout(p=self.dropout))
        if self.normalization is not None:
            layers.append(getattr(nn, self.normalization)())
        return nn.ModuleList(layers)

    def forward(self, x, t):
        """
        forward method of our Resnet
        """
        if self.encode_t:
            t = self.embed(t)
        for block in self.blocks[:-1]:
            e = block[0](x)
            q, k, v = e.split(self.dim + self.encode_t_dim, dim=1)
            #e = block[]
            x = x + block(torch.cat([x, t], 1))
        x = self.blocks[-1](torch.cat([x, t], 1)) + x
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
