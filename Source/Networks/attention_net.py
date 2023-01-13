import torch
import torch.nn as nn
import math
from torch.nn import functional as F
from Source.Util.util import get_device


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, params):
        super().__init__()
        self.n_head = params["n_head"]
        self.intermediate_dim = params["intermediate_dim"]
        self.attn_pdrop = params.get("attn_pdrop", 0.1)
        self.resid_pdrop = params.get("resid_pdrop", 0.1)

        assert self.intermediate_dim % self.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(self.intermediate_dim, 3 * self.intermediate_dim)
        # output projection
        self.c_proj = nn.Linear(self.intermediate_dim, self.intermediate_dim)
        # regularization
        self.attn_dropout = nn.Dropout(self.attn_pdrop)
        self.resid_dropout = nn.Dropout(self.resid_pdrop)

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (intermediate_dim)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.intermediate_dim, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class TransformerBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, params):
        super().__init__()
        self.intermediate_dim = params["intermediate_dim"]
        self.intermediate_fac = params.get("intermediate_fac", 4)
        self.resid_pdrop = params.get("resid_pdrop", 0.1)

        self.ln_1 = nn.LayerNorm(self.intermediate_dim)
        self.attn = CausalSelfAttention(params)
        self.ln_2 = nn.LayerNorm(self.intermediate_dim)
        self.mlp = nn.ModuleDict(dict(
            c_fc=nn.Linear(self.intermediate_dim, self.intermediate_fac * self.intermediate_dim),
            c_proj=nn.Linear(self.intermediate_fac * self.intermediate_dim, self.intermediate_dim),
            act=NewGELU(),
            dropout=nn.Dropout(self.resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))  # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x


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
        self.n_con = self.param.get('n_con',0)
        self.dropout = self.param.get("dropout", None)
        self.normalization = self.param.get("normalization", None)
        self.activation = self.param.get("activation", "SiLU")
        self.conditional = self.param.get("conditional", False)

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
            TransformerBlock(param)
            for _ in range(self.n_blocks)])

        self.up_project = nn.Linear(self.dim + self.encode_t_dim + self.n_con,
                                    (self.dim + self.encode_t_dim + self.n_con) * self.intermediate_dim)
        self.down_project = nn.Linear((self.dim + self.encode_t_dim + self.n_con) * self.intermediate_dim,
                                      self.dim)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, x, t, condition=None):
        """
        forward method of our Resnet
        """
        if self.encode_t:
            t = self.embed(t)

        if self.conditional:
            add_input = torch.cat([t, condition], 1)
        else:
            add_input = t

        x = self.up_project(torch.cat([x, add_input], 1)).reshape(-1, self.dim + self.encode_t_dim + self.n_con,
                                                                  self.intermediate_dim)
        for block in self.blocks[:-1]:
            x = block(x) + x
        x = self.blocks[-1](x) + x
        x = self.down_project(x.reshape(-1, (self.dim + self.encode_t_dim + self.n_con) * self.intermediate_dim))
        return x


class AttentionNet2(nn.Module):
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
        self.n_con = self.param['n_con']
        self.dropout = self.param.get("dropout", None)
        self.normalization = self.param.get("normalization", None)
        self.activation = self.param.get("activation", "SiLU")
        self.conditional = self.param.get("conditional", False)
        self.device = self.param.get("device", get_device())

        # Use GaussianFourierProjection for the time if specified
        self.encode_t_scale = self.param.get("encode_t_scale", 30)
        self.t_embed = nn.Sequential(GaussianFourierProjection(embed_dim=self.intermediate_dim,
                                                             scale=self.encode_t_scale, device=self.device))
        self.c_embed = nn.Sequential(GaussianFourierProjection(embed_dim=self.intermediate_dim,
                                                               scale=self.encode_t_scale, input_dim=self.n_con, device=self.device))
                                   #nn.Linear(self.intermediate_dim, self.intermediate_dim))
        # Build the blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(param)
            for _ in range(self.n_blocks)])

        self.up_project = nn.Linear(self.dim, self.dim * self.intermediate_dim)
        self.down_project = nn.Linear(self.dim * self.intermediate_dim, self.dim)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, x, t, condition=None):
        """
        forward method of our Resnet
        """
        t = self.t_embed(t).unsqueeze(1)

        if self.conditional:
            condition = self.c_embed(condition).unsqueeze(1)
            add_input = t+condition
        else:
            add_input = t

        x = self.up_project(x).reshape(-1, self.dim, self.intermediate_dim)
        for block in self.blocks:
            x = block(x+add_input)+x
        x = self.down_project(x.reshape(-1, self.dim * self.intermediate_dim))
        return x


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, input_dim=1, scale=30., device=get_device()):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn((input_dim, embed_dim // 2)) * scale, requires_grad=False)
        self.W.to(device)

    def forward(self, x):
        x_proj = torch.matmul(x, self.W) * 2 * torch.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=1)