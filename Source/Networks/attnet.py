import torch
import torch.nn as nn
import math
from torch.nn import functional as F
from Source.Networks.vblinear import VBLinear

'''
This is a minimal implementation of the autoregressive transformer architecture used in the GPT models,
the code is essentially copied from https://github.com/karpathy/minGPT
'''

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
        self.block_size = params["block_size"]
        self.attn_pdrop = params.get("attn_pdrop", 0.1)
        self.resid_pdrop = params.get("resid_pdrop", 0.1)
        self.bayesian = params.get("bayesian", 0)
        self.prior_prec = params.get("prior_prec", 1.)
        self.allowflash = params.get("allowflash", True) and hasattr(F, "scaled_dot_product_attention")

        self.c_attn = VBLinear(self.intermediate_dim, 3 * self.intermediate_dim, prior_prec = self.prior_prec) \
                      if self.bayesian==2 or self.bayesian==3 \
                      else nn.Linear(self.intermediate_dim, 3 * self.intermediate_dim)
        self.c_proj = VBLinear(self.intermediate_dim, self.intermediate_dim, prior_prec = self.prior_prec) \
                      if self.bayesian>=2 or self.bayesian==3 \
                      else nn.Linear(self.intermediate_dim, self.intermediate_dim)
            
        if not self.allowflash:
            self.register_buffer("bias", torch.tril(torch.ones(self.block_size, self.block_size))
                                     .view(1, 1, self.block_size, self.block_size))
        
        self.attn_dropout = nn.Dropout(self.attn_pdrop)
        self.resid_dropout = nn.Dropout(self.resid_pdrop)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (intermediate_dim)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.intermediate_dim, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.allowflash:
            # efficient attention using Flash Attention CUDA kernels
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.attn_pdrop, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

    def KL(self):
        if self.bayesian==2 or self.bayesian==3:
            return self.c_attn.KL() + self.c_proj.KL()
        else:
            return 0.

class TransformerBlock(nn.Module):
    """ A transformer block, consisting of a CausalSelfAttention block and a multilayer perceptron"""

    def __init__(self, params):
        super().__init__()
        self.intermediate_dim = params["intermediate_dim"]
        self.intermediate_fac = params.get("intermediate_fac", 4)
        self.resid_pdrop = params.get("resid_pdrop", 0.1)
        self.bayesian = params.get("bayesian", 0)
        self.prior_prec = params.get("prior_prec", 1.)
        
        self.ln_1 = nn.LayerNorm(self.intermediate_dim)
        self.attn = CausalSelfAttention(params)
        self.ln_2 = nn.LayerNorm(self.intermediate_dim)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = VBLinear(self.intermediate_dim, self.intermediate_fac*self.intermediate_dim, self.prior_prec) \
                if self.bayesian==1 or self.bayesian==2 or self.bayesian==3 \
                else nn.Linear(self.intermediate_dim, self.intermediate_fac * self.intermediate_dim),
            c_proj  = VBLinear(self.intermediate_dim*self.intermediate_fac, self.intermediate_dim, self.prior_prec) \
                if self.bayesian==1 or self.bayesian==2 or self.bayesian==3 \
                else nn.Linear(self.intermediate_dim*self.intermediate_fac, self.intermediate_dim),
            act     = NewGELU(),
            dropout = nn.Dropout(self.resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))

    # forward using the ResNet concept
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x
    def KL(self):
        kl = 0.
        kl += self.attn.KL()
        if self.bayesian==1 or self.bayesian==2 or self.bayesian==3:
            kl += self.mlp.c_fc.KL() + self.mlp.c_proj.KL()
        return kl
