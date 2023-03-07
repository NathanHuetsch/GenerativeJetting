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
        self.iterations = params.get("iterations", 1)

        self.c_attn = VBLinear(self.intermediate_dim, 3 * self.intermediate_dim, prior_prec = self.prior_prec) \
                      if self.bayesian>=2 else nn.Linear(self.intermediate_dim, 3 * self.intermediate_dim)
        self.c_proj = VBLinear(self.intermediate_dim, self.intermediate_dim, prior_prec = self.prior_prec) \
                      if self.bayesian>=2 else nn.Linear(self.intermediate_dim, self.intermediate_dim)

        self.attn_dropout = nn.Dropout(self.attn_pdrop)
        self.resid_dropout = nn.Dropout(self.resid_pdrop)
        self.register_buffer("bias", torch.tril(torch.ones(self.block_size, self.block_size))
                                     .view(1, 1, self.block_size, self.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (intermediate_dim)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.intermediate_dim, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
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
        if self.bayesian >= 2:
            c_attn_KL = self.c_attn.KL()
            c_proj_KL = self.c_proj.KL()
        elif self.bayesian == 0 and self.iterations > 1:
            c_attn_KL = .5 * self.prior_prec * self.c_attn.weight.pow(2).sum()
            c_proj_KL = .5 * self.prior_prec * self.c_proj.weight.pow(2).sum()
        return c_attn_KL + c_attn_KL

class TransformerBlock(nn.Module):
    """ A transformer block, consisting of a CausalSelfAttention block and a multilayer perceptron"""

    def __init__(self, params):
        super().__init__()
        self.intermediate_dim = params["intermediate_dim"]
        self.intermediate_fac = params.get("intermediate_fac", 4)
        self.resid_pdrop = params.get("resid_pdrop", 0.1)
        self.bayesian = params.get("bayesian", 0)
        self.prior_prec = params.get("prior_prec", 1.)
        self.iterations = params.get("iterations", 1)

        self.ln_1 = nn.LayerNorm(self.intermediate_dim)
        self.attn = CausalSelfAttention(params)
        self.ln_2 = nn.LayerNorm(self.intermediate_dim)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = VBLinear(self.intermediate_dim, self.intermediate_fac*self.intermediate_dim, self.prior_prec) \
                if self.bayesian>=1 else nn.Linear(self.intermediate_dim, self.intermediate_fac * self.intermediate_dim),
            c_proj  = VBLinear(self.intermediate_dim*self.intermediate_fac, self.intermediate_dim, self.prior_prec) \
                if self.bayesian>=1 else nn.Linear(self.intermediate_dim*self.intermediate_fac, self.intermediate_dim),
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
        if self.bayesian >= 1:
            mlpc_fc_KL = self.mlp.c_fc.KL()
            mlpc_proj_KL = self.mlp.c_proj.KL()
        elif self.bayesian==0 and self.iterations > 1:
            mlpc_fc_KL = .5 * self.prior_prec * self.mlp.c_fc.weight.sum()
            mlpc_proj_KL = .5 * self.prior_prec * self.mlp.c_proj.weight.sum()
        return self.attn.KL() + mlpc_fc_KL + mlpc_proj_KL
            
class attnetGMM(nn.Module):
    """Autoregressive transformer model, following the GPT architecture"""

    def __init__(self, params):
        super().__init__()

        self.vocab_size = params["vocab_size"]
        self.in_size = 1
        self.block_size = params["block_size"]
        self.n_blocks = params["n_blocks"]
        self.n_head = params["n_head"]
        self.intermediate_dim = params["intermediate_dim"]
        self.n_gauss = params["n_gauss"]
        self.embd_pdrop = params.get("embd_pdrop", 0.1)
        self.bayesian = params.get("bayesian", False)
        self.prior_prec = params.get("prior_prec", 1.)
        self.iterations = params.get("iterations", 1)

        self.transformer = nn.ModuleDict(dict(
            wte = VBLinear(1, self.intermediate_dim, self.prior_prec) \
                if self.bayesian>=3 else nn.Linear(1, self.intermediate_dim), 
            wpe = nn.Embedding(self.block_size, self.intermediate_dim),
            drop = nn.Dropout(self.embd_pdrop),
            h = nn.ModuleList([TransformerBlock(params) for _ in range(self.n_blocks)]),
            ln_f = nn.LayerNorm(self.intermediate_dim),
        ))
        self.lm_head = VBLinear(self.intermediate_dim, self.vocab_size) \
                       if self.bayesian>=3 else nn.Linear(self.intermediate_dim, self.vocab_size)

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * self.n_blocks))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, VBLinear):
            pass #initialization already done in VBLinear.__init__
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        idx = idx.reshape(idx.size(0), idx.size(1), 1)
        tok_emb = self.transformer.wte(idx)        
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x).reshape(idx.size(0), idx.size(1), self.n_gauss, 3)

        mu = logits[:,:,:,0]
        sigma = torch.exp(logits[:,:,:,1]) #ensures positivity and slowly goes to zero
        weights = F.softmax(logits[:,:,:,2], dim=-1) #ensures normalization

        return mu, sigma, weights

    def KL(self):
        kl = 0.
        for i in range(self.n_blocks):
            kl += self.transformer.h[i].KL()
            
        if self.bayesian >= 3:
            wte_KL = self.transformer.wte.KL()
            lm_head_KL = self.lm_head.KL()
            kl += wte_KL + lm_head_KL
        if self.bayesian == 0 and self.iterations > 1:
            wte_KL = .5 * self.prior_prec * self.transformer.wte.weight.sum()
            lm_head_KL = .5 * self.prior_prec * self.lm_head.weight.sum()
            kl += wte_KL + lm_head_KL
        return kl
