import torch
import torch.nn as nn
import math
from Source.Networks.vblinear import VBLinear
from Source.Networks.attnet import TransformerBlock
from Source.Util.util import get

class attnetBinned(nn.Module):
    """
    Autoregressive transformer model, following the GPT architecture
    This version takes discrete inputs (i.e. binned data) and returns normalized logits (bin likelihoods)
    """

    def __init__(self, params):
        super().__init__()
        self.params = params

        self.vocab_size = params["vocab_size"]
        self.block_size = params["block_size"]
        self.n_blocks = params["n_blocks"]
        self.n_head = params["n_head"]
        self.intermediate_dim = params["intermediate_dim"]
        self.embd_pdrop = params.get("embd_pdrop", 0.1)
        self.bayesian = params.get("bayesian", 0)
        self.prior_prec = params.get("prior_prec", 1.)

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(self.vocab_size, self.intermediate_dim),
            wpe = nn.Embedding(self.block_size, self.intermediate_dim),
            drop = nn.Dropout(self.embd_pdrop),
            h = nn.ModuleList([TransformerBlock(params) for _ in range(self.n_blocks)]),
            ln_f = nn.LayerNorm(self.intermediate_dim),
        ))
        self.lm_head = VBLinear(self.intermediate_dim, self.vocab_size) \
                       if self.bayesian==3 or self.bayesian==4 else nn.Linear(self.intermediate_dim, self.vocab_size)

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * self.n_blocks))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx):
        device = idx.device
        b, t = idx.size() #batch and token length
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)
        
        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        return logits

    def KL(self):
        kl = 0.
        for i in range(self.n_blocks):
            kl += self.transformer.h[i].KL()

        if self.bayesian == 3 or self.bayesian == 4:
            kl+= self.lm_head.KL()
            
        if self.bayesian == 0 and self.iterations > 1: #ensemble regularization
            wte_KL = .5 * self.prior_prec * self.transformer.wte.weight.sum()
            lm_head_KL = .5 * self.prior_prec * self.lm_head.weight.sum()
            kl += wte_KL + lm_head_KL
        return kl

    def reset_BNN(self):
        if self.bayesian != 0:
            self.map = get(self.params, "fix_mu", False)
        if self.bayesian == 1 or self.bayesian == 2 or self.bayesian == 3:
            for i in range(self.n_blocks):
                self.transformer.h[i].mlp.c_fc.random = None
                self.transformer.h[i].mlp.c_proj.random = None
        if self.bayesian == 2 or self.bayesian == 3:
            for i in range(self.n_blocks):
                self.transformer.h[i].attn.c_attn.random = None
                self.transformer.h[i].attn.c_proj.random = None
        if self.bayesian == 3 or self.bayesian == 4:
            self.lm_head.random = None    
