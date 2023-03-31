import torch
import torch.nn as nn
import math
from torch.nn import functional as F
from Source.Networks.vblinear import VBLinear
from Source.Networks.attnet import TransformerBlock
            
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
                if self.bayesian==3 else nn.Linear(1, self.intermediate_dim), 
            wpe = nn.Embedding(self.block_size, self.intermediate_dim),
            drop = nn.Dropout(self.embd_pdrop),
            h = nn.ModuleList([TransformerBlock(params) for _ in range(self.n_blocks)]),
            ln_f = nn.LayerNorm(self.intermediate_dim),
        ))
        self.lm_head = VBLinear(self.intermediate_dim, self.vocab_size) \
                       if self.bayesian==3 or self.bayesian==4 \
                       else nn.Linear(self.intermediate_dim, self.vocab_size)

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
            
        if self.bayesian == 3 or self.bayesian == 4:
            kl+= self.lm_head.KL()
        if self.bayesian == 3:
            kl += self.transformer.wte.KL()
            
        if self.bayesian == 0 and self.iterations > 1: #ensemble regularization
            wte_KL = .5 * self.prior_prec * self.transformer.wte.weight.sum()
            lm_head_KL = .5 * self.prior_prec * self.lm_head.weight.sum()
            kl += wte_KL + lm_head_KL
        return kl
