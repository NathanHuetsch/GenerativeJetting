import numpy as np
import torch
from Source.Networks.transformer import GPT
from Source.Util.util import get_device, get
from Source.Models.ModelBase import GenerativeModel

from torch.nn import functional as F

class GPTwrapper(GenerativeModel):
    """
     Class for GPT 
    """

    def __init__(self, params):
        super().__init__(params)
        self.block_size = params["block_size"]

    def build_net(self):
        """
        Build the transformer
        """
        return GPT(self.params).to(self.device)

    def batch_loss(self, x):
        """
        Loss function for autoregressive model
        TBD: Extension by start-of-sequence token
        """
        idx = x[:, :-1] 
        targets = x[:, 1:]
        logits = self.net(idx)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        return loss

    
    def sample_n(self, n_samples, data):
        """
        Event generation for autoregressive model
        TBD: Extension by start-of-sequence token
        """
        self.eval()

        idx = data[torch.randperm(len(data[:,0]))[:n_samples]][:, [0]]

        batch_size = get(self.params, "batch_size", 8192)
        temperature = 1.0
        top_k = None
        do_sample = True

        for _ in range(self.block_size):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits = self.net(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx.detach().cpu().numpy()
