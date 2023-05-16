import numpy as np
import time
import Source.Networks
from Source.Util.util import get
from Source.Models.ModelBase import GenerativeModel
import torch
from torch.nn import functional as F

class AutoRegBinned(GenerativeModel):
    """
    Autoregressive transformer with binned likelihood
    Implementation based on https://github.com/karpathy/minGPT
    """
    def __init__(self, params, out=True):
        self.bayesian = get(params, "bayesian", 0)
        n_blocks = get(params, "n_blocks", None)
        assert n_blocks is not None, "build_model: n_blocks not specified"
        n_head = get(params, "n_head", None)
        assert n_head is not None, "build_model: n_head not specified"
        n_per_head = get(params, "n_per_head", None)
        assert n_per_head is not None, "build_model: n_per_head not specified"
        intermediate_fac = get(params, "intermediate_fac", None)
        assert intermediate_fac is not None, "build_model: intermediate_fac not specified"
        params["intermediate_dim"] = n_head * n_per_head
        n_bins = get(params, "n_bins", n_head * n_per_head)
        self.n_bins = n_bins
        if out:
            print(f"Build model AutoRegBinned parameters: n_head={n_head}, n_per_head={n_per_head}, n_blocks={n_blocks}, "
                  f"intermediate_fac={intermediate_fac}, n_bins={n_bins}")
        
        params["vocab_size"] = self.n_bins
        params["block_size"] = params['dim']
        self.block_size = params["block_size"]
        super().__init__(params)
        if out:
            print(f"Bayesianization hyperparameters: bayesian={self.bayesian}, prior_prec={get(self.params, 'prior_prec', 1.)}, iterations={self.iterations}")

        if self.params["discretize"]==0:
            raise ValueError("AutoReg.__init__: No discretization, please set discretize to a non-zero value")
        if self.conditional:
            raise ValueError("AutoReg.__init__: conditional=True not implemented for autoregressive models")

    def build_net(self):
        """Build the network"""
        return Source.Networks.attnetBinned(self.params).to(self.device)

    def batch_loss(self, x, conditional=False, getMore=False):
        """
        Loss function for autoregressive transformer
        conditional is needed for conditional DM/INN
        getMore also returns bin likelihoods and event likelihoods
        """
        idx = x[:, :-1]             
        targets = x[:, 1:]          
        logits = self.net(idx)

        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        self.regular_loss.append(F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                                 targets.reshape(-1), ignore_index=-1).detach().cpu().numpy())

        if self.bayesian:
            loss += self.net.KL() / len(self.data_train)
            self.kl_loss.append( (self.net.KL() / len(self.data_train)).detach().cpu().tolist())

        if getMore:
            logits_targets = torch.zeros(targets.size())
            for i in range(np.shape(targets)[0]): #TBD: Implement this more efficiently
                for j in range(np.shape(targets)[1]):
                    logits_targets[i,j] = logits[i,j,targets[i,j]]
            logLikelihood = torch.sum(F.log_softmax(logits_targets, dim=-1), dim=-1)
            return loss, torch.exp(logLikelihood), logits
        else:
            return loss
    
    def sample_n(self, n_samples, conditional=False, prior_samples=None, con_depth=0):
        """
        Event generation for autoregressive model
        conditional, prior_samples, con_depth needed for conditional DM/INN
        """
        
        self.eval()

        n_batches = int(n_samples / self.batch_size)+1
        sample= np.zeros((0, self.block_size+1), dtype="long")
        for i in range(n_batches):
            t0=time.time()
            
            idx = self.n_jets * torch.ones(self.batch_size, 1, dtype=torch.int, device=self.device)
            for _ in range(self.block_size):
                logits = self.net(idx) # forward the model to get the logits for the index in the sequence
                
                logits = logits[:, -1, :] # pluck the logits at the final step and scale by desired temperature
                probs = F.softmax(logits, dim=-1) # apply softmax to convert logits to (normalized) probabilities

                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1) # append sampled index to the running sequence and continue
            sample = np.append(sample, idx.detach().cpu().numpy(), axis=0)

            if(i==0): # sampling time estimate after first sampling
                t1=time.time()
                dtEst = (t1-t0)*n_batches
                print(f"Sampling time estimate: {dtEst:.2f} s = {dtEst/60:.2f} min")
                
        sample = sample[:n_samples]

        return sample
