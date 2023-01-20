import numpy as np
import time
import Source.Networks
from Source.Util.util import get
from Source.Models.ModelBase import GenerativeModel
import torch
import torch.nn.functional as F
import torch.distributions as D

class AutoRegGMM(GenerativeModel):
    """
    Implementation of an autoregressive transformer model following the minimal
    GPT implementation in https://github.com/karpathy/minGPT.
    """

    def __init__(self, params):
        super().__init__(params)
        self.block_size = params["block_size"]
        self.n_gauss = params["n_gauss"]
        self.batch_size = get(self.params, "batch_size", 8192)

        n_blocks = get(self.params, "n_blocks", None)
        assert n_blocks is not None, "build_model: n_blocks not specified"
        n_head = get(self.params, "n_head", None)
        assert n_head is not None, "build_model: n_head not specified"
        n_per_head = get(self.params, "n_per_head", None)
        assert n_per_head is not None, "build_model: n_per_head not specified"
        intermediate_fac = get(self.params, "intermediate_fac", None)
        assert intermediate_fac is not None, "build_model: intermediate_fac not specified"
        self.params["intermediate_dim"] = n_head * n_per_head
        n_gauss = get(self.params, "n_gauss", None)
        assert n_gauss is not None, "build_model: n_gauss not specified"
        print(f"Build model AutoRegGMM with n_head={n_head}, n_per_head={n_per_head}, n_blocks={n_blocks}, "
              f"intermediate_fac={intermediate_fac}, n_gauss={n_gauss}")

        if get(params, "conditional", False):
            raise ValueError("conditional=True not implemented for autoregressive models")

    def build_net(self):
        """Build the network"""
        return Source.Networks.attnetGMM(self.params).to(self.device)
    
    def batch_loss(self, x, n_jets=2):
        """
        Loss function for autoregressive model
        :x: Training data in shape (batch_size, block_size)
        :returns: torch loss object
        """
        x = torch.cat((n_jets*torch.ones(x.size(0), 1, device=self.device), x), dim=1)
        idx = x[:, :-1]
        targets = x[:, 1:]

        mu, sigma, weights = self.net(idx)
        mix = D.Categorical(weights)
        comp = D.Normal(mu, sigma)
        gmm = D.MixtureSameFamily(mix, comp)
        loss = -gmm.log_prob(targets).mean()
        
        return loss
    
    def sample_n(self, n_samples, n_jets=2):
        """
        Event generation for autoregressive model
        :n_samples: Number of samples to be generated
        :n_jets: Number of jets to be generated (only relevant when training on multiple multiplicities at the same time)
        :returns: Generated samples in the shape (n_samples, block_size)
        """
        self.eval()

        n_batches = int(n_samples / self.batch_size)+1
        sample= np.zeros((0, self.block_size), dtype="int")
        for i in range(n_batches):
            t0=time.time()

            idx = n_jets * torch.ones(self.batch_size, 1, dtype=torch.int, device=self.device).float()            
            for iBlock in range(self.block_size):
                mu, sigma, weights = self.net(idx)
                
                mix = D.Categorical(weights[:,-1,:])
                comp = D.Normal(mu[:,-1,:], sigma[:,-1,:])
                gmm = D.MixtureSameFamily(mix, comp)
                idx_next = gmm.sample_n(1).permute(1,0)

                idx = torch.cat((idx, idx_next), dim=1)
            sample = np.append(sample, idx[:,1:].detach().cpu().numpy(), axis=0)

            if(i==0): # sampling time estimate after first sampling
                t1=time.time()
                dtEst = (t1-t0)*n_batches
                print(f"Sampling time estimate: {dtEst:.2f} s = {dtEst/60:.2f} min")
                
        sample = sample[:n_samples]
        return sample
