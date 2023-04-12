import numpy as np
import time
import Source.Networks
from Source.Util.util import get
from Source.Models.ModelBase import GenerativeModel
import torch
import torch.nn.functional as F
import torch.distributions as D
import scipy.stats as stats
from Source.Util.util import get, get_device
from Source.Util.NNdistribution import NNdistribution
import matplotlib.pyplot as plt

class AutoRegNN(GenerativeModel):
    """
    Implementation of an autoregressive transformer model following the minimal
    GPT implementation in https://github.com/karpathy/minGPT.
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
        self.n_conditions = get(params, "n_conditions", n_head * n_per_head)
        params["n_conditions"] = self.n_conditions
        if out:
            print(f"Model AutoRegNN hyperparameters: n_head={n_head}, n_per_head={n_per_head}, n_blocks={n_blocks}, "
              f"intermediate_fac={intermediate_fac}, n_conditions={self.n_conditions}")
        self.trained_before = False
        
        params["vocab_size"] = self.n_conditions
        params["block_size"] = params["dim"]
        self.block_size = params["block_size"]

        super().__init__(params)
        if out:
            print(f"Bayesianization hyperparameters: bayesian={self.bayesian}, prior_prec={get(self.params, 'prior_prec', 1.)}, iterations={self.iterations}")
        if self.conditional:
            raise ValueError("conditional=True not implemented for autoregressive models")
        self.enet = Source.Networks.EnergyNet(params, out=False).to(self.device)

    def build_net(self):
        """Build the network"""
        return Source.Networks.attnet(self.params).to(self.device)

    def set_obs_ranges_NN(self):
        if self.istoy:
            self.obs_ranges_NN = self.obs_ranges
        else:
            self.obs_ranges_NN = torch.zeros(self.train_loader.dataset.size(1), 2)
            for i in range(self.train_loader.dataset.size(1)):
                self.obs_ranges_NN[i,0] = torch.min(self.train_loader.dataset[:,i])
                self.obs_ranges_NN[i,1] = torch.max(self.train_loader.dataset[:,i])
        self.trained_before = True
    
    def batch_loss(self, x, conditional=False, getMore=False):
        """
        Loss function for autoregressive model
        TBD: Implement conditional
        :x: Training data in shape (batch_size, block_size)
        :returns: torch loss object
        """
        #set self.obs_ranges_NN (need training data for this which is not available in __init__)
        if not self.trained_before: 
            self.set_obs_ranges_NN()
        
        x = torch.cat((self.n_jets*torch.ones(x.size(0), 1, device=self.device), x), dim=1)
        idx = x[:, :-1]
        targets = x[:, 1:]

        #t0 = time.time()
        conditions = self.net(idx)
        #t1 = time.time()
        NNdistr = NNdistribution(self.enet, self.obs_ranges_NN, self.params, conditions)
        #t2 = time.time()
        logprob = NNdistr.log_prob(targets)
        #t3 = time.time()

        #print(f"Training: GPT {(t1-t0)/(t3-t0):.2e}s vs enet prepare {(t2-t1)/(t3-t0):.2e}s vs enet eval {(t3-t2)/(t3-t0):.2e}")

        # log Likelihood loss
        loss = -logprob.mean()
        self.regular_loss.append(-logprob.mean().detach().cpu().numpy())

        # KL loss
        if self.bayesian or self.iterations > 1:
            loss += self.net.KL() / len(self.data_train)
            self.kl_loss.append( (self.net.KL() / len(self.data_train)).detach().cpu().tolist())
        

        if getMore: #TBD
            raise ValueError("batch_loss: getMore not implemented for AutoRegNN")
        return loss
    
    def sample_n(self, n_samples, conditional=False, prior_samples=None, con_depth=0):
        """
        Event generation for autoregressive model
        TBD: Implement conditional
        :n_samples: Number of samples to be generated
        :returns: Generated samples in the shape (n_samples, block_size)
        """
        if not self.trained_before: 
            self.set_obs_ranges_NN()
        
        if self.net.bayesian != 0:
            self.net.map = get(self.params, "fix_mu", False)
        if self.net.bayesian == 1 or self.net.bayesian == 2 or self.net.bayesian == 3:
            for i in range(self.net.n_blocks):
                self.net.transformer.h[i].mlp.c_fc.random = None
                self.net.transformer.h[i].mlp.c_proj.random = None
        if self.net.bayesian == 2 or self.net.bayesian == 3:
            for i in range(self.net.n_blocks):
                self.net.transformer.h[i].attn.c_attn.random = None
                self.net.transformer.h[i].attn.c_proj.random = None
        if self.net.bayesian == 3 or self.net.bayesian == 4:
            self.net.lm_head.random = None
        if self.net.bayesian == 3:
            self.net.transformer.wte.random = None
        self.eval()

        n_batches = int(n_samples / self.batch_size_sample)+1
        sample= np.zeros((0, self.block_size), dtype="int")
        for i in range(n_batches):
            t0=time.time()

            idx = self.n_jets * torch.ones(self.batch_size_sample, 1, dtype=torch.int, device=self.device).float()            
            for iBlock in range(self.block_size):
                #t0 = time.time()
                conditions = self.net(idx)
                #t1 = time.time()
                NNdistr = NNdistribution(self.enet, torch.Tensor(self.obs_ranges_NN)[[-1]],
                                         self.params, conditions[:,[-1]])
                #t2 = time.time()
                idx_next = NNdistr.sample_1()
                #t3 = time.time()
                #print(f"Sampling: GPT {(t1-t0)/(t3-t0):.2e}s vs enet prepare {(t2-t1)/(t3-t0):.2e}s vs enet eval {(t3-t2)/(t3-t0):.2e}")

                idx = torch.cat((idx, idx_next), dim=1)
            sample = np.append(sample, idx[:,1:].detach().cpu().numpy(), axis=0)

            if(i==0): # sampling time estimate after first sampling
                t1=time.time()
                dtEst = (t1-t0)*n_batches
                print(f"Sampling time estimate: {dtEst:.2f} s = {dtEst/60:.2f} min")

        sample = sample[:n_samples]
        return sample
