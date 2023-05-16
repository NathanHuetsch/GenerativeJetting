import numpy as np
import time
import Source.Networks
from Source.Util.util import get
from Source.Models.ModelBase import GenerativeModel
import torch
import torch.nn.functional as F
import torch.distributions as D
import scipy.stats as stats

class AutoRegGMM(GenerativeModel):
    """
    Autoregressive transformer with Gaussian mixture (GMM) likelihood
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
        n_gauss = get(params, "n_gauss", round(n_head * n_per_head/3))
        self.n_gauss = n_gauss
        if out:
            print(f"Model AutoRegGMM hyperparameters: n_head={n_head}, n_per_head={n_per_head}, n_blocks={n_blocks}, "
                  f"intermediate_fac={intermediate_fac}, n_gauss={n_gauss}")
        
        params["vocab_size"] = 3 * n_gauss
        params["block_size"] = params["dim"]
        self.block_size = params["block_size"]
        super().__init__(params)
        if out:
            print(f"Bayesianization hyperparameters: bayesian={self.bayesian}, prior_prec={get(self.params, 'prior_prec', 1.)}, iterations={self.iterations}")

        if self.conditional:
            raise ValueError("conditional=True not implemented for autoregressive models")

    def build_net(self):
        """Build the network"""
        return Source.Networks.attnetGMM(self.params).to(self.device)
    
    def batch_loss(self, x, conditional=False, getMore=False, pos=None, n_jets = None):
        """
        Loss function for autoregressive transformer
        conditional is needed for conditional DM/INN
        pos, n_jets is needed for conditional transformer
        getMore also returns GMM parameters and likelihoods
        """
            
        x = torch.cat((torch.zeros(x.size(0), 1, device=self.device), x), dim=1) #add start-of-sequence token
        idx = x[:, :-1]
        targets = x[:, 1:]

        mu, sigma, weights = self.net(idx, pos=pos, n_jets=n_jets)

        mix = D.Categorical(weights)
        comp = D.Normal(mu, sigma)
        gmm = D.MixtureSameFamily(mix, comp)
            
        loss = -gmm.log_prob(targets)
        
        # log Likelihood loss
        loss = loss.sum(dim=1).mean(dim=0) #sum over components to get single-event likelihoods, then average over those
        #self.regular_loss.append(loss.detach().cpu().numpy())

        # KL loss
        if self.bayesian or self.iterations > 1:
            loss += self.net.KL() / len(self.data_train)
            #self.kl_loss.append( (self.net.KL() / len(self.data_train)).detach().cpu().tolist())

        if getMore:
            logLikelihood = torch.sum(gmm.log_prob(targets), dim=-1)
            return loss, torch.exp(logLikelihood), mu, sigma, weights
        else:
            return loss
    
    def sample_n(self, n_samples, conditional=False, prior_samples=None, con_depth=0, n_jets=None, pos=None):
        """
        Event generation for autoregressive model
        conditional, prior_samples, con_depth are needed for the conditional DM/INN
        pos, n_jets are needed for the conditional transformer
        """
        self.net.reset_BNN()
        self.eval()

        if type(pos) is np.ndarray:
            nElements = len(pos) #could be smaller than self.block_size
        else:
            nElements = self.block_size

        n_batches = int(n_samples / self.batch_size_sample)+1
        sample= np.zeros((0, nElements), dtype="int")
        for i in range(n_batches):
            t0=time.time()

            idx = torch.zeros(self.batch_size_sample, 1, device=self.device)
            for iBlock in range(nElements):
                pos_i = pos[:iBlock+1] if type(pos) is np.ndarray else None
                mu, sigma, weights = self.net(idx, n_jets=n_jets, pos=pos_i)
                
                mix = D.Categorical(weights[:,-1,:])
                comp = D.Normal(mu[:,-1,:], sigma[:,-1,:]) #use gaussian mixture
                gmm = D.MixtureSameFamily(mix, comp)
                idx_next = gmm.sample((1,)).permute(1,0)     

                idx = torch.cat((idx, idx_next), dim=1)
            sample = np.append(sample, idx[:,1:].detach().cpu().numpy(), axis=0)

            if i==0: # sampling time estimate after first sampling step
                t1=time.time()
                dtEst = (t1-t0)*n_batches
                print(f"Sampling time estimate: {dtEst:.2f} s = {dtEst/60:.2f} min")
    
        sample = sample[:n_samples]
        return sample
