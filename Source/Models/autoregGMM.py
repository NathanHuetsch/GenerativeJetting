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
    Implementation of an autoregressive transformer model following the minimal
    GPT implementation in https://github.com/karpathy/minGPT.
    """

    def __init__(self, params):
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
        n_gauss = get(params, "n_gauss", None)
        self.n_gauss = n_gauss
        self.l2_lambda = get(params, "l2_lambda", 0.)
        self.l2_p = get(params, "l2_p", 2)
        assert n_gauss is not None, "build_model: n_gauss not specified"
        print(f"Model AutoRegGMM hyperparameters: n_head={n_head}, n_per_head={n_per_head}, n_blocks={n_blocks}, "
              f"intermediate_fac={intermediate_fac}, n_gauss={n_gauss}")
        
        params["vocab_size"] = 3 * n_gauss
        params["block_size"] = params["dim"]
        self.block_size = params["block_size"]
        super().__init__(params)
        print(f"Bayesianization hyperparameters: bayesian={self.bayesian}, prior_prec={get(self.params, 'prior_prec', 1.)}, iterations={self.iterations}")

        if self.conditional:
            raise ValueError("conditional=True not implemented for autoregressive models")

    def build_net(self):
        """Build the network"""
        return Source.Networks.attnetGMM(self.params).to(self.device)
    
    def batch_loss(self, x, conditional=False, getMore=False):
        """
        Loss function for autoregressive model
        TBD: Implement conditional
        :x: Training data in shape (batch_size, block_size)
        :returns: torch loss object
        """
        x = torch.cat((self.n_jets*torch.ones(x.size(0), 1, device=self.device), x), dim=1)
        idx = x[:, :-1]
        targets = x[:, 1:]

        mu, sigma, weights = self.net(idx)
        mix = D.Categorical(weights)
        comp = D.Normal(mu, sigma)
        gmm = D.MixtureSameFamily(mix, comp)

        # log Likelihood loss
        loss = -gmm.log_prob(targets).mean()
        self.regular_loss.append(-gmm.log_prob(targets).mean().detach().cpu().numpy())

        # GMM weight regularization
        if self.l2_lambda > 0.:
            loss -= self.l2_lambda * self.n_gauss * torch.mean(weights.pow(self.l2_p))
            self.regularizeGMM_loss.append( (-self.l2_lambda * self.n_gauss * torch.mean(weights.pow(self.l2_p))).detach().cpu().tolist())

        # KL loss
        if self.bayesian:
            loss += self.net.KL() / len(self.data_train)
            self.kl_loss.append( (self.net.KL() / len(self.data_train)).detach().cpu().tolist())

        if getMore:
            logLikelihood = torch.sum(gmm.log_prob(targets), dim=-1)
            return loss, torch.exp(logLikelihood), mu, sigma, weights
        else:
            return loss
    
    def sample_n(self, n_samples, conditional=False, prior_samples=None, con_depth=0):
        """
        Event generation for autoregressive model
        TBD: Implement conditional
        :n_samples: Number of samples to be generated
        :returns: Generated samples in the shape (n_samples, block_size)
        """
        if self.net.bayesian >= 1:
            self.net.map = get(self.params, "fix_mu", False)
            for i in range(self.net.n_blocks):
                self.net.transformer.h[i].mlp.c_fc.random = None
                self.net.transformer.h[i].mlp.c_proj.random = None
        if self.net.bayesian >= 2:
            for i in range(self.net.n_blocks):
                self.net.transformer.h[i].attn.c_attn.random = None
                self.net.transformer.h[i].attn.c_proj.random = None
        if self.net.bayesian >= 3:
            self.net.transformer.wte.random = None
            self.net.lm_head.random = None
        self.eval()

        n_batches = int(n_samples / self.batch_size_sample)+1
        sample= np.zeros((0, self.block_size), dtype="int")
        for i in range(n_batches):
            t0=time.time()

            idx = self.n_jets * torch.ones(self.batch_size_sample, 1, dtype=torch.int, device=self.device).float()            
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

    def sample_n_bonus(self, n_samples, conditional=False, prior_samples=None, con_depth=0, prec=1000):
        '''
        Variant of sample_n that returns not only the samples, but also the generate Gaussian
        mixture pdf (probstotal) and the pdfs of all individual gaussians (probsindiv)
        '''
        assert n_samples <= self.batch_size_sample, "sample_n_bonus: Specified n_samples > batch_size. " \
               "This is probably not intended, as this function should be used for visualization only."
        
        self.eval()

        probstotal = np.zeros((n_samples, self.block_size, prec))                   # pdf of gaussian mixture
        probsindiv = np.zeros((n_samples, self.block_size, self.n_gauss, prec))     # pdfs of individual gaussians
        xs = np.zeros((self.block_size, prec))                                      # x-values of the pdfs
        
        idx = self.n_jets * torch.ones(self.batch_size_sample, 1, dtype=torch.int, device=self.device).float()
        for idim in range(self.block_size):
            mu, sigma, weights = self.net(idx)

            # generate distribution and next event (standard)                
            mix = D.Categorical(weights[:,-1,:])
            comp = D.Normal(mu[:,-1,:], sigma[:,-1,:])
            gmm = D.MixtureSameFamily(mix, comp)
            idx_next = gmm.sample_n(1).permute(1,0)
            idx = torch.cat((idx, idx_next), dim=1)

            # generate total pdf (using torch.distributions)
            probs = torch.zeros(self.batch_size_sample, prec, dtype=torch.float, device=self.device)
            vals = torch.linspace(torch.min(idx_next), torch.max(idx_next), prec)
            for ix in range(prec):
                x=vals[ix]
                probs[:,ix] = torch.exp(gmm.log_prob(x))
            probstotal[:,idim,:] = probs.detach().cpu().numpy()[:n_samples,:]
            xs[idim,:] = vals.detach().cpu().numpy()

            # generate individual pdfs (by hand)
            for igauss in range(self.n_gauss):
                for isample in range(n_samples):
                    probsindiv[isample,idim,igauss,:] = weights[isample,idim,igauss].detach().cpu().numpy() \
                                * stats.norm.pdf(xs[idim,:], loc=mu[isample,idim,igauss].detach().cpu().numpy(),
                                        scale=sigma[isample,idim,igauss].detach().cpu().numpy())
        sample = idx.detach().cpu().numpy()[:n_samples,:]
        return sample, xs, probstotal, probsindiv
