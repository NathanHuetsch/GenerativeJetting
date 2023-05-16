import numpy as np
import time
import Source.Networks
from Source.Util.util import get, get_device
import torch
import torch.nn.functional as F
import torch.distributions as D
from scipy.stats import poisson

'''
Distribution parametrized by a summary network,
used for the autoregressive transformer with summary network likelihood (AutoRegNN)

Work on this not finished, but code should run
'''

class NNdistribution(torch.distributions.distribution.Distribution):
    def __init__(self, enet, obs_ranges_NN, params, conditions):
        #super().__init__()
        self.enet = enet
        self.obs_ranges_NN = obs_ranges_NN
        self.params = params
        self.prec_mean = get(params, "enet_prec_mean", params["intermediate_dim"])
        self.conditions = conditions
        self.batch_size = conditions.size(0)
        self.block_size = conditions.size(1)
        self.device = get_device()
        self.prepare()
        
    def prepare(self):
        t0 = time.time()
        self.prec = poisson.rvs(self.prec_mean, size=1)[0] #sample resolution from poisson distribution
        #print(f"Precision: {self.prec}")
        self.x = torch.zeros(self.batch_size, self.block_size, self.prec, device=self.device)
        for iblock in range(self.block_size):
            x_one = torch.linspace(self.obs_ranges_NN[iblock][0], self.obs_ranges_NN[iblock][1], self.prec, device=self.device)
            self.x[:,iblock,:] = x_one.reshape(1, x_one.size(0)).expand(self.batch_size, x_one.size(0)) #torch.meshgrid(torch.zeros(self.batch_size), x_one) #ugly hack
        #t1 = time.time()
        self.energy = self.enet(self.x, self.conditions)
        #t2 = time.time()
        integrated_energy = torch.zeros(self.batch_size, self.block_size, self.prec, device=self.device)
        for i in range(self.prec):
            integrated_energy[:,:,i] = torch.trapezoid(torch.exp(-self.energy[:,:,:i+1]), dim=-1) #maybe even parallelize this?
            
        #t3 = time.time()  
        self.omega = integrated_energy[:,:,-1]
        self.cdf = integrated_energy / self.omega.reshape(self.omega.size(0), self.omega.size(1), 1)
        #t4 = time.time()
        #print(f"Time consumption: {(t1-t0)/(t4-t0):.2e} {(t2-t1)/(t4-t0):.2e} {(t3-t2)/(t4-t0):.2e} {(t4-t3)/(t4-t0):.2e}")

    @staticmethod
    def linInterp(x, y, idx0, value):
        a = (y[torch.arange(idx0.size(0))[:, None], torch.arange(idx0.size(1))[None, :], idx0+1]-y[torch.arange(idx0.size(0))[:, None], torch.arange(idx0.size(1))[None, :], idx0]) \
            /(x[torch.arange(idx0.size(0))[:, None], torch.arange(idx0.size(1))[None, :], idx0+1]-x[torch.arange(idx0.size(0))[:, None], torch.arange(idx0.size(1))[None, :], idx0])
        b = y[torch.arange(idx0.size(0))[:, None], torch.arange(idx0.size(1))[None, :], idx0] - a * x[torch.arange(idx0.size(0))[:, None], torch.arange(idx0.size(1))[None, :], idx0]
        return a * value + b

    # TBD: Use linear interpolation instead of rectangular interpolation
    def log_prob(self, value):
        value1 = value.reshape(value.size(0), value.size(1), 1).expand(value.size(0), value.size(1), self.x.size(2))
        test = self.x < value1
        idx = torch.argmax(torch.cumsum(test, dim=2), dim=2)
        
        #naive version: just evaluate at max value
        #ret = -self.energy[torch.arange(idx.shape[0])[:, None], torch.arange(idx.shape[1])[None, :], idx] \
        #      -torch.log(self.omega) #advanced indexing to avoid the for loop (copied from chatGPT, insane speedup)

        #improved version: Use linear interpolation
        ret = -self.linInterp(self.x, self.energy, idx, value) - torch.log(self.omega)
        return ret

    def icdf(self, value):
        value1 = value.reshape(value.size(0), value.size(1), 1).expand(value.size(0), value.size(1), self.x.size(2))
        test = self.cdf < value1
        idx = torch.argmax(torch.cumsum(test, dim=2), dim=2)
        return self.linInterp(self.cdf, self.x, idx, value)

    def sample_1(self):
        rand_x = torch.rand(self.x.size(0), self.x.size(1), device=self.device)
        samples = self.icdf(rand_x)
        return samples
