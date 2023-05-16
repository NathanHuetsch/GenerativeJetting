import numpy as np
import time
import Source.Networks
from Source.Util.util import get, get_device
import torch
import torch.nn.functional as F
import torch.distributions as D

'''
Rational quadratic spline (RQS) implementation, used for the autoregressive transformer
with spline likelihoods (AutoReGRQS)

Work on this not finished, but code should run
'''

class RQS:
    
    def __init__(self, params, obs_ranges, theta):
        self.rqs_n = params["rqs_n"] #maybe other initialization
        self.device = get_device()
        self.theta = theta
        self.obs_ranges = obs_ranges.to(self.device)
        self.prepare()

    def prepare(self):
        theta_x = self.theta[:,:,:self.rqs_n]
        theta_y = self.theta[:,:,self.rqs_n:2*self.rqs_n]
        theta_delta = self.theta[:,:,2*self.rqs_n:]

        zeros = torch.zeros(self.theta.size(0), self.theta.size(1), 1, device=self.device)

        deltatilde = F.softplus(theta_delta)
        self.delta = torch.cat((zeros, deltatilde, zeros), dim=-1)
        
        ytilde = F.softmax(theta_x, dim=-1)
        self.y = torch.cat((zeros, torch.cumsum(ytilde, dim=-1)), dim=-1)

        xmin = self.obs_ranges[:,0].reshape(1, self.theta.size(1), 1).expand(self.theta.size(0), self.theta.size(1), 1)
        xmax = self.obs_ranges[:,1].reshape(1, self.theta.size(1), 1).expand(self.theta.size(0), self.theta.size(1), 1)
        xtilde = torch.cat((zeros, F.softmax(theta_x, dim=-1)), dim=-1)
        self.x = xmin + (xmax - xmin) * torch.cumsum(xtilde, dim=-1)
        '''
        self.x = torch.linspace(self.obs_ranges[0,0], self.obs_ranges[0,1], self.rqs_n+1)\
                 .expand(self.theta.size(0), 1, self.rqs_n+1).expand(self.theta.size(0), self.theta.size(1), self.rqs_n+1)
        print(self.x.size())

        self.y = torch.linspace(0, 1, self.rqs_n+1).reshape(1, 1, self.rqs_n+1)\
                 .expand(self.theta.size(0), 1, self.rqs_n+1).expand(self.theta.size(0), self.theta.size(1), self.rqs_n+1)

        self.delta = torch.ones(self.theta.size(0), self.theta.size(1), self.rqs_n+1)
        '''
        #print(self.x.size(), self.y.size(), self.delta.size())
        #print(self.x[0,0,:], self.y[0,0,:], self.delta[0,0,:])
    '''
    def __init__(self, params, theta):
        self.rqs_n = params["rqs_n"] #maybe other initialization
        self.device = get_device()
        self.theta = theta
        self.prepare()

    def prepare(self):
        xmin = self.theta[:,:,[0]]
        theta_x = self.theta[:,:,1:self.rqs_n+1]
        theta_y = self.theta[:,:,self.rqs_n+1:2*self.rqs_n+1]
        theta_delta = self.theta[:,:,2*self.rqs_n+1:]

        deltatilde = F.softplus(theta_delta)
        zeros = torch.zeros(self.theta.size(0), self.theta.size(1), 1, device=self.device)
        self.delta = torch.cat((zeros, deltatilde, zeros), dim=-1)

        xtilde = torch.cat((xmin, F.softplus(theta_x)), dim=-1)
        self.x = torch.cumsum(xtilde, dim=-1)
        
        ytilde = F.softmax(theta_x, dim=-1)
        self.y = torch.cat((zeros, torch.cumsum(ytilde, dim=-1)), dim=-1)

        #print(self.x[0,0,:])
        #print(self.x.size(), self.y.size(), self.delta.size())
        #print(self.x[0,0,:], self.y[0,0,:], self.delta[0,0,:])
    '''

    def log_prob(self, value):
        value1 = value.reshape(value.size(0), value.size(1), 1).expand(value.size(0), value.size(1), self.x.size(2))
        test = self.x < value1
        idx = torch.argmax(torch.cumsum(test, dim=-1), dim=-1) #=k
        #print(test[0,0,:], idx[0,0])
        
        xk = self.x[torch.arange(idx.size(0))[:, None], torch.arange(idx.size(1))[None, :], idx]
        xkp = self.x[torch.arange(idx.size(0))[:, None], torch.arange(idx.size(1))[None, :], idx+1]
        dx = xkp - xk
        yk = self.y[torch.arange(idx.size(0))[:, None], torch.arange(idx.size(1))[None, :], idx]
        ykp = self.y[torch.arange(idx.size(0))[:, None], torch.arange(idx.size(1))[None, :], idx+1]
        dy = ykp - yk
        deltak = self.delta[torch.arange(idx.size(0))[:, None], torch.arange(idx.size(1))[None, :], idx]
        deltakp = self.delta[torch.arange(idx.size(0))[:, None], torch.arange(idx.size(1))[None, :], idx+1]
        xi = (value - xk) / dx
        sk = dy / dx
        
        prob1 = 2*torch.log(sk)
        prob2 = torch.log(xi**2 * deltakp + (1-xi)**2 * deltak + 2*(1-xi)*xi*sk)
        prob3 = -2*torch.log(sk + (deltakp + deltak - 2*sk)*xi*(1-xi))
        prob = prob1 + prob2 + prob3
        #pdf = sk**2 * (deltakp*xi**2 + 2*sk*xi*(1-xi) +deltak*(1-xi)**2) \
        #      / (sk + (deltakp + deltak - 2*sk)*xi*(1-xi))**2
        #prob = torch.log(pdf)
        #print(prob1[:10,0], prob2[:10,0], prob3[:10,0], prob[:10,0])
        return prob

    def cdf(self, value):
        value1 = value.reshape(value.size(0), value.size(1), 1).expand(value.size(0), value.size(1), self.x.size(2))
        idx = torch.argmax(torch.cumsum(self.x < value1, dim=-1), dim=-1) #=k
        
        xk = self.x[torch.arange(idx.size(0))[:, None], torch.arange(idx.size(1))[None, :], idx]
        xkp = self.x[torch.arange(idx.size(0))[:, None], torch.arange(idx.size(1))[None, :], idx+1]
        dx = xkp - xk
        yk = self.y[torch.arange(idx.size(0))[:, None], torch.arange(idx.size(1))[None, :], idx]
        ykp = self.y[torch.arange(idx.size(0))[:, None], torch.arange(idx.size(1))[None, :], idx+1]
        dy = ykp - yk
        deltak = self.delta[torch.arange(idx.size(0))[:, None], torch.arange(idx.size(1))[None, :], idx]
        deltakp = self.delta[torch.arange(idx.size(0))[:, None], torch.arange(idx.size(1))[None, :], idx+1]
        xi = (value - xk) / dx
        sk = dy / dx

        cdf = yk + (ykp - yk)*(sk * xi**2 + deltak*xi*(1-xi)) / (sk + (deltakp + deltak - 2*sk)*xi*(1-xi))
        return cdf

    def icdf(self, value):
        value1 = value.reshape(value.size(0), value.size(1), 1).expand(value.size(0), value.size(1), self.x.size(2))
        idx = torch.argmax(torch.cumsum(self.x < value1, dim=-1), dim=-1) #=k
        
        xk = self.x[torch.arange(idx.size(0))[:, None], torch.arange(idx.size(1))[None, :], idx]
        xkp = self.x[torch.arange(idx.size(0))[:, None], torch.arange(idx.size(1))[None, :], idx+1]
        dx = xkp - xk
        yk = self.y[torch.arange(idx.size(0))[:, None], torch.arange(idx.size(1))[None, :], idx]
        ykp = self.y[torch.arange(idx.size(0))[:, None], torch.arange(idx.size(1))[None, :], idx+1]
        dy = ykp - yk
        deltak = self.delta[torch.arange(idx.size(0))[:, None], torch.arange(idx.size(1))[None, :], idx]
        deltakp = self.delta[torch.arange(idx.size(0))[:, None], torch.arange(idx.size(1))[None, :], idx+1]

        sk = dy / dx
        a = dy * (sk - deltak) + (value - yk)*(deltakp + deltak - 2*sk)                                            
        b = dy * deltak - (value - yk) * (deltakp + deltak - 2*sk)
        c = - sk * (value - yk)
        xi = 2*c/(-b - (b**2 - 4*a*c)**.5)
        #xi = (-b + (b**2 - 4*a*c)**.5) / (2*a)
        x_value = xk + xi * dx
        return x_value
    
    def sample_1(self):
        rand_x = torch.rand(self.x.size(0), self.x.size(1), device=self.device)
        samples = self.icdf(rand_x)
        return samples
