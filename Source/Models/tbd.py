import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp
from Source.Networks.resnet import Resnet
from Source.Util.util import get_device, get
from torch.utils.tensorboard import SummaryWriter
import os
from torch.utils.tensorboard import SummaryWriter
from Source.Models.ModelBase import GenerativeModel


class TBD(GenerativeModel):
    """
     Class for Trajectory Based Diffusion
     Inheriting from the GenerativeModel BaseClass
    """

    def __init__(self, params):
        super().__init__(params)

    def build_net(self):
        """
        Build the Resnet
        """
        return Resnet(self.params)

    def batch_loss(self, x):
        """
        Calculate batch loss as described by Peter
        TODO Add some more documentation?
        """
        t = torch.rand(x.size(0), 1, device=x.device)
        c = torch.cos(t * np.pi / 2)
        s = torch.sin(t * np.pi / 2)
        c_dot = -np.pi / 2 * s
        s_dot = np.pi / 2 * c
        x_1 = torch.randn_like(x)
        x_t = c * x + s * x_1
        x_t_dot = c_dot * x + s_dot * x_1
        drift = self.net(x_t, t)
        loss = 0.5 * torch.mean((drift - x_t_dot) ** 2)
        return loss

    def sample_n(self, n_samples):
        """
        Generate n_samples new samples.
        Start from Gaussian random noise and solve the reverse ODE to obtain samples
        """
        self.eval()

        batch_size = get(self.params, "batch_size", 8192)
        x_T = np.random.randn(n_samples+batch_size, self.dim)

        def f(t, x_t):
            x_t_torch = torch.Tensor(x_t).reshape((batch_size, self.dim)).to(self.device)
            t_torch = t * torch.ones_like(x_t_torch[:, [0]])
            with torch.no_grad():
                f_t = self.net(x_t_torch, t_torch).detach().cpu().numpy().flatten()
            return f_t

        events = []
        with torch.no_grad():
            for i in range(int(n_samples / batch_size) + 1):
                sol = solve_ivp(f, (1, 0), x_T[batch_size * i : batch_size * (i+1)].flatten())
                events.append(sol.y[:, -1].reshape(batch_size, self.dim))
        return np.concatenate(events, axis=0)[:n_samples]




