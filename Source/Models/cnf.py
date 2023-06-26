import numpy as np
import torch
from scipy.integrate import solve_ivp
import Source.Networks
from Source.Util.util import get
from Source.Models.ModelBase import GenerativeModel
from torchdiffeq import odeint_adjoint as odeint
import time


class CNF(GenerativeModel):
    """
     Class for Trajectory Based Diffusion
     Inheriting from the GenerativeModel BaseClass
    """

    def __init__(self, params):
        super().__init__(params)

        self.latent = get(self.params, "latent", "normal")
        if self.latent == "normal":
            print("Using latent normal")
            self.latent = torch.distributions.normal.Normal(0, 1)
        elif self.latent == "uniform":
            print("Using latent uniform")
            self.latent = torch.distributions.uniform.Uniform(low=0, high=1)

        else:
            print("Latent not recognised. Using normal")
            self.latent = torch.distributions.normal.Normal(0, 1)

        self.net_wrapper = net_wrapper(self.net)

    def build_net(self):
        """
        Build the network
        """
        network = get(self.params, "network", "Resnet")
        try:
            return getattr(Source.Networks, network)(self.params).to(self.device)
        except AttributeError:
            raise NotImplementedError(f"build_model: Network class {network} not recognised")

    def batch_loss(self, x):
        t0 = time.time()
        t = torch.Tensor([1, 0]).float().to(x.device)
        z_t, logp_diff_t = odeint(
            self.net_wrapper,
            (x.float(), torch.zeros(len(x), 1).float().to(self.device)),
            t)
        z_0, logp_diff_0 = z_t[-1], logp_diff_t[-1]
        logp_x = -z_0.pow(2)/2 - logp_diff_0
        loss = -logp_x.mean()
        print(loss.item(), time.time()-t0)
        return loss

    def sample_n(self, n_samples, prior_samples=None, con_depth=0):
        self.eval()
        batch_size = get(self.params, "batch_size_sample", 10000)
        x_T = self.latent.sample((n_samples, self.dim)).to(self.device)
        t = torch.Tensor([0, 1]).float().to(x_T.device)
        events = []
        #with torch.no_grad():
        for i in range(int(n_samples / batch_size)):
            x_1 = x_T[batch_size * i: batch_size * (i + 1)]
            z_t, logp_diff_t = odeint(
                self.net_wrapper,
                (x_1.float(), torch.zeros(self.batch_size, 1).float().to(self.device)),
                t)
            events.append(z_t[-1].cpu().detach().numpy())
        return np.concatenate(events, axis=0)

def trace_df_dz(f, z):
    sum_diag = 0.
    for i in range(z.shape[1]):
        sum_diag += torch.autograd.grad(f[:, i].sum(), z, create_graph=True)[0][:, i]
    return sum_diag

class net_wrapper(torch.nn.Module):

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, t, states):
        with torch.set_grad_enabled(True):
            z = states[0]
            z.requires_grad_(True)
            log_pz = states[1]
            t = t.repeat(len(z))[:, None]
            v = self.net(z, t)
            dlogp_dt = -trace_df_dz(v, z).view(-1, 1)
        return (v, dlogp_dt)