import numpy as np
import torch
import Source.Networks
from Source.Util.util import get
from Source.Models.ModelBase import GenerativeModel
from torchdiffeq import odeint
from torch.autograd import grad
import time 
import math


class CFM(GenerativeModel):
    """
     Class for Trajectory Based Diffusion
     Inheriting from the GenerativeModel BaseClass
    """

    def __init__(self, params):

        super().__init__(params)
        trajectory = get(self.params, "trajectory", "sine_cosine_trajectory")
        try:
            self.trajectory = getattr(Source.Models.cfm, trajectory)
        except AttributeError:
            raise NotImplementedError(f"build_model: Trajectory type {trajectory} not implemented")

        self.C = get(self.params, "C", 1)
        if self.C != 1:
            print(f"build_model: C is {self.C}")

        self.beta_dist = get(self.params, "beta_dist", False)
        if self.beta_dist:
            self.beta_dist_a = get(self.params, "beta_dist_a", 1)
            self.beta_dist_b = get(self.params, "beta_dist_b", 0.7)
            self.dist = torch.distributions.beta.Beta(concentration1=float(self.beta_dist_a),
                                                      concentration0=float(self.beta_dist_b))
            print(f"build_model: Using beta distribution to sample t with params {self.beta_dist_a, self.beta_dist_b}")
        else:
            self.dist = torch.distributions.uniform.Uniform(low=0, high=1)
            print(f"build_model: Using uniform distribution to sample t")

        self.bayesian = get(self.params, "bayesian", 0)

        self.magic_transformation = get(self.params, "magic_transformation", False)
        if self.magic_transformation:
            self.unweighted_loss = []
            self.weightsum = []

        self.latent = get(self.params, "latent", "normal")
        if self.latent == "normal":
            print("build_model: Using latent normal")
            self.latent = torch.distributions.normal.Normal(0, 1)
        elif self.latent == "uniform":
            print("build_model: Using latent uniform")
            self.latent = torch.distributions.uniform.Uniform(low=0, high=1)

        else:
            print("build_model: Latent not recognised. Using normal")
            self.latent = torch.distributions.normal.Normal(0, 1)

        self.rtol = get(self.params, "rtol", 1.e-3)
        self.atol = get(self.params, "atol", 1.e-6)

        self.hutch = get(self.params, "hutch", False)

    def build_net(self):
        """
        Build the network
        """
        network = get(self.params, "network", "MLP")
        try:
            return getattr(Source.Networks, network)(self.params).to(self.device)
        except AttributeError:
            raise NotImplementedError(f"build_model: Network class {network} not recognised")

    def batch_loss(self, x):
        """
        Calculate batch loss as described by Peter
        """

        if self.magic_transformation:
            weights = x[:, -1]
            x = x[:, :-1]

        condition = None

        t = self.dist.sample((x.size(0),1)).to(x.device)
        x_0 = self.latent.sample(x.size()).to(x.device)

        x_t, x_t_dot = self.trajectory(x_0, x, t)

        self.net.kl = 0
        drift = self.net(x_t, t, condition)

        if self.magic_transformation:
            loss = torch.mean((drift - x_t_dot) ** 2 * weights[:, None])/weights.mean()
            self.regular_loss.append(loss.detach().cpu().numpy())
            if self.bayesian:
                kl_loss = self.C * self.net.kl / self.n_traindata
                self.kl_loss.append(kl_loss.detach().cpu().numpy())
                loss = loss + kl_loss
        else:
            loss = torch.mean((drift - x_t_dot) ** 2)
            self.regular_loss.append(loss.detach().cpu().numpy())
            if self.bayesian:
                kl_loss = self.C * self.net.kl / self.n_traindata
                self.kl_loss.append(kl_loss.detach().cpu().numpy())
                loss = loss + kl_loss
        return loss

    def sample_n(self, n_samples):
        """
        Generate n_samples new samples.
        Start from Gaussian random noise and solve the reverse ODE to obtain samples
        """
        if self.net.bayesian:
            self.net.map = get(self.params,"fix_mu", False)
            for bay_layer in self.net.bayesian_layers:
                bay_layer.random = None
        self.eval1 = 0
        self.net.eval()
        t1 = time.time()
        batch_size = get(self.params, "batch_size_sample", 8192)
        x_T = self.latent.sample((n_samples, self.dim_x)).to(self.device)

        def net_wrapper(t, x_t):
            t_torch = t * torch.ones_like(x_t[:, [0]], dtype=x_t.dtype, device=x_t.device)
            if self.conditional:
                v = self.net(x_t, t_torch, c)
            else:
                v = self.net(x_t, t_torch)

            self.eval1 = self.eval1 + 1 
            # 160 
            return v
    
        events = []
        calls = []
        batches = torch.split(x_T, batch_size)
        with torch.no_grad():
            for batch in batches:
                self.eval1 = 0
                c = None
                ode_solution = odeint(
                    net_wrapper,
                    batch,
                    torch.tensor([0, 1], dtype=x_T.dtype, device=x_T.device),
                    atol=self.atol,
                    rtol=self.rtol,
                    method='dopri5',
                ).detach().cpu().numpy()
                events.append(ode_solution[-1])
                calls.append(self.eval1)
        t2 = time.time()

        print(f"generate_samples: Finished generation of {n_samples} samples with {np.mean(calls)} steps after {(t2-t1):.02}s  ")
        return np.concatenate(events, axis=0)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the log probability

        Args:
            x: input tensor, shape (n_events, dims_in)
            c: condition tensor, shape (n_events, dims_c)
        Returns:
            log probabilities, shape (n_events, )
        """
        batch_size = x.size(0)
        dtype = x.dtype
        device = x.device

        def net_wrapper(t, state):
            with torch.set_grad_enabled(True):
                x_t = state[0].requires_grad_(True)
                t_torch = t * torch.ones_like(x_t[:, [0]], dtype=dtype, device=device)
                v = self.net(t_torch, x_t)
                if self.hutch:
                    dlogp_dt = -hutch_trace(v, x_t).view(-1, 1)
                else:
                    dlogp_dt = -autograd_trace(v, x_t).view(-1, 1)
            return v.detach(), dlogp_dt.detach()

        # Sample from the latent distribution
        logp_diff_1 = torch.zeros((batch_size, 1), dtype=dtype, device=device)
        states = (x, logp_diff_1)
        # Solve the ODE from t=1 to t=0 from the sampled initial condition
        x_t, logp_diff_t = odeint(
            net_wrapper,
            states,
            torch.tensor([0, 1], dtype=dtype, device=device),
            atol=self.atol,
            rtol=self.rtol,
            method='dopri5',
            )
        x_0 = x_t[-1].detach()
        jac = logp_diff_t[-1].detach()
        latent_log_prob  = -(x_0**2 / 2 + 0.5 * np.log(2 * np.pi)).sum(dim=1)
        return latent_log_prob.squeeze() - jac.squeeze()



def sine_cosine_trajectory(x_0, x_1, t):
    c = torch.cos(t * np.pi / 2)
    s = torch.sin(t * np.pi / 2)
    x_t = c * x_0 + s * x_1

    c_dot = -np.pi / 2 * s
    s_dot = np.pi / 2 * c
    x_t_dot = c_dot * x_0 + s_dot * x_1
    return x_t, x_t_dot

def sine2_cosine2_trajectory(x_0, x_1, t):
    c = torch.cos(t * np.pi / 2)
    s = torch.sin(t * np.pi / 2)
    x_t = c**2 * x_0 + s**2 * x_1

    c_dot = -np.pi / 2 * s
    s_dot = np.pi / 2 * c
    x_t_dot = 2 * c_dot * c * x_0 + 2 * s_dot * s * x_1
    return x_t, x_t_dot

def linear_trajectory(x_0, x_1, t):
    x_t = (1 - t) * x_0 + t * x_1
    x_t_dot = x_1 - x_0
    return x_t, x_t_dot

def vp_trajectory(x_0, x_1, t, a=19.9, b=0.1):

    e = -1./4. * a * (1-t)**2 - 1./2. * b * (1-t)
    alpha_t = torch.exp(e)
    beta_t = torch.sqrt(1-alpha_t**2)
    x_t = x_0 * alpha_t + x_1 * beta_t

    e_dot = 2 * a * (1-t) + 1./2. * b
    alpha_t_dot = e_dot * alpha_t
    beta_t_dot = -2 * alpha_t * alpha_t_dot / beta_t
    x_t_dot = x_0 * alpha_t_dot + x_1 * beta_t_dot
    return x_t, x_t_dot


def compute_div(f, x, t):
    """
    f: Callable[[Time, Sample], torch.tensor],
    x: torch.tensor,
    t: torch.tensor  # [batch x dim]
) -> torch.tensor:
    Compute the divergence of f(x,t) with respect to x, assuming that x is batched
    """
    bs = x.shape[0]
    with torch.set_grad_enabled(True):
        x.requires_grad_(True)
        t.requires_grad_(True)
        f_val = f(x, t)
        divergence = 0.0
        for i in range(x.shape[1]):
            divergence += \
                    torch.autograd.grad(
                            f_val[:, i].sum(), x, create_graph=True
                        )[0][:, i]

    return divergence.view(bs)

def gaussian_density_2d(X):
    """
    Evaluates the 2D Gaussian (normal) density function at a given array of input vectors `X`,
    with mean 0 and unit covariance matrix.
    """
    exponent = -0.5 * torch.sum(torch.square(X), axis=1)
    coefficient = 1 / (2 * torch.pi)
    return coefficient * torch.exp(exponent)

def weighting(t):
    return ((1.-t)/t).clip(min=0.1, max=10)


def autograd_trace(x_out, x_in):
    """Standard brute-force means of obtaining trace of the Jacobian, O(d) calls to autograd"""
    trJ = 0.
    for i in range(x_in.shape[1]):
        trJ += grad(x_out[:, i].sum(), x_in, allow_unused=False, create_graph=True)[0][:, i]
    return trJ


def hutch_trace2(x_out, x_in):
    """Hutchinson's trace Jacobian estimator, O(1) call to autograd"""
    noise = torch.randint_like(x_in, low=0, high=2).float() * 2 - 1.
    jvp = grad(x_out, x_in, noise, create_graph=True)[0]
    trJ = torch.einsum('bi,bi->b', jvp, noise)

    return trJ

def hutch_trace(x_out, x_in):
    """Hutchinson's trace Jacobian estimator, O(1) call to autograd"""
    noise = torch.randint_like(x_in, low=0, high=2).float() * 2 - 1.
    x_out_noise = torch.sum(x_out * noise)
    gradient = grad(x_out_noise, x_in)[0]
    return torch.sum(gradient * noise, dim=1)

