import numpy as np
import torch
from scipy.integrate import solve_ivp
import Source.Networks
from Source.Util.util import get
from Source.Models.ModelBase import GenerativeModel
from torchdiffeq import odeint
from torch.autograd import grad


class TBD(GenerativeModel):
    """
     Class for Trajectory Based Diffusion
     Inheriting from the GenerativeModel BaseClass
    """

    def __init__(self, params):

        self.loss_type = get(params, "loss_type", "l2")
        assert self.loss_type in ["l1", "l2", "mle", "weighted"], "Unknown loss type"
        if self.loss_type == "mle":
            print("Using MLE loss. Setting out_dim to 2*dim")
            params["out_dim"] = 2*params["dim"]

        super().__init__(params)
        trajectory = get(self.params, "trajectory", "sine_cosine_trajectory")
        try:
            self.trajectory = getattr(Source.Models.tbd, trajectory)
        except AttributeError:
            raise NotImplementedError(f"build_model: Trajectory type {trajectory} not implemented")

        self.C = get(self.params, "C", 1)
        if self.C != 1:
            print(f"C is {self.C}")

        self.beta_dist = get(self.params, "beta_dist", False)
        if self.beta_dist:
            self.beta_dist_a = get(self.params, "beta_dist_a", 1)
            self.beta_dist_b = get(self.params, "beta_dist_b", 0.7)
            self.dist = torch.distributions.beta.Beta(concentration1=float(self.beta_dist_a),
                                                      concentration0=float(self.beta_dist_b))
            print(f"Using beta distribution to sample t with params {self.beta_dist_a, self.beta_dist_b}")
        else:
            self.dist = torch.distributions.uniform.Uniform(low=0, high=1)
            print(f"Using uniform distribution to sample t")

        self.bayesian = get(self.params, "bayesian", 0)

        self.t_weighting = get(self.params, "t_weighting", False)
        self.t_factor = get(self.params, "t_factor", 0)
        if self.t_factor !=0:
            print(f"t_factor is {self.t_factor}")

        self.magic_transformation = get(self.params, "magic_transformation", False)
        if self.magic_transformation:
            self.unweighted_loss = []
            self.weightsum = []

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

        self.add_noise = get(self.params, "add_noise", 0)
        if self.add_noise !=0:
            print(f"adding noise of scale {self.add_noise}")

        self.rtol = get(self.params, "rtol", 1.e-3)
        self.atol = get(self.params, "atol", 1.e-6)

        self.hutch = get(self.params, "hutch", False)

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
        """
        Calculate batch loss as described by Peter
        """

        if self.magic_transformation:
            weights = x[:, -1]
            x = x[:, :-1]

        if self.conditional and self.n_jets == 1:
            condition = x[:, :3]
            x = x[:, 3:]

        elif self.conditional and self.n_jets == 2:
            condition_1 = x[:, 2:11]
            #condition_1 = prior_model(condition_1)
            condition_2 = x[:, :2]
            condition = torch.cat([condition_1, condition_2], 1)
            x = x[:, 11:]

        elif self.conditional and self.n_jets == 3:
            condition = x[:, :13]
            #condition = prior_model(condition)
            x = x[:, 13:]

        else:
            condition = None

        t = self.dist.sample((x.size(0),1)).to(x.device)
        x_0 = self.latent.sample(x.size()).to(x.device)
        if self.add_noise != 0:
            x += torch.randn_like(x)*self.add_noise
        x_t, x_t_dot = self.trajectory(x_0, x, t)

        self.net.kl = 0
        drift = self.net(x_t, t, condition)

        if self.magic_transformation:
            loss = torch.mean((drift - x_t_dot) ** 2 * weights[:, None])/weights.mean()
            #unweighted_loss = torch.mean((drift - x_t_dot)**2)
            self.regular_loss.append(loss.detach().cpu().numpy())
            #self.unweighted_loss.append(unweighted_loss.detach().cpu().numpy())
            #self.weightsum.append(weights.sum())
            #print(loss.mean(), weights.mean())
            if self.C != 0:
                kl_loss = self.C * self.net.kl / self.n_traindata
                self.kl_loss.append(kl_loss.detach().cpu().numpy())
                loss = loss + kl_loss
        else:
            if self.loss_type=="l2":
                loss = torch.mean((drift - x_t_dot) ** 2 * torch.exp(self.t_factor * t))
                self.regular_loss.append(loss.detach().cpu().numpy())
                if self.C != 0:
                    kl_loss = self.C*self.net.kl / self.n_traindata
                    self.kl_loss.append(kl_loss.detach().cpu().numpy())
                    loss = loss + kl_loss

            elif self.loss_type == "weighted":
                loss = torch.mean((drift - x_t_dot) ** 2 * weighting(1-t))
                self.regular_loss.append(loss.detach().cpu().numpy())
                if self.C != 0:
                    kl_loss = self.C * self.net.kl / self.n_traindata
                    self.kl_loss.append(kl_loss.detach().cpu().numpy())
                    loss = loss + kl_loss

            elif self.loss_type == "mle":
                mu, logsigma = drift.chunk(2, dim=1)

                mse = (mu - x_t_dot) ** 2
                mse = mse/(2*torch.exp(logsigma)**2)
                loss = torch.mean(mse+logsigma)
                self.regular_loss.append(loss.detach().cpu().numpy())
                if self.C != 0:
                    kl_loss = self.C*self.net.kl / self.n_traindata
                    self.kl_loss.append(kl_loss.detach().cpu().numpy())
                    loss = loss + kl_loss

            elif self.loss_type=="l1":
                loss = torch.mean(torch.abs(drift-x_t_dot)) + self.C*self.net.kl / self.n_traindata
        return loss

    def sample_n_scipy(self, n_samples, prior_samples=None, con_depth=0):
        """
        Generate n_samples new samples.
        Start from Gaussian random noise and solve the reverse ODE to obtain samples
        """
        if self.net.bayesian:
            self.net.map = get(self.params,"fix_mu", False)
            for bay_layer in self.net.bayesian_layers:
                bay_layer.random = None
        self.eval()
        batch_size = get(self.params, "batch_size_sample", 8192)
        x_T = self.latent.sample((n_samples + batch_size, self.dim))

        if self.conditional:
            if self.n_jets == 1 and con_depth == 0:
                #n_c = (n_samples + batch_size) // 3
                #n_r = (n_samples + batch_size) - 2 * n_c
                n_c = (n_samples) // 3
                n_r = n_c + 1

                c_1 = np.array([[1, 0, 0]] * n_c)
                c_2 = np.array([[0, 1, 0]] * n_c)
                c_3 = np.array([[0, 0, 1]] * n_r)

                condition = np.concatenate([c_1, c_2, c_3])
                print(len(condition))

            elif self.n_jets == 1 and con_depth == 1:
                n_c = (n_samples + batch_size) // 2
                n_r = (n_samples + batch_size) - n_c

                c_1 = np.array([[0, 1, 0]] * n_c)
                c_2 = np.array([[0, 0, 1]] * n_r)

                condition = np.concatenate([c_1, c_2])

            elif self.n_jets == 1 and con_depth == 2:
                n_c = n_samples + batch_size

                condition = np.array([[0, 0, 1]] * n_c)

            elif self.n_jets == 2:

                condition_1 = prior_samples[:,3:12]
                condition_2 = prior_samples[:, 1:3]

                condition = np.concatenate([condition_1, condition_2], axis=1)
                n_samples = len(condition)
            elif self.n_jets == 3:
                condition = prior_samples[:,:13]
                n_samples = len(condition)
        else:
            condition = None

        def f(t, x_t, c=None):
            x_t_torch = torch.Tensor(x_t).reshape((batch_size, self.dim)).to(self.device)
            t_torch = t * torch.ones_like(x_t_torch[:, [0]])

            with torch.no_grad():
                if c is not None:
                    c_torch = torch.Tensor(c).reshape((batch_size, self.n_con)).to(self.device)
                    f_t = self.net(x_t_torch, t_torch, c_torch)
                else:
                    f_t = self.net(x_t_torch, t_torch)

                if self.loss_type == "mle":
                    f_t, _ = f_t.chunk(2, dim=1)
            return f_t.detach().cpu().numpy().flatten()

        events = []
        function_calls = []
        with torch.no_grad():
            for i in range(int(n_samples / batch_size)):
                if self.conditional:

                    c = condition[batch_size * i: batch_size * (i + 1)].flatten()

                else:
                    c = None
                sol = solve_ivp(f,
                                (0, 1),
                                x_T[batch_size * i: batch_size * (i + 1)].flatten(),
                                args=[c],
                                rtol=self.rtol,
                                atol=self.atol)

                if self.conditional:
                    c = condition[batch_size * i: batch_size * (i + 1)]
                    if self.n_jets == 1:
                        s = np.concatenate([c, sol.y[:, -1].reshape(batch_size, self.dim)], axis=1)
                    elif self.n_jets == 2:
                        s = np.concatenate([c[:,-2:], sol.y[:, -1].reshape(batch_size, self.dim)], axis=1)
                    elif self.n_jets == 3:
                        s = sol.y[:, -1].reshape(batch_size, self.dim)
                else:
                    s = sol.y[:, -1].reshape(batch_size, self.dim)
                events.append(s)
                function_calls.append(sol.nfev)

        function_calls = np.array(function_calls)
        self.params["function_calls_mean"] = float(function_calls.mean())
        self.params["function_calls_std"] = float(function_calls.std())
        self.params["function_calls_max"] = float(function_calls.max())
        return np.concatenate(events, axis=0)[:n_samples]


    def sample_n(self, n_samples, prior_samples=None, con_depth=0):
        """
        Generate n_samples new samples.
        Start from Gaussian random noise and solve the reverse ODE to obtain samples
        """
        if self.net.bayesian:
            self.net.map = get(self.params,"fix_mu", False)
            for bay_layer in self.net.bayesian_layers:
                bay_layer.random = None
        self.eval()
        batch_size = get(self.params, "batch_size_sample", 8192)
        x_T = self.latent.sample((n_samples + batch_size, self.dim)).to(self.device)

        if self.conditional:
            if self.n_jets == 1 and con_depth == 0:
                #n_c = (n_samples + batch_size) // 3
                #n_r = (n_samples + batch_size) - 2 * n_c
                n_c = (n_samples) // 3
                n_r = n_c + 1

                c_1 = np.array([[1, 0, 0]] * n_c)
                c_2 = np.array([[0, 1, 0]] * n_c)
                c_3 = np.array([[0, 0, 1]] * n_r)

                condition = np.concatenate([c_1, c_2, c_3])

            elif self.n_jets == 1 and con_depth == 1:
                n_c = (n_samples + batch_size) // 2
                n_r = (n_samples + batch_size) - n_c

                c_1 = np.array([[0, 1, 0]] * n_c)
                c_2 = np.array([[0, 0, 1]] * n_r)

                condition = np.concatenate([c_1, c_2])

            elif self.n_jets == 1 and con_depth == 2:
                n_c = n_samples + batch_size

                condition = np.array([[0, 0, 1]] * n_c)

            elif self.n_jets == 2:

                condition_1 = prior_samples[:,3:12]
                condition_2 = prior_samples[:, 1:3]

                condition = np.concatenate([condition_1, condition_2], axis=1)
                n_samples = len(condition)
            elif self.n_jets == 3:
                condition = prior_samples[:,:13]
                n_samples = len(condition)
        else:
            condition = None


        def net_wrapper(t, x_t):
            t_torch = t * torch.ones_like(x_t[:, [0]], dtype=x_t.dtype, device=x_t.device)
            if self.conditional:
                v = self.net(x_t, t_torch, c)
            else:
                v = self.net(x_t, t_torch)
            return v

        events = []
        with torch.no_grad():
            for i in range(int(n_samples / batch_size)):
                if self.conditional:
                    c = condition[batch_size * i: batch_size * (i + 1)]
                    c = torch.tensor(c, dtype=x_T.dtype, device=x_T.device)
                else:
                    c = None

                x_t = odeint(
                    net_wrapper,
                    x_T[batch_size * i: batch_size * (i + 1)],
                    torch.tensor([0, 1], dtype=x_T.dtype, device=x_T.device),
                    atol=self.atol,
                    rtol=self.rtol,
                    method='dopri5',
                ).detach().cpu().numpy()

                if self.conditional:
                    c = condition[batch_size * i: batch_size * (i + 1)]
                    if self.n_jets == 1:
                        s = np.concatenate([c, x_t[-1]], axis=1)
                    elif self.n_jets == 2:
                        s = np.concatenate([c[:,-2:], x_t[-1]], axis=1)
                    elif self.n_jets == 3:
                        s = x_t[ -1]
                else:
                    s = x_t[ -1]
                events.append(s)
        return np.concatenate(events, axis=0)[:n_samples]




    def invert_n(self, samples, evolution = False, timesteps = 1000):
        """
        Generate n_samples new samples.
        Start from Gaussian random noise and solve the reverse ODE to obtain samples
        """
        if self.net.bayesian:
            self.net.map = get(self.params, "fix_mu", False)
            for bay_layer in self.net.bayesian_layers:
                bay_layer.random = None
        self.eval()
        batch_size = get(self.params, "batch_size", 8192)
        n_samples = samples.shape[0]

        if evolution:
            t_frames = np.linspace(1, 0, timesteps)

        def f(t, x_t):
            x_t_torch = torch.Tensor(x_t).reshape((-1, self.dim)).to(self.device)
            t_torch = t * torch.ones_like(x_t_torch[:, [0]])
            with torch.no_grad():
                f_t = self.net(x_t_torch, t_torch).detach().cpu().numpy().flatten()
            return f_t

        events = []
        with torch.no_grad():
            for i in range(int(n_samples / batch_size)):
                if evolution:
                    sol = solve_ivp(f, (1, 0), samples[batch_size * i: batch_size * (i + 1)].flatten(), t_eval=t_frames)
                    s = sol.y.reshape(-1, self.dim, timesteps)
                else:
                    sol = solve_ivp(f, (1, 0), samples[batch_size * i: batch_size * (i + 1)].flatten())
                    s = sol.y[:, -1].reshape(-1, self.dim)
                events.append(s)
            if evolution:
                sol = solve_ivp(f, (1, 0), samples[batch_size * int(n_samples / batch_size):].flatten(), t_eval=t_frames)
                s = sol.y.reshape(-1, self.dim, timesteps)
            else:
                sol = solve_ivp(f, (1, 0), samples[batch_size * int(n_samples / batch_size):].flatten())
                s = sol.y[:, -1].reshape(-1, self.dim)
            events.append(s)
        return np.concatenate(events, axis=0)[:n_samples]


    def sample_n_evolution(self, n_samples):

        n_frames = get(self.params, "n_frames", 1000)
        t_frames = np.linspace(0, 1, n_frames)

        batch_size = get(self.params, "batch_size", 8192)
        x_T = self.latent.sample((n_samples + batch_size, self.dim))

        def f(t, x_t):
            x_t_torch = torch.Tensor(x_t).reshape((batch_size, self.dim)).to(self.device)
            t_torch = t * torch.ones_like(x_t_torch[:, [0]])
            with torch.no_grad():
                f_t = self.net(x_t_torch, t_torch).detach().cpu().numpy().flatten()
            return f_t

        events = []
        with torch.no_grad():
            for i in range(int(n_samples / batch_size) + 1):
                sol = solve_ivp(f, (0, 1), x_T[batch_size * i: batch_size * (i + 1)].flatten(), t_eval=t_frames)
                s = sol.y.reshape(batch_size, self.dim, -1)
                events.append(s)
        return np.concatenate(events, axis=0)[:n_samples]


    def calculate_likelihood_old(self, x, timesteps = 1000):

        self.eval()
        batch_size = get(self.params, "batch_size_sample", 8192)
        n_samples = x.shape[0]
        times = torch.linspace(1, 0, timesteps).to(self.device)
        delta_t = (1. / timesteps)

        likelihoods = []
        for i in range(int(n_samples / batch_size)):
            inverse_trajectory = torch.from_numpy(self.invert_n(x[batch_size * i: batch_size * (i + 1)],
                                                                evolution=True,
                                                                timesteps=timesteps))
            log_jac = 0
            for i in range(timesteps):
                divergence = compute_div(self.net,
                                         inverse_trajectory[:, :, -i].float().to(self.device),
                                         times[-i].repeat(inverse_trajectory.shape[0])[:, None].float()).detach()
                log_jac = log_jac - delta_t * divergence

            jac = torch.exp(log_jac).cpu()
            inverse_x = inverse_trajectory[:, :, -1].cpu()
            likelihoods.append(gaussian_density_2d(inverse_x) * jac)
        inverse_trajectory = torch.from_numpy(self.invert_n(x[batch_size * int(n_samples / batch_size):],
                                                            evolution=True,
                                                            timesteps=timesteps))
        log_jac = 0
        for i in range(timesteps):
            divergence = compute_div(self.net,
                                     inverse_trajectory[:, :, -i].float().to(self.device),
                                     times[-i].repeat(inverse_trajectory.shape[0])[:, None].float()).detach()
            log_jac = log_jac - delta_t * divergence

        jac = torch.exp(log_jac).cpu()
        inverse_x = inverse_trajectory[:, :, -1].cpu()
        likelihoods.append(gaussian_density_2d(inverse_x) * jac)

        return torch.cat(likelihoods, dim=0).detach().numpy()

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
        return latent_log_prob.squeeze() + jac.squeeze()



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
