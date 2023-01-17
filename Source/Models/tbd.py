import numpy as np
import torch
from scipy.integrate import solve_ivp
import Source.Networks
from Source.Util.util import get
from Source.Models.ModelBase import GenerativeModel


class TBD(GenerativeModel):
    """
     Class for Trajectory Based Diffusion
     Inheriting from the GenerativeModel BaseClass
    """

    def __init__(self, params):
        super().__init__(params)
        trajectory = get(self.params, "trajectory", "sine_cosine_trajectory")
        try:
            self.trajectory = getattr(Source.Models.tbd, trajectory)
        except AttributeError:
            raise NotImplementedError(f"build_model: Trajectory type {trajectory} not implemented")

    def build_net(self):
        """
        Build the network
        """
        network = get(self.params, "network", "Resnet")
        try:
            return getattr(Source.Networks, network)(self.params).to(self.device)
        except AttributeError:
            raise NotImplementedError(f"build_model: Network class {network} not recognised")

    def batch_loss(self, x, conditional=False):
        """
        Calculate batch loss as described by Peter
        TODO Write section in dropbox
        """

        if conditional and self.n_jets == 1:
            condition = x[:, -3:]
            x = x[:, :-3]

        elif conditional and self.n_jets == 2:
            condition_1 = x[:, :9]
            #condition_1 = prior_model(condition_1)
            condition_2 = x[:, -2:]
            condition = torch.cat([condition_1, condition_2], 1)
            x = x[:, 9:-2]

        elif conditional and self.n_jets == 3:
            condition = x[:, :13]
            #condition = prior_model(condition)
            x = x[:, 13:]


        else:
            condition = None

        t = torch.rand(x.size(0), 1, device=x.device)
        x_1 = torch.randn_like(x)

        x_t, x_t_dot = self.trajectory(x, x_1, t)

        drift = self.net(x_t, t, condition)

        loss = 0.5 * torch.mean((drift - x_t_dot) ** 2)
        return loss

    def sample_n(self, n_samples, conditional=False, prior_samples=None, con_depth=0):
        """
        Generate n_samples new samples.
        Start from Gaussian random noise and solve the reverse ODE to obtain samples
        """
        self.eval()
        batch_size = get(self.params, "batch_size", 8192)
        x_T = np.random.randn(n_samples + batch_size, self.dim)

        if conditional:
            if self.n_jets == 1 and con_depth == 0:
                n_c = (n_samples + batch_size) // 3
                n_r = (n_samples + batch_size) - 2 * n_c

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

                condition_1 = prior_samples[:,:9]
                condition_2 = prior_samples[:, -2:]

                condition = np.concatenate([condition_1, condition_2], axis=1)

            elif self.n_jets == 3:
                condition = prior_samples[:,:13]

        else:
            condition = None

        def f(t, x_t, c=None):
            x_t_torch = torch.Tensor(x_t).reshape((batch_size, self.dim)).to(self.device)
            t_torch = t * torch.ones_like(x_t_torch[:, [0]])

            with torch.no_grad():
                if c is not None:
                    c_torch = torch.Tensor(c).reshape((batch_size, self.n_con)).to(self.device)
                    f_t = self.net(x_t_torch, t_torch, c_torch).detach().cpu().numpy().flatten()
                else:
                    f_t = self.net(x_t_torch, t_torch).detach().cpu().numpy().flatten()
            return f_t

        events = []
        with torch.no_grad():
            for i in range(int(n_samples / batch_size) + 1):
                if self.conditional:
                    c = condition[batch_size * i: batch_size * (i + 1)].flatten()
                else:
                    c = None
                sol = solve_ivp(f, (1, 0), x_T[batch_size * i: batch_size * (i + 1)].flatten(), args=[c])

                if self.conditional:
                    c = condition[batch_size * i: batch_size * (i + 1)]
                    if self.n_jets == 1:
                        s = np.concatenate([sol.y[:, -1].reshape(batch_size, self.dim), c], axis=1)
                    elif self.n_jets == 2:
                        s = np.concatenate([sol.y[:, -1].reshape(batch_size, self.dim), c[:,-2:]], axis=1)
                    elif self.n_jets == 3:
                        s = sol.y[:, -1].reshape(batch_size, self.dim)
                else:
                    s = sol.y[:, -1].reshape(batch_size, self.dim)

                events.append(s)
        return np.concatenate(events, axis=0)[:n_samples]

def sine_cosine_trajectory(x, x_1, t):
    c = torch.cos(t * np.pi / 2)
    s = torch.sin(t * np.pi / 2)
    x_t = c * x + s * x_1

    c_dot = -np.pi / 2 * s
    s_dot = np.pi / 2 * c
    x_t_dot = c_dot * x + s_dot * x_1
    return x_t, x_t_dot

def linear_trajectory(x, x_1, t):
    x_t = (1 - t) * x + t * x_1
    x_t_dot = x_1 - x
    return x_t, x_t_dot
