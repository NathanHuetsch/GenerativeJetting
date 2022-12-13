import numpy as np
import torch
from scipy.integrate import solve_ivp
from Source.Networks.resnet import Resnet
from Source.Util.util import get
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
        return Resnet(self.params).to(self.device)

    def batch_loss(self, x, conditional=False):
        """
        Calculate batch loss as described by Peter
        TODO Write section in dropbox
        """

        if conditional:
            condition = x[:, -3:]
            x = x[:, :-3]
        else:
            condition = None

        t = torch.rand(x.size(0), 1, device=x.device)
        c = torch.cos(t * np.pi / 2)
        s = torch.sin(t * np.pi / 2)
        c_dot = -np.pi / 2 * s
        s_dot = np.pi / 2 * c
        x_1 = torch.randn_like(x)
        x_t = c * x + s * x_1
        x_t_dot = c_dot * x + s_dot * x_1

        drift = self.net(x_t, t, condition)

        loss = 0.5 * torch.mean((drift - x_t_dot) ** 2)
        return loss

    def sample_n(self, n_samples, n_jets=None):
        """
        Generate n_samples new samples.
        Start from Gaussian random noise and solve the reverse ODE to obtain samples
        """
        self.eval()
        batch_size = get(self.params, "batch_size", 8192)
        x_T = np.random.randn(n_samples + batch_size, self.dim - self.n_con)

        if n_jets is not None:
            condition = torch.tensor(n_jets * (n_samples + batch_size))
        else:
            n_c = (n_samples + batch_size) // 3
            n_r = (n_samples + batch_size) - 2 * n_c

            c_1 = np.array([[1, 0, 0]] * n_c)
            c_2 = np.array([[0, 1, 0]] * n_c)
            c_3 = np.array([[0, 0, 1]] * n_r)

            condition = np.concatenate([c_1, c_2, c_3])



        def f(t, x_t, c=None):
            x_t_torch = torch.Tensor(x_t).reshape((batch_size, self.dim - self.n_con)).to(self.device)
            t_torch = t * torch.ones_like(x_t_torch[:, [0]])

            c_torch = torch.Tensor(c).reshape((batch_size, self.n_con)).to(self.device)

            with torch.no_grad():
                if c is not None:
                    f_t = self.net(x_t_torch, t_torch, c_torch).detach().cpu().numpy().flatten()
                else:
                    f_t = self.net(x_t_torch, t_torch).detach().cpu().numpy().flatten()
            return f_t

        events = []
        with torch.no_grad():
            for i in range(int(n_samples / batch_size) + 1):
                c = condition[batch_size * i: batch_size * (i + 1)].flatten()
                sol = solve_ivp(f, (1, 0), x_T[batch_size * i: batch_size * (i + 1)].flatten(), args=[c])
                s = np.concatenate([sol.y[:, -1].reshape(batch_size, self.dim - self.n_con), c.reshape(batch_size, self.n_con)], axis=1)
                events.append(s)
        return np.concatenate(events, axis=0)[:n_samples]
