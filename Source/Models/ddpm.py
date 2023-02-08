import numpy as np
from Source.Util.util import linear_beta_schedule, cosine_beta_schedule
import torch
import torch.nn.functional as F
from Source.Util.util import get
from Source.Models.ModelBase import GenerativeModel
import Source.Networks


class DDPM(GenerativeModel):

    def __init__(self, params):
        super().__init__(params)

        self.timesteps = get(self.params, "timesteps", 1000)
        self.beta_schedule = get(self.params, "beta_schedule", "linear")
        if self.beta_schedule == 'linear':
            self.register_buffer('betas', linear_beta_schedule(self.timesteps))
        elif self.beta_schedule == 'cosine':
            self.register_buffer('betas', cosine_beta_schedule(self.timesteps))
        else:
            raise ValueError(f'unknown beta schedule {self.beta_schedule}')

        # calculate and save some coefficients (Notation from Ho et. al. 2020)
        self.register_buffer('alphas', 1-self.betas)
        self.register_buffer('alphas_bar', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('alphas_bar_prev', F.pad(self.alphas_bar[:-1], (1, 0), value=1.))
        self.register_buffer('One_minus_alphas_bar', 1-self.alphas_bar)
        self.register_buffer('sqrt_alphas', torch.sqrt(self.alphas))
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(self.alphas_bar))
        self.register_buffer('sqrt_One_minus_alphas_bar', torch.sqrt(self.One_minus_alphas_bar))
        self.register_buffer('betas_tilde', self.betas * (1-self.alphas_bar_prev) / (1-self.alphas_bar))

        self.sigma_mode = get(self.params, "sigma_mode", "beta")
        if self.sigma_mode == "beta":
            self.register_buffer("sigmas", torch.sqrt(self.betas))
        elif self.sigma_mode == "beta_tilde":
            self.register_buffer("sigmas", torch.sqrt(self.betas_tilde))
        else:
            raise ValueError(f'unknown sigma mode {self.sigma_mode}')

        self.to(self.device)
    def build_net(self):
        """
        Build the network
        """
        network = get(self.params, "network", "Resnet")
        try:
            return getattr(Source.Networks, network)(self.params).to(self.device)
        except AttributeError:
            raise NotImplementedError(f"build_model: Network class {network} not recognised")

    def xT_from_x0_and_noise(self, x0, t, noise):
        return self.sqrt_alphas_bar[t]*x0 + self.sqrt_One_minus_alphas_bar[t]*noise

    def x0_from_xT_and_noise(self, xT, t, noise):
        return (1./self.sqrt_alphas_bar[t]) * (xT - self.sqrt_One_minus_alphas_bar[t] * noise)

    def mu_tilde_t(self, xT, t, noise):
        return (1./self.sqrt_alphas[t]) * (xT - noise*self.betas[t]/self.sqrt_One_minus_alphas_bar[t])

    def batch_loss(self, x):

        if self.conditional and self.n_jets == 1:
            condition = x[:, -3:]
            condition = condition.float()
            x = x[:, :-3]

        elif self.conditional and self.n_jets == 2:
            condition_1 = x[:, :9]
            #condition_1 = prior_model(condition_1)
            condition_2 = x[:, -2:]
            condition = torch.cat([condition_1, condition_2], 1)
            condition = condition.float()
            x = x[:, 9:-2]

        elif self.conditional and self.n_jets == 3:
            condition = x[:, :13]
            condition = condition.float()
            #condition = prior_model(condition)
            x = x[:, 13:]



        else:
            condition = None

        t = torch.randint(low=1, high=self.timesteps, size=(x.size(0), 1), device=self.device)
        noise = torch.randn_like(x, device=self.device)

        xT = self.xT_from_x0_and_noise(x, t, noise)
        #xT = torch.empty(size=x.size(), device=self.device)
        #for i in range(x.size(0)):
        #    xT[i] = self.xT_from_x0_and_noise(x[i], t[i], noise[i])
        self.net.kl = 0
        model_pred = self.net(xT.float(), t.float(), condition)
        loss = F.mse_loss(model_pred, noise) + self.net.kl / len(self.data_train)

        self.regular_loss.append(F.mse_loss(model_pred, noise).detach().cpu().numpy())
        try:
            self.kl_loss.append((self.net.kl / len(self.data_train)).detach().cpu().numpy())
        except:
            pass

        return loss

    def sample_n(self, n_samples, prior_samples=None, con_depth=0):
        self.eval()
        batch_size = get(self.params, "batch_size_sample", 8192)
        events = []

        if self.conditional:
            if self.n_jets == 1 and con_depth == 0:
                n_c = (n_samples + batch_size) // 3
                n_r = (n_samples + batch_size) - 2 * n_c

                c_1 = np.array([[1, 0, 0]] * n_c)
                c_2 = np.array([[0, 1, 0]] * n_c)
                c_3 = np.array([[0, 0, 1]] * n_r)

                condition = np.concatenate([c_1, c_2, c_3])
                condition = torch.Tensor(condition).to(self.device)

            elif self.n_jets == 1 and con_depth == 1:
                n_c = (n_samples + batch_size) // 2
                n_r = (n_samples + batch_size) - n_c

                c_1 = np.array([[0, 1, 0]] * n_c)
                c_2 = np.array([[0, 0, 1]] * n_r)

                condition = np.concatenate([c_1, c_2])
                condition = torch.Tensor(condition).to(self.device)

            elif self.n_jets == 1 and con_depth == 2:
                n_c = n_samples + batch_size

                condition = np.array([[0, 0, 1]] * n_c)
                condition = torch.Tensor(condition).to(self.device)

            elif self.n_jets == 2:

                condition_1 = prior_samples[:,:9]
                condition_2 = prior_samples[:, -2:]

                condition = np.concatenate([condition_1, condition_2], axis=1)
                condition = torch.Tensor(condition).to(self.device)

            elif self.n_jets == 3:
                condition = prior_samples[:,:13]
                condition = torch.Tensor(condition).to(self.device)

        else:
            condition = None

        for i in range(int(n_samples / batch_size) + 1):
            if self.conditional:
                c = condition[batch_size * i: batch_size * (i + 1)]
            else:
                c = None
            noise_i = torch.randn(self.timesteps, batch_size, self.dim).to(self.device)
            x = noise_i[0]
            for t in reversed(range(self.timesteps)):
                z = noise_i[t] if t > 0 else 0
                with torch.no_grad():
                    model_pred = self.net(x, t*torch.ones_like(x[:, [0]]), c).detach()
                x = self.mu_tilde_t(x, t, model_pred) + self.sigmas[t] * z

            if self.conditional and self.n_jets == 1:
                s = torch.concatenate([x, c], dim=1)
            elif self.conditional and self.n_jets == 2:
                s = torch.concatenate([x, c[:, -2:]], dim=1)
            else:
                s = x
            events.append(s.cpu().numpy())
        return np.concatenate(events, axis=0)[:n_samples]

    def sample(self):
        noise = torch.randn(self.timesteps, 1, self.dim).to(self.device)
        x = noise[0]
        for t in reversed(range(self.timesteps)):
            z = noise[t] if t > 0 else 0
            model_pred = self.net(x, t*torch.ones_like(x[:, [0]])).detach()
            x = self.mu_tilde_t(x, t, model_pred) + self.sigmas[t]*z
        return x.cpu().numpy().flatten()
