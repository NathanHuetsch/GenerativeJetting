import numpy as np
from Source.Util.util import linear_beta_schedule, cosine_beta_schedule
import torch
import torch.nn.functional as F
from Source.Util.util import get
from Source.Models.ModelBase import GenerativeModel
from Source.Networks.resnet import Resnet


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
        return Resnet(self.params)

    def xT_from_x0_and_noise(self, x0, t, noise):
        return self.sqrt_alphas_bar[t]*x0 + self.sqrt_One_minus_alphas_bar[t]*noise

    def x0_from_xT_and_noise(self, xT, t, noise):
        return (1./self.sqrt_alphas_bar[t]) * (xT - self.sqrt_One_minus_alphas_bar[t] * noise)

    def mu_tilde_t(self, xT, t, noise):
        return (1./self.sqrt_alphas[t]) * (xT - noise*self.betas[t]/self.sqrt_One_minus_alphas_bar[t])

    def batch_loss(self, x):

        t = torch.randint(low=1, high=self.timesteps, size=(x.size(0), 1), device=self.device)
        noise = torch.randn_like(x, device=self.device)
        xT = self.xT_from_x0_and_noise(x, t, noise)
        #xT = torch.empty(size=x.size(), device=self.device)
        #for i in range(x.size(0)):
        #    xT[i] = self.xT_from_x0_and_noise(x[i], t[i], noise[i])

        model_pred = self.net(xT.float(), t.float())
        loss = F.mse_loss(model_pred, noise)
        return loss

    def sample_n_parallel(self, n_samples):
        batch_size = get(self.params, "batch_size", 8192)
        events = []
        for i in range(int(n_samples / batch_size) + 1):
            noise_i = torch.randn(self.timesteps, batch_size, self.dim).to(self.device)
            x = noise_i[0]
            for t in reversed(range(self.timesteps)):
                z = noise_i[t] if t > 0 else 0
                with torch.no_grad():
                    model_pred = self.net(x, t*torch.ones_like(x[:, [0]])).detach()
                x = self.mu_tilde_t(x, t, model_pred) + self.sigmas[t] * z
            events.append(x.cpu().numpy())
        return np.concatenate(events, axis=0)[:n_samples]

    def sample(self):
        noise = torch.randn(self.timesteps, 1, self.dim).to(self.device)
        x = noise[0]
        for t in reversed(range(self.timesteps)):
            z = noise[t] if t > 0 else 0
            model_pred = self.net(x, t*torch.ones_like(x[:, [0]])).detach()
            x = self.mu_tilde_t(x, t, model_pred) + self.sigmas[t]*z
        return x.cpu().numpy().flatten()
