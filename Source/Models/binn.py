import torch
import math
from Source.Util.util import get
from Source.Networks.binn_net import build_BINN
from Source.Models.inn import INN
from Source.Models.ModelBase import GenerativeModel


class BINN(GenerativeModel):

    def __init__(self, params):
        super().__init__(params)
        self.bayesian_layers = []


    def build_net(self):
        return build_BINN(self.params).to(self.device)

    def batch_loss(self, x):

        z, jac = self.net(x)
        net_loss = torch.mean(z ** 2) / 2 - torch.mean(jac) / z.shape[1]
        kl_loss = sum(layer.KL() for layer in self.bayesian_layers)
        return net_loss

    def sample_n(self, n_samples, jets=None, prior_samples=None, con_depth=0):
        self.eval()
        gauss_input = torch.randn((n_samples, self.dim)).to(self.device)
        events_predict = []
        batch_size = get(self.params, "batch_size", 8192)
        with torch.no_grad():
            for i in range(math.ceil(n_samples / batch_size)):
                events_batch = self.net(gauss_input[i * batch_size:(i+1) * batch_size],
                                        rev=True)[0].squeeze()
                events_predict.append(events_batch)
            events_predict = torch.cat(events_predict, dim=0).cpu().detach().numpy()
        return events_predict
