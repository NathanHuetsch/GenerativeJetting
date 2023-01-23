import torch
import math
from Source.Util.util import get
from Source.Networks.inn_net import build_INN
from Source.Models.ModelBase import GenerativeModel


class INN(GenerativeModel):

    def __init__(self, params):
        super().__init__(params)

    def build_net(self):
        return build_INN(self.params).to(self.device)

    def batch_loss(self, x, conditional=False):
        if conditional and self.n_jets == 1:
            condition = x[:, -3:]
            x = x[:, :-3]

        elif conditional and self.n_jets == 2:
            condition_1 = x[:, :9]
            condition_2 = x[:, -2:]
            condition = torch.cat([condition_1, condition_2], 1)
            x = x[:, 9:-2]

        elif conditional and self.n_jets == 3:
            condition = x[:, :13]
            x = x[:, 13:]

        else:
            condition = None

        z, jac = self.net(x)
        loss = torch.mean(z ** 2) / 2 - torch.mean(jac) / z.shape[1]
        return loss

    def sample_n(self, n_samples, conditional=False, prior_samples=None, con_depth=0):
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
