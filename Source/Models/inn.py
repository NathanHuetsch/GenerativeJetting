import torch
import math
import os
from Source.Util.util import get
from Source.Networks.inn_net import build_INN
from Source.Models.ModelBase import GenerativeModel


class INN(GenerativeModel):

    def __init__(self, params):
        super().__init__(params)

    def build_net(self):
        return build_INN(self.params).to(self.device)

    def batch_loss(self, x):
        z, jac = self.net(x)
        loss = torch.mean(z ** 2) / 2 - torch.mean(jac) / z.shape[1]
        return loss

    def sample_n_parallel(self, n_samples):
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

    def save(self, epoch=""):
        """Save the model, its optimizer and the test/train split, as well as the epoch"""
        os.makedirs(self.doc.get_file("model", False), exist_ok=True)
        torch.save({"opt": self.optim.state_dict(),
                    "net": self.net.state_dict(),
                    "epoch": self.epoch}, self.doc.get_file(f"model/model{epoch}", False))

    def load(self, epoch=""):
        """Load the model, its optimizer and the test/train split, as well as the epoch"""
        name = self.doc.get_file(f"model/model{epoch}", False)
        state_dicts = torch.load(name, map_location=self.device)
        self.net.load_state_dict(state_dicts["net"])

        try:
            self.epoch = state_dicts["epoch"]
        except:
            self.epoch = 0
            print(f"Warning: Epoch number not provided in save file, setting to {self.epoch}")
        try:
            self.optim.load_state_dict(state_dicts["opt"])
        except ValueError as e:
            print(e)
        self.net.to(self.device)