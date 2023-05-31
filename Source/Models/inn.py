import torch
import math
from Source.Util.util import get
from Source.Networks.inn_net import INNnet
from Source.Models.ModelBase import GenerativeModel
import numpy as np


class INN(GenerativeModel):

    def __init__(self, params):
        super().__init__(params)
        self.C = get(self.params, "C", 1)
        if self.C != 1:
            print(f"C is {self.C}")
        self.magic_transformation = get(self.params,"magic_transformation", False)

    def build_net(self):

        return INNnet(self.params).to(self.device)

    def get_condition_and_input(self, input):
        """
        :param input: model input + conditional input
        :return: model input, conditional input
        """
        x = input.clone()
        # if true use last entry of input as weights
        if self.magic_transformation:
            weights = x[:, -1]
            x = x[:, :-1]
        else:
            weights = None

        # get conditions for different networks
        if self.conditional and self.n_jets == 1:
            # get hot-encoded jet count (3d base) from first three entries of input
            condition = x[:, :3]
            x = x[:, 3:]

        elif self.conditional and self.n_jets == 2:
            # get first 9 entries of input corresponding to mu, mu, jet
            condition_1 = x[:, 2:11]
            # get hot-encoded jet count (2d base) from first two entries of input
            condition_2 = x[:, :2]
            # add all conditions
            condition = torch.cat([condition_1, condition_2], 1)
            # define model input to last four entries corresponding to second jet
            x = x[:, 11:]

        elif self.conditional and self.n_jets == 3:
            # get first 13 entries of input corresponding to mu, mu, jet, jet
            condition = x[:, :13]
            # define model input to last four entries corresponding to third jet
            x = x[:, 13:]


        else:
            condition = []

        return x, condition, weights
    def batch_loss(self, x):

        x, condition, weights = self.get_condition_and_input(x)

        z, jac = self.net(x, c=condition)


        # Get likelihood loss as defined in paper
        regular_loss = torch.mean(z ** 2) / 2 - torch.mean(jac) / z.shape[1]
        kl_loss = self.C * self.net.kl() / (len(self.data_train))


        loss = regular_loss + kl_loss


        # save loss for loss plot
        self.regular_loss.append(regular_loss.detach().cpu().numpy())
        try:
            # save bayesian loss for loss plot
            self.kl_loss.append((self.net.kl() / (len(self.data_train))).detach().cpu().numpy())
        except:
            pass

        return loss

    def get_condition_for_sample(self, n_samples,prior_samples=None, con_depth=0):
        """
        :param n_samples: number of samples
        :param batch_size: batch size of sampling process
        :param prior_samples: if not none, the samples produced with prior_model
        :param con_depth: number of models used priorly for the first dimensions
        :return: conditions as additional input for model during sampling
        """

        if self.conditional:

            if self.n_jets == 1 and con_depth == 0:
                n_c = (n_samples) // 3
                n_r = (n_samples ) - 2 * n_c

                c_1 = np.array([[1, 0, 0]] * n_c)
                c_2 = np.array([[0, 1, 0]] * n_c)
                c_3 = np.array([[0, 0, 1]] * n_r)

                condition = np.concatenate([c_1, c_2, c_3])

            elif self.n_jets == 1 and con_depth == 1:
                n_c = (n_samples ) // 2
                n_r = (n_samples) - n_c

                c_1 = np.array([[0, 1, 0]] * n_c)
                c_2 = np.array([[0, 0, 1]] * n_r)

                condition = np.concatenate([c_1, c_2])

            elif self.n_jets == 1 and con_depth == 2:
                n_c = n_samples

                condition = np.array([[0, 0, 1]] * n_c)

            elif self.n_jets == 2:

                condition_1 = prior_samples[:, 3:12]
                condition_2 = prior_samples[:, 1:3]

                condition = np.concatenate([condition_1, condition_2], axis=1)

            else:
                condition = prior_samples[:, :13]

        else:
            condition = []

        return condition

    def sample_n(self, n_samples, jets=None, prior_samples=None, con_depth=0):
        self.eval()

        # get conditions and size of unfolded data
        batch_size = get(self.params, "batch_size_sample", 8192)
        condition = self.get_condition_for_sample(n_samples=n_samples, con_depth=0)

        gauss_input = torch.randn((n_samples, self.dim)).to(self.device)

        events_predict = []

        with torch.no_grad():
            for i in range(math.ceil(n_samples / batch_size)):
                if self.conditional:
                    c = torch.from_numpy(condition[batch_size * i: batch_size * (i + 1)]).to(self.device)

                    events_batch = self.net(gauss_input[i * batch_size:(i + 1) * batch_size], c,
                                        rev=True)[0].squeeze()
                else:
                    events_batch = self.net(gauss_input[i * batch_size:(i + 1) * batch_size],
                                            rev=True)[0].squeeze()
                events_predict.append(events_batch)
            events_predict = torch.cat(events_predict, dim=0).cpu().detach().numpy()


        if self.conditional and self.n_jets == 1:
            s = np.concatenate([condition,events_predict], axis=1)
        elif self.conditional and self.n_jets == 2:
            s = np.concatenate([condition[:, -2:], events_predict], axis=1)
        else:
            s = events_predict

        return s
