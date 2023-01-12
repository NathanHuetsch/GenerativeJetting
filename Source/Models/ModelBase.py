import numpy as np
import torch
import torch.nn as nn
import os
from torch.utils.tensorboard import SummaryWriter
from Source.Util.util import get, get_device
from Source.Util.preprocessing import undo_preprocessing
from Source.Util.plots import plot_obs, delta_r, plot_deta_dphi
from matplotlib.backends.backend_pdf import PdfPages


class GenerativeModel(nn.Module):
    """
    Base Class for Generative Models to inherit from.
    Children classes should overwrite the individual methods as needed.
    Every child class MUST overwrite the methods:

    def build_net(self): should register some NN architecture as self.net
    def batch_loss(self, x): takes a batch of samples as input and returns the loss
    def sample_n_parallel(self, n_samples): generates and returns n_samples new samples

    See tbd.py for an example of child class

    Structure:

    __init__(params)      : Read in parameters and register the important ones
    build_net()           : Create the NN and register it as self.net
                            HAS TO BE OVERWRITTEN IN CHILD CLASS
    prepare_training()    : Read in the appropriate parameters and prepare the model for training
                            Currently this is called from run_training(), so it should not be called on its own
    run_training()        : Run the actual training.
                            Necessary parameters are read in and the training is performed.
                            This calls on the methods train_one_epoch() and validate_one_epoch()
    train_one_epoch()     : Performs one epoch of model training.
                            This calls on the method batch_loss(x)
    validate_one_epoch()  : Performs one epoch of validation.
                            This calls on the method batch_loss(x)
    batch_loss(x)         : Takes one batch of samples as input and returns the loss.
                            HAS TO BE OVERWRITTEN IN CHILD CLASS
    sample_n(n_samples)   : Generates and returns n_samples new samples as a numpy array
                            HAS TO BE OVERWRITTEN IN CHILD CLASS
    sample_and_plot       : Generates n_samples and makes plots with them.
                            This is meant to be used during training if intermediate plots are wanted
    save()                : Saves the model and all relevant information. TODO
    load()                : Loads a saved model and all relevant information. TODO
    """
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.device = get(self.params, "device", get_device())
        self.dim = self.params["dim"]
        self.net = self.build_net()
        self.conditional = self.params['conditional']
        self.n_con = self.params['n_con']
        self.n_jets = self.params['n_jets']
        self.con_depth = get(self.params,'con_depth',0)
        self.batch_size = self.params["batch_size"]

    def build_net(self):
        pass

    def prepare_training(self):
        print("train_model: Preparing model training")
        self.train_losses = np.array([])
        self.train_losses_epoch = np.array([])
        self.n_trainbatches = len(self.train_loader)

        self.validate = get(self.params, "validate", True)
        if self.validate:
            self.force_checkpoints = get(self.params, "force_checkpoints", False)
            self.n_valbatches = len(self.val_loader)
            self.val_losses_epoch = np.array([])
            self.validate_every = get(self.params, "validate_every", 10)
            self.best_val_loss = get(self.params, "best_val_loss", 1e30)
            self.best_val_epoch = get(self.params, "best_val_loss", 0)
            self.no_improvements = get(self.params, "no_improvements", 0)
            print(f"train_model: validate set to True. Validating every {self.validate_every} epochs")
        else:
            print("train_model: validate set to False. No checkpoints will be created")

        self.sample_periodically = get(self.params, "sample_periodically", False)
        if self.sample_periodically:
            self.sample_every = get(self.params, "sample_every", 1)
            self.sample_every_n_samples = get(self.params, "sample_every_n_samples", 100000)
            self.data_train = undo_preprocessing(data=self.train_loader.dataset.detach().cpu().numpy(),
                                                 events_mean=self.data_mean,
                                                 events_std=self.data_std,
                                                 u=self.data_u,
                                                 s=self.data_s,
                                                 channels=self.params["channels"],
                                                 keep_all=True,
                                                 conditional=self.conditional,
                                                 n_jets=self.n_jets)
            self.data_test = undo_preprocessing(data=self.test_loader.dataset.detach().cpu().numpy(),
                                                 events_mean=self.data_mean,
                                                 events_std=self.data_std,
                                                 u=self.data_u,
                                                 s=self.data_s,
                                                 channels=self.params["channels"],
                                                 keep_all=True,
                                                 conditional=self.conditional,
                                                 n_jets=self.n_jets)
            print(f'train_model: sample_periodically set to True. Sampling {self.sample_every_n_samples} every'
                  f' {self.sample_every} epochs. This may significantly slow down training!')

        self.log = get(self.params, "log", True)
        if self.log:
            log_dir = os.path.join(self.params["out_dir"], "logs")
            self.logger = SummaryWriter(log_dir)
            print(f"train_model: Logging to log_dir {log_dir}")
        else:
            print("train_model: log set to False. No logs will be written")

    def run_training(self, prior_model=None, prior_prior_model=None):

        self.prepare_training()
        samples = []
        n_epochs = get(self.params, "n_epochs", 100)
        past_epochs = get(self.params, "total_epochs", 0)
        print(f"train_model: Model has been trained for {past_epochs} epochs before.")
        print(f"train_model: Beginning training. n_epochs set to {n_epochs}")
        for e in range(n_epochs):
            self.epoch = past_epochs + e
            self.train()
            self.train_one_epoch()

            if self.validate:
                if (self.epoch + 1) % self.validate_every == 0:
                    self.eval()
                    self.validate_one_epoch()

            if self.sample_periodically:
                if (self.epoch + 1) % self.sample_every == 0:
                    self.eval()
                    if prior_model is not None:
                        if prior_prior_model is not None:
                            prior_prior_sample = prior_prior_model.sample_n(self.sample_every_n_samples+2*self.batch_size,
                                                                            con_depth=self.con_depth)
                        else:
                            prior_prior_sample = None
                        prior_sample = prior_model.sample_n(self.sample_every_n_samples+self.batch_size,
                                                            prior_samples=prior_prior_sample, con_depth=self.con_depth)
                    else:
                        prior_sample = None

                    sample = self.sample_n(self.sample_every_n_samples, prior_samples=prior_sample,
                                                 con_depth=self.con_depth)
                    sample = np.concatenate([prior_sample,sample],axis=1)
                    samples.append(sample)

                    #self.sample_and_plot(self.sample_every_n_samples,
                     #                    x=prior_sample)

        if self.validate:
            self.params["no_improvements"] = self.no_improvements
            self.params["best_val_epoch"] = self.best_val_epoch
            self.params["best_val_loss"] = float(self.best_val_loss)
        if self.sample_periodically:
            self.samples = samples
        else:
            self.samples = None

    def train_one_epoch(self):
        train_losses = np.array([])
        for batch_id, x in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            loss = self.batch_loss(x, conditional=self.conditional)

            #loss_m = self.train_losses[-1000:].mean()
            #loss_s = self.train_losses[-1000:].std()

            if np.isfinite(loss.item()): # and (abs(loss.item() - loss_m) / loss_s < 5 or len(self.train_losses_epoch) == 0):
                loss.backward()
                self.optimizer.step()
                train_losses = np.append(train_losses, loss.item())
                if self.log:
                    self.logger.add_scalar("train_losses", train_losses[-1], self.epoch*self.n_trainbatches + batch_id)
            else:
                print(f"train_model: Unstable loss. Skipped backprop for epoch {self.epoch}, batch_id {batch_id}")

        self.train_losses_epoch = np.append(self.train_losses_epoch, train_losses.mean())
        self.train_losses = np.concatenate([self.train_losses, train_losses], axis=0)
        if self.log:
            self.logger.add_scalar("train_losses_epoch", self.train_losses_epoch[-1], self.epoch)

    def validate_one_epoch(self):
        val_losses = np.array([])
        for batch_id, x in enumerate(self.val_loader):
            with torch.no_grad():
                loss = self.batch_loss(x, conditional=self.conditional)
            val_losses = np.append(val_losses, loss.item())
            if self.log:
                self.logger.add_scalar("val_losses", val_losses[-1],
                                       int(self.n_valbatches*self.epoch/self.validate_every) + batch_id)
        self.val_losses_epoch = np.append(self.val_losses_epoch, val_losses.mean())
        if self.log:
            self.logger.add_scalar("val_losses_epoch", self.val_losses_epoch[-1], int(self.epoch/self.validate_every))
        print(f"train_model: Validated after epoch {self.epoch}. Current val_loss is {val_losses.mean()}")
        if val_losses.mean() < self.best_val_loss:
            self.no_improvements = 0
            self.best_val_epoch = self.epoch
            self.best_val_loss = val_losses.mean()
            torch.save(self.state_dict(), "models/checkpoint.pt")
        if self.force_checkpoints:
            torch.save(self.state_dict(), f"models/checkpoint_val{len(self.val_losses_epoch)}.pt")
        else:
            self.no_improvements = self.no_improvements + 1
            print(f"train_model: val_loss has not improved for {self.no_improvements} consecutive validations")

    def batch_loss(self, x, conditional=False):
        pass

    def sample_n(self, n_samples, jets=None, prior_samples=None, con_depth=0):
        pass

    def sample_and_plot(self, n_samples, jets=None, x=None):
        os.makedirs(f"plots", exist_ok=True)
        samples = self.sample_n(n_samples, jets=jets, prior_samples=x, con_depth=self.con_depth)
        samples = undo_preprocessing(data=samples,
                                     events_mean=self.data_mean,
                                     events_std=self.data_std,
                                     u=self.data_u,
                                     s=self.data_s,
                                     channels=self.params["channels"],
                                     keep_all=True,
                                     conditional=self.conditional,
                                     n_jets=self.n_jets)

        with PdfPages(f"plots/1d_hist_epoch_{self.epoch}.pdf") as out:

            plot_train = []
            plot_test = []
            plot_samples = []

            if self.conditional and self.n_jets != 3:
                for i in range(self.n_jets, 4):
                    plot_train_jets = self.data_train[self.data_train[:, -1] == i]
                    plot_train.append(plot_train_jets)

                    plot_test_jets = self.data_test[self.data_test[:, -1] == i]
                    plot_test.append(plot_test_jets)

                    plot_samples_jets = samples[samples[:, -1] == i]
                    plot_samples.append(plot_samples_jets)

            else:
                plot_train.append(self.data_train)
                plot_test.append(self.data_test)
                plot_samples.append(samples)

            for j, _ in enumerate(plot_train):
                # Loop over the plot_channels
                for i, channel in enumerate(self.params["channels"]):
                    # Get the train data, test data and generated data for the channel
                    obs_train = plot_train[j][:, channel]
                    obs_test = plot_test[j][:, channel]
                    obs_generated = plot_samples[j][:, channel]
                    # Get the name and the range of the observable
                    obs_name = self.obs_names[channel]
                    obs_range = self.obs_ranges[channel]
                    n_epochs = self.epoch
                    # Create the plot
                    plot_obs(pp=out,
                             obs_train=obs_train,
                             obs_test=obs_test,
                             obs_predict=obs_generated,
                             name=obs_name,
                             range=obs_range,
                             n_epochs=n_epochs,
                             n_jets=j + self.n_jets,
                             conditional=self.conditional)

        if all(c in self.params["channels"] for c in [9, 10, 13, 14]):
            obs_name = "\Delta R_{j_1 j_2}"
            with PdfPages(f"plots/deltaR_j1_j2_epoch_{self.epoch}.pdf") as out:
                for j, _ in enumerate(plot_train):
                    obs_train = delta_r(plot_train[j])
                    obs_test = delta_r(plot_test[j])
                    obs_generated = delta_r(plot_samples[j])
                    plot_obs(pp=out,
                             obs_train=obs_train,
                             obs_test=obs_test,
                             obs_predict=obs_generated,
                             name=obs_name,
                             n_epochs=n_epochs,
                             n_jets=j + self.n_jets,
                             conditional=self.conditional,
                             range=[0, 8])
                    if self.n_jets == 3:
                        obs_name = "\Delta R_{j_1 j_3}"
                        obs_train = delta_r(plot_train[j], idx_phi1=9, idx_eta1=10, idx_phi2=17, idx_eta2=18)
                        obs_test = delta_r(plot_test[j], idx_phi1=9, idx_eta1=10, idx_phi2=17, idx_eta2=18)
                        obs_generated = delta_r(plot_samples[j], idx_phi1=9, idx_eta1=10, idx_phi2=17,
                                                idx_eta2=18)
                        plot_obs(pp=out,
                                 obs_train=obs_train,
                                 obs_test=obs_test,
                                 obs_predict=obs_generated,
                                 name=obs_name,
                                 n_epochs=n_epochs,
                                 n_jets=j + self.n_jets,
                                 conditional=self.conditional,
                                 range=[0, 8])
                        obs_name = "\Delta R_{j_2 j_3}"
                        obs_train = delta_r(plot_train[j], idx_phi1=13, idx_eta1=14, idx_phi2=17, idx_eta2=18)
                        obs_test = delta_r(plot_test[j], idx_phi1=13, idx_eta1=14, idx_phi2=17, idx_eta2=18)
                        obs_generated = delta_r(plot_samples[j], idx_phi1=13, idx_eta1=14, idx_phi2=17,
                                                idx_eta2=18)
                        plot_obs(pp=out,
                                 obs_train=obs_train,
                                 obs_test=obs_test,
                                 obs_predict=obs_generated,
                                 name=obs_name,
                                 n_epochs=n_epochs,
                                 n_jets=j + self.n_jets,
                                 conditional=self.conditional,
                                 range=[0, 8])

            for j, _ in enumerate(plot_train):
                file_name = f"plots/deta_dphi_jets_{j + self.n_jets}_epoch_{self.epoch}.pdf"
                plot_deta_dphi(file_name=file_name,
                               data_train=plot_train[j],
                               data_test=plot_test[j],
                               data_generated=plot_samples[j],
                               n_jets=j + self.n_jets,
                               conditional=self.conditional,
                               n_epochs=n_epochs)

                if self.n_jets == 3:
                    file_name = f"plots/deta_dphi_jets_{j + self.n_jets}_13_epoch_{self.epoch}.pdf"
                    plot_deta_dphi(file_name=file_name,
                                   data_train=plot_train[j],
                                   data_test=plot_test[j],
                                   data_generated=plot_samples[j],
                                   idx_phi1=9,
                                   idx_phi2=17,
                                   idx_eta1=10,
                                   idx_eta2=18,
                                   n_jets=j + self.n_jets,
                                   conditional=self.conditional,
                                   n_epochs=n_epochs)

                    file_name = f"plots/deta_dphi_jets_{j + self.n_jets}_23_epoch_{self.epoch}.pdf"
                    plot_deta_dphi(file_name=file_name,
                                   data_train=plot_train[j],
                                   data_test=plot_test[j],
                                   data_generated=plot_samples[j],
                                   idx_phi1=13,
                                   idx_phi2=14,
                                   idx_eta1=10,
                                   idx_eta2=18,
                                   n_jets=j + self.n_jets,
                                   conditional=self.conditional,
                                   n_epochs=n_epochs)


    def save(self, epoch=""):
        # Deprecated TOD0
        os.makedirs(self.doc.get_file("model", False), exist_ok=True)
        torch.save({"opt": self.optim.state_dict(),
                    "net": self.net.state_dict(),
                    "epoch": self.epoch}, self.doc.get_file(f"model/model{epoch}", False))

    def load(self, epoch=""):
        # Deprecated TOD0
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