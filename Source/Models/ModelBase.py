import numpy as np
import torch
import torch.nn as nn
import os, time
from torch.utils.tensorboard import SummaryWriter
from Source.Util.util import get, get_device, magic_trafo, inverse_magic_trafo
from Source.Util.preprocessing import undo_preprocessing
from Source.Util.plots import plot_obs, delta_r, plot_deta_dphi, plot_obs_2d, plot_loss
from Source.Util.physics import get_M_ll
from Source.Util.simulateToyData import ToySimulator
from matplotlib.backends.backend_pdf import PdfPages

import os



class GenerativeModel(nn.Module):
    """
    Base Class for Generative Models to inherit from.
    Children classes should overwrite the individual methods as needed.
    Every child class MUST overwrite the methods:

    def build_net(self): should register some NN architecture as self.net
    def batch_loss(self, x): takes a batch of samples as input and returns the loss
    def sample_n_parallel(self, n_samples): generates and returns n_samples new samples

    See cfm.py for an example of child class

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
        self.dim_x = self.params["dim_x"]
        self.conditional = get(self.params,'conditional',False)

        self.n_jets = get(self.params,'n_jets',2)

        self.batch_size = self.params["batch_size"]
        self.batch_size_sample = get(self.params, "batch_size_sample", self.batch_size)
        self.istoy = get(self.params, "istoy", False)
        self.epoch = get(self.params, "total_epochs", 0)
        self.net = self.build_net()
        self.iterations = get(self.params,"iterations", 1)
        self.regular_loss = []
        self.kl_loss = []
        self.runs = get(self.params, "runs", 0)

    def build_net(self):
        pass

    def prepare_training(self):
        print("train_model: Preparing model training")
        self.use_scheduler = get(self.params, "use_scheduler", False)
        self.train_losses = np.array([])
        self.train_losses_epoch = np.array([])
        self.n_trainbatches = len(self.train_loader)
        self.n_traindata = len(self.data_train)

        self.sample_periodically = get(self.params, "sample_periodically", False)
        if self.sample_periodically:
            self.sample_every = get(self.params, "sample_every", 1)
            self.sample_every_n_samples = get(self.params, "sample_every_n_samples", 1000000)
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
        n_epochs = get(self.params, "n_epochs", 100)
        print_every = get(self.params, "print_every", int(n_epochs/20))
        if print_every == 0:
            print_every = 1

        print(f"train_model: Beginning training. n_epochs set to {n_epochs}")
        for e in range(n_epochs):

            self.epoch =  e
            self.train()
            t0 = time.time()
            self.train_one_epoch()
            t1 = time.time()
            if e % print_every == 0:
                print(f"train_model: Finished epoch {len(self.train_losses_epoch)}"
                      f" with average loss {np.round(self.train_losses_epoch[-1], 5)} "
                      f"in {np.round(t1 - t0, 1)} seconds")

            if self.sample_periodically:
                if (self.epoch + 1) % self.sample_every == 0:
                    sample_t0 = time.time()
                    print(f"Doing intermediate sampling at epoch {self.epoch}")
                    self.eval()
                    if not self.istoy:
                        samples = self.sample_and_undo(self.sample_every_n_samples)
                        self.plot_samples(samples=samples)
                    else:
                        samples = self.sample_n(self.sample_every_n_samples)
                        self.plot_toy(samples=samples)
                    sample_t1 = time.time()
                    print(f"Finished intermediate sampling at epoch {self.epoch} after {sample_t1 - sample_t0} seconds")
            if get(self.params,"save_periodically",False):
                if (self.epoch + 1) % get(self.params,"save_every",10) == 0 or self.epoch==0:
                    torch.save(self.state_dict(), f"models/model_epoch_{e+1}.pt")

            if e==0:
                t1 = time.time()
                dtEst= (t1-t0) * n_epochs
                print(f"Training time estimate: {dtEst/60:.2f} min = {dtEst/60**2:.2f} h")

    def train_one_epoch(self):
        self.train()
        self.net.train()
        train_losses = np.array([])
        for batch_id, x in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            loss = self.batch_loss(x)
            if np.isfinite(loss.item()):
                loss.backward()
                self.optimizer.step()
                train_losses = np.append(train_losses, loss.item())
                if self.log:
                    self.logger.add_scalar("train_losses", train_losses[-1], self.epoch*self.n_trainbatches + batch_id)

                if self.use_scheduler:
                    self.scheduler.step()
                    if self.log:
                        self.logger.add_scalar("learning_rate", self.scheduler.get_last_lr()[0],
                                               self.epoch * self.n_trainbatches + batch_id)

            else:
                print(f"train_model: Unstable loss. Skipped backprop for epoch {self.epoch}, batch_id {batch_id}")
        self.train_losses_epoch = np.append(self.train_losses_epoch, train_losses.mean())
        self.train_losses = np.concatenate([self.train_losses, train_losses], axis=0)
        if self.log:
            self.logger.add_scalar("train_losses_epoch", self.train_losses_epoch[-1], self.epoch)
            if self.use_scheduler:
                self.logger.add_scalar("learning_rate_epoch", self.scheduler.get_last_lr()[0],
                                       self.epoch)

    def batch_loss(self, x):
        pass

    def sample_n(self, n_samples):
        pass

    def sample_and_undo(self, n_samples):

        samples = self.sample_n(n_samples)
        samples = undo_preprocessing(samples, self.data_mean, self.data_std, self.params)

        return samples

    def plot_samples(self, samples, finished=False):
        os.makedirs(f"plots", exist_ok=True)
        if finished:
            path = f"plots/run{self.runs}"
            os.makedirs(path, exist_ok=True)
            iterations = self.iterations
        else:
            path = "plots"
            iterations = 1

        n_epochs = self.epoch

        plot_train = []
        plot_test = []
        plot_samples = []
        plot_weights = []
        weights = None

        plot_train.append(self.data_train)
        plot_test.append(self.data_test)
        plot_samples.append(samples)

        if get(self.params, "magic_transformation", False):
            R_minus = get(self.params, "R_minus", 0.2)
            R_plus = get(self.params, "R_plus", 1.5)
            if self.n_jets == 2:
                deltaR12 = delta_r(samples, idx_phi1=9, idx_eta1=10, idx_phi2=13, idx_eta2=14)
                weights = inverse_magic_trafo(deltaR12, R_minus=R_minus, R_plus=R_plus)
            elif self.n_jets == 3:
                deltaR12 = delta_r(samples, idx_phi1=9, idx_eta1=10, idx_phi2=13, idx_eta2=14)
                deltaR13 = delta_r(samples, idx_phi1=9, idx_eta1=10, idx_phi2=17, idx_eta2=18)
                deltaR23 = delta_r(samples, idx_phi1=13, idx_eta1=14, idx_phi2=17, idx_eta2=18)
                weights12 = inverse_magic_trafo(deltaR12, R_minus=R_minus, R_plus=R_plus)
                weights13 = inverse_magic_trafo(deltaR13, R_minus=R_minus, R_plus=R_plus)
                weights23 = inverse_magic_trafo(deltaR23, R_minus=R_minus, R_plus=R_plus)
                weights = weights12 * weights13 * weights23

        plot_weights.append(weights)

        with PdfPages(f"{path}/1d_hist_epoch_{n_epochs}.pdf") as out:
            for j, _ in enumerate(plot_train):
                # Loop over the plot_channels
                for i, channel in enumerate(self.params["plot_channels"]):
                    # Get the train data, test data and generated data for the channel
                    obs_train = plot_train[j][:, channel]
                    obs_test = plot_test[j][:, channel]
                    obs_generated = plot_samples[j][:, channel]
                    weights = plot_weights[j]
                    # Get the name and the range of the observable
                    obs_name = self.obs_names[channel]
                    obs_range = self.obs_ranges[channel]

                    # Create the plot
                    plot_obs(pp=out,
                             obs_train=obs_train,
                             obs_test=obs_test,
                             obs_predict=obs_generated,
                             name=obs_name,
                             range=obs_range,
                             n_epochs=n_epochs,
                             n_jets=j + self.n_jets,
                             weight_samples=iterations,
                             predict_weights=weights)

        if get(self.params,"plot_deltaR", True) and self.n_jets >= 2:
            with PdfPages(f"{path}/deltaR_jl_jm_epoch_{n_epochs}.pdf") as out:
                for j, _ in enumerate(plot_train):
                    obs_name = "\Delta R_{j_1 j_2}"
                    obs_train = delta_r(plot_train[j])
                    obs_test = delta_r(plot_test[j])
                    weights = plot_weights[j]
                    obs_generated = delta_r(plot_samples[j])
                    plot_obs(pp=out,
                         obs_train=obs_train,
                         obs_test=obs_test,
                         obs_predict=obs_generated,
                         name=obs_name,
                         n_epochs=n_epochs,
                         n_jets=j + self.n_jets,
                         range=[0, 8],
                         weight_samples=iterations,
                         predict_weights=weights)
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
                             range=[0, 8],
                             weight_samples=iterations,
                             predict_weights=weights)
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
                             range=[0, 8],
                             weight_samples=iterations,
                             predict_weights=weights)

        if get(self.params,"plot_Deta_Dphi", True) and self.n_jets >= 2:
            with PdfPages(f"{path}/deta_dphi_jets_epoch_{n_epochs}.pdf") as out:
                for j, _ in enumerate(plot_train):
                    plot_deta_dphi(pp=out,
                           data_train=plot_train[j],
                           data_test=plot_test[j],
                           data_generated=plot_samples[j],
                           n_jets=j + self.n_jets,
                           n_epochs=n_epochs)

                    if self.n_jets == 3:
                        plot_deta_dphi(pp=out,
                               data_train=plot_train[j],
                               data_test=plot_test[j],
                               data_generated=plot_samples[j],
                               idx_phi1=9,
                               idx_phi2=17,
                               idx_eta1=10,
                               idx_eta2=18,
                               n_jets=j + self.n_jets,
                               n_epochs=n_epochs)

                        plot_deta_dphi(pp=out,
                               data_train=plot_train[j],
                               data_test=plot_test[j],
                               data_generated=plot_samples[j],
                               idx_phi1=13,
                               idx_phi2=17,
                               idx_eta1=14,
                               idx_eta2=18,
                               n_jets=j + self.n_jets,
                               n_epochs=n_epochs)

        if get(self.params, "plot_deltaR_all", True):
            with PdfPages(f"{path}/deltaR_all_epoch_{n_epochs}.pdf") as out:
                for j, _ in enumerate(plot_train):
                    obs_name = "\Delta R_{l_1 l_2}"
                    obs_train = delta_r(plot_train[j], idx_phi1=1, idx_eta1=2, idx_phi2=5, idx_eta2=6)
                    obs_test = delta_r(plot_test[j], idx_phi1=1, idx_eta1=2, idx_phi2=5, idx_eta2=6)
                    obs_generated = delta_r(plot_samples[j], idx_phi1=1, idx_eta1=2, idx_phi2=5, idx_eta2=6)
                    weights = plot_weights[j]
                    plot_obs(pp=out,
                             obs_train=obs_train,
                             obs_test=obs_test,
                             obs_predict=obs_generated,
                             name=obs_name,
                             n_epochs=n_epochs,
                             n_jets=j + self.n_jets,
                             range=[0, 8],
                             weight_samples=iterations,
                             predict_weights=weights)
                    
                    obs_name = "\Delta R_{l_1 j_1}"
                    obs_train = delta_r(plot_train[j], idx_phi1=1, idx_eta1=2, idx_phi2=9, idx_eta2=10)
                    obs_test = delta_r(plot_test[j], idx_phi1=1, idx_eta1=2, idx_phi2=9, idx_eta2=10)
                    obs_generated = delta_r(plot_samples[j], idx_phi1=1, idx_eta1=2, idx_phi2=9, idx_eta2=10)
                    plot_obs(pp=out,
                             obs_train=obs_train,
                             obs_test=obs_test,
                             obs_predict=obs_generated,
                             name=obs_name,
                             n_epochs=n_epochs,
                             n_jets=j + self.n_jets,
                             range=[0, 8],
                             weight_samples=iterations,
                             predict_weights=weights)
                    
                    obs_name = "\Delta R_{l_2 j_1}"
                    obs_train = delta_r(plot_train[j], idx_phi1=5, idx_eta1=6, idx_phi2=9, idx_eta2=10)
                    obs_test = delta_r(plot_test[j], idx_phi1=5, idx_eta1=6, idx_phi2=9, idx_eta2=10)
                    obs_generated = delta_r(plot_samples[j], idx_phi1=5, idx_eta1=6, idx_phi2=9, idx_eta2=10)
                    plot_obs(pp=out,
                             obs_train=obs_train,
                             obs_test=obs_test,
                             obs_predict=obs_generated,
                             name=obs_name,
                             n_epochs=n_epochs,
                             n_jets=j + self.n_jets,
                             range=[0, 8],
                             weight_samples=iterations,
                             predict_weights=weights)

                    if self.n_jets >= 2:
                        obs_name = "\Delta R_{l_1 j_2}"
                        obs_train = delta_r(plot_train[j], idx_phi1=1, idx_eta1=2, idx_phi2=13, idx_eta2=14)
                        obs_test = delta_r(plot_test[j], idx_phi1=1, idx_eta1=2, idx_phi2=13, idx_eta2=14)
                        obs_generated = delta_r(plot_samples[j], idx_phi1=1, idx_eta1=2, idx_phi2=13, idx_eta2=14)
                        plot_obs(pp=out,
                                 obs_train=obs_train,
                                 obs_test=obs_test,
                                 obs_predict=obs_generated,
                                 name=obs_name,
                                 n_epochs=n_epochs,
                                 n_jets=j + self.n_jets,
                                 range=[0, 8],
                                 weight_samples=iterations,
                                 predict_weights=weights)

                        obs_name = "\Delta R_{l_2 j_2}"
                        obs_train = delta_r(plot_train[j], idx_phi1=5, idx_eta1=6, idx_phi2=13, idx_eta2=14)
                        obs_test = delta_r(plot_test[j], idx_phi1=5, idx_eta1=6, idx_phi2=13, idx_eta2=14)
                        obs_generated = delta_r(plot_samples[j], idx_phi1=5, idx_eta1=6, idx_phi2=13, idx_eta2=14)
                        plot_obs(pp=out,
                                 obs_train=obs_train,
                                 obs_test=obs_test,
                                 obs_predict=obs_generated,
                                 name=obs_name,
                                 n_epochs=n_epochs,
                                 n_jets=j + self.n_jets,
                                 range=[0, 8],
                                 weight_samples=iterations,
                                 predict_weights=weights)

                    if self.n_jets >= 3:
                        obs_name = "\Delta R_{l_1 j_3}"
                        obs_train = delta_r(plot_train[j], idx_phi1=1, idx_eta1=2, idx_phi2=17, idx_eta2=18)
                        obs_test = delta_r(plot_test[j], idx_phi1=1, idx_eta1=2, idx_phi2=17, idx_eta2=18)
                        obs_generated = delta_r(plot_samples[j], idx_phi1=1, idx_eta1=2, idx_phi2=17, idx_eta2=18)
                        plot_obs(pp=out,
                                 obs_train=obs_train,
                                 obs_test=obs_test,
                                 obs_predict=obs_generated,
                                 name=obs_name,
                                 n_epochs=n_epochs,
                                 n_jets=j + self.n_jets,
                                 range=[0, 8],
                                 weight_samples=iterations,
                                 predict_weights=weights)

                        obs_name = "\Delta R_{l_2 j_3}"
                        obs_train = delta_r(plot_train[j], idx_phi1=5, idx_eta1=6, idx_phi2=17, idx_eta2=18)
                        obs_test = delta_r(plot_test[j], idx_phi1=5, idx_eta1=6, idx_phi2=17, idx_eta2=18)
                        obs_generated = delta_r(plot_samples[j], idx_phi1=5, idx_eta1=6, idx_phi2=17, idx_eta2=18)
                        plot_obs(pp=out,
                                 obs_train=obs_train,
                                 obs_test=obs_test,
                                 obs_predict=obs_generated,
                                 name=obs_name,
                                 n_epochs=n_epochs,
                                 n_jets=j + self.n_jets,
                                 range=[0, 8],
                                 weight_samples=iterations,
                                 predict_weights=weights)


        if get(self.params, "plot_Mll", True):
            with PdfPages(f"{path}/M_ll_epochs_{n_epochs}.pdf") as out:
                for j,_ in enumerate(plot_train):
                    obs_name = "M_{\ell \ell}"
                    obs_range = [75,110]
                    bin_num = 40
                    data_train = get_M_ll(plot_train[j])
                    data_test = get_M_ll(plot_test[j])
                    data_generated = get_M_ll(plot_samples[j])
                    weights = plot_weights[j]
                    plot_obs(pp=out,
                             obs_train=data_train,
                             obs_test=data_test,
                             obs_predict=data_generated,
                             name=obs_name,
                             n_epochs=n_epochs,
                             range=obs_range,
                             n_jets=j+self.n_jets,
                             weight_samples=iterations,
                             predict_weights=weights)

        plot_1d_differences = get(self.params, "plot_1d_differences", True)
        if plot_1d_differences:
            if self.n_jets == 1:
                differences = [[2, 6], [2, 10], [6, 10], [5, 9], [1,5], [1,9]]
            elif self.n_jets == 2:
                differences = [[2, 6], [2, 10], [2, 14], [6, 10], [6, 14], [10, 14], [5, 9], [5, 13], [9, 13], [1,5], [1,9], [1,13]]
            else:
                differences = [[2, 6], [2, 10], [2, 14], [6, 10], [6, 14], [10, 14], [5, 9], [5, 13], [9, 13],
                               [2, 18], [6, 18], [10, 18], [14, 18], [5, 17], [9, 17], [13, 17], [1,5], [1,9], [1,13], [1,17]]
    #
            with PdfPages(f"{path}/1d_differences_{n_epochs}.pdf") as out:
                for j, _ in enumerate(plot_train):
                    for channels in differences:
                        channel1 = channels[0]
                        channel2 = channels[1]
                        obs_name = self.obs_names[channel1] + " - " + self.obs_names[channel2]
                        obs_train = plot_train[j][:, channel1] - plot_train[j][:, channel2]
                        obs_test = plot_test[j][:, channel1] - plot_test[j][:, channel2]
                        obs_generated = plot_samples[j][:, channel1] - plot_samples[j][:, channel2]
                        if channels in [[1, 5], [1, 9], [1, 13], [1, 17], [5, 9], [5, 13], [5, 17],
                                        [9, 13], [9, 17], [13, 17]]:
                            obs_train = (obs_train + np.pi)%(2*np.pi) - np.pi
                            obs_test = (obs_test + np.pi)%(2*np.pi) - np.pi
                            obs_generated = (obs_generated + np.pi) % (2 * np.pi) - np.pi
                        weights = plot_weights[j]
                        plot_obs(pp=out,
                                 obs_train=obs_train,
                                 obs_test=obs_test,
                                 obs_predict=obs_generated,
                                 name=obs_name,
                                 n_epochs=n_epochs,
                                 n_jets=j + self.n_jets,
                                 weight_samples=iterations,
                                 predict_weights=weights)


        if get(self.params,"plot_loss", True):
            out = f"{path}/loss_epoch_{n_epochs}.pdf"
            try:
                plot_loss(out, self.train_losses, self.regular_loss, self.kl_loss, self.regularizeGMM_loss, loss_log=get(self.params, "loss_log", True))
            except:
                print("plot_loss failed")
    def plot_toy(self, samples = None, finished=False):
        self.sigma_path = get(self.params, "sigma_path", None)
        os.makedirs(f"plots", exist_ok=True)
        if finished:
            path = f"plots/run{self.runs}"
            os.makedirs(path, exist_ok=True)
            iterations = self.iterations
        else:
            path = "plots"
            if self.iterate_periodically:
                iterations = self.iterations
            else:
                iterations = 1

        n_epochs = self.epoch + get(self.params, "total_epochs", 0)
        with PdfPages(f"{path}/1d_hist_epoch_{n_epochs}.pdf") as out:
            for i in range(0, self.dim_x):
                obs_train = self.data_train[:,i]
                obs_test = self.data_test[:,i]
                obs_generated = samples[:,i]
                # Get the name and the range of the observable
                obs_name = self.obs_names[i]
                obs_range = None if self.obs_ranges==None else self.obs_ranges[i]
                # Create the plot
                plot_obs(pp=out,
                         obs_train=obs_train,
                         obs_test=obs_test,
                         obs_predict=obs_generated,
                         name=obs_name,
                         range=obs_range,
                         n_epochs=n_epochs,
                         n_jets=None,
                         weight_samples=iterations)

        if get(self.params, "toy_type", "ramp") == "gauss_sphere":
            with PdfPages(f"{path}/spherical_{n_epochs}.pdf") as out:
                R_train, phi_train = ToySimulator.getSpherical(self.data_train)
                R_test, phi_test = ToySimulator.getSpherical(self.data_test)
                R_gen, phi_gen = ToySimulator.getSpherical(samples)
                obs_name = "R"
                obs_range = [0.5,1.5]
                plot_obs(pp=out, obs_train=R_train, obs_test=R_test, obs_predict=R_gen,
                     name=obs_name, range=obs_range, weight_samples=iterations)
                for i in range(self.dim_x-1):
                    obs_name=f"\phi_{i}"
                    obs_range = [0, 2*np.pi] if i==self.dim_x-2 else [0, np.pi]
                    obs_train = phi_train[:,i]
                    obs_test = phi_test[:,i]
                    obs_gen = phi_gen[:,i]
                    plot_obs(pp=out, obs_train=obs_train, obs_test=obs_test, obs_predict=obs_gen,
                         name=obs_name, range=obs_range, weight_samples=iterations)

        if get(self.params, "toy_type", "ramp") == "camel":
            n_dim = get(self.params, "n_dim", 2)
            obs_name = "\sum_{i=1}"+f"^{n_dim} x_i"
            out = f"{path}/xsum_epoch_{n_epochs}.pdf"
            obs_train = ToySimulator.get_xsum(self.data_train)
            obs_test = ToySimulator.get_xsum(self.data_test)
            obs_generated = ToySimulator.get_xsum(samples)
            obs_range = [-1.5*n_dim, 1.5*n_dim]
            plot_obs(pp=out, obs_train=obs_train, obs_test=obs_test, obs_predict=obs_generated,
                     name=obs_name, range=obs_range)

        if self.dim_x == 2 and get(self.params,"plot_Deta_Dphi", True):
            out = f"{path}/hist2d_{n_epochs}.pdf"
            plot_obs_2d(pp=out, data_train=self.data_train, data_test=self.data_test, data_generated=samples,
                        obs_ranges=self.obs_ranges, obs_names=self.obs_names, n_epochs=n_epochs)

        if get(self.params,"plot_loss", False):
            out = f"{path}/loss_epoch_{n_epochs}.pdf"
            plot_loss(out, self.train_losses, self.regular_loss, self.kl_loss, self.regularizeGMM_loss, loss_log=get(self.params, "loss_log", True))

