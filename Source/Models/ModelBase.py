import numpy as np
import torch
import torch.nn as nn
import os, time
from torch.utils.tensorboard import SummaryWriter
from Source.Util.util import get, get_device
from Source.Util.preprocessing import undo_preprocessing
from Source.Util.discretize import undo_discretize
from Source.Util.plots import plot_obs, delta_r, plot_deta_dphi, plot_obs_2d, plot_loss, plot_binned_sigma, plot_mu_sigma
from Source.Util.physics import get_M_ll
from Source.Util.simulateToyData import ToySimulator
from matplotlib.backends.backend_pdf import PdfPages

import cv2
import os
from natsort import natsorted, ns



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
        self.conditional = get(self.params,'conditional',False)
        self.n_con = get(self.params,'n_con',0)
        self.n_jets = get(self.params,'n_jets',2)
        self.con_depth = get(self.params,'con_depth',0)
        self.batch_size = self.params["batch_size"]
        self.batch_size_sample = get(self.params, "batch_size_sample", self.batch_size)
        self.istoy = get(self.params, "istoy", False)
        self.epoch = get(self.params, "total_epochs", 0)
        self.net = self.build_net()
        self.iterations = get(self.params,"iterations", 1)
        self.regular_loss = []
        self.kl_loss = []
        self.regularizeGMM_loss = []
        self.runs = get(self.params, "runs", 0)
        self.iterate_periodically = get(self.params, "iterate_periodically", False)
    def build_net(self):
        pass

    def prepare_training(self):
        print("train_model: Preparing model training")
        self.use_scheduler = get(self.params, "use_scheduler", False)
        self.train_losses = np.array([])
        self.train_losses_epoch = np.array([])
        self.n_trainbatches = len(self.train_loader)

        self.sample_periodically = get(self.params, "sample_periodically", False)
        if self.sample_periodically:
            self.sample_every = get(self.params, "sample_every", 1)
            self.sample_every_n_samples = get(self.params, "sample_every_n_samples", 100000)
            print(f'train_model: sample_periodically set to True. Sampling {self.sample_every_n_samples} every'
                  f' {self.sample_every} epochs. This may significantly slow down training!')

        self.log = get(self.params, "log", True)
        if self.log:
            log_dir = os.path.join(self.params["out_dir"], "logs")
            self.logger = SummaryWriter(log_dir)
            print(f"train_model: Logging to log_dir {log_dir}")
        else:
            print("train_model: log set to False. No logs will be written")

    def run_training(self):

        self.prepare_training()
        samples = []
        n_epochs = get(self.params, "n_epochs", 100)
        past_epochs = get(self.params, "total_epochs", 0)
        print(f"train_model: Model has been trained for {past_epochs} epochs before.")
        print(f"train_model: Beginning training. n_epochs set to {n_epochs}")
        for e in range(n_epochs):
            t0 = time.time()
            self.epoch = past_epochs + e
            self.train()
            self.train_one_epoch()

            if self.sample_periodically:
                if (self.epoch + 1) % self.sample_every == 0:
                    self.eval()
                    if not self.istoy:
                        samples = self.sample_and_undo(self.sample_every_n_samples)
                        self.plot_samples(samples=samples)
                    else:
                        iterations = self.iterations if (self.iterate_periodically and self.bayesian) else 1
                        bay_samples = []
                        for i in range(0, iterations):
                            sample = self.sample_n(self.sample_every_n_samples)
                            if self.params["model"] == "AutoRegBinned":
                                sample = undo_discretize(sample, self.params, self.bin_edges, self.bin_means)
                            bay_samples.append(sample)

                        samples = np.concatenate(bay_samples)
                        self.plot_toy(samples=samples)

            if get(self.params,"save_periodically",False):
                if (self.epoch + 1) % get(self.params,"save_every",10) == 0 or self.epoch==0:
                    torch.save(self.state_dict(), f"models/model_epoch_{e+1}.pt")


            if e==0:
                t1 = time.time()
                dtEst= (t1-t0) * n_epochs
                print(f"Training time estimate: {dtEst/60:.2f} min = {dtEst/60**2:.2f} h")

    def train_one_epoch(self):
        train_losses = np.array([])
        for batch_id, x in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            loss = self.batch_loss(x)

            #loss_m = self.train_losses[-1000:].mean()
            #loss_s = self.train_losses[-1000:].std()

            if np.isfinite(loss.item()): # and (abs(loss.item() - loss_m) / loss_s < 5 or len(self.train_losses_epoch) == 0):
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

    def sample_n(self, n_samples, prior_samples=None, con_depth=0):
        pass

    def sample_and_undo(self, n_samples):
        if self.conditional and self.n_jets ==2:
            prior_samples = self.prior_model.sample_n(n_samples+self.batch_size, con_depth=self.con_depth)
            samples = self.sample_n(n_samples, prior_samples=prior_samples,
                               con_depth=self.con_depth)
            prior_samples = undo_preprocessing(prior_samples, self.prior_mean, self.prior_std,
                                                self.prior_u, self.prior_s, self.prior_bin_edges,
                                               self.prior_bin_means, self.prior_params)
            samples = undo_preprocessing(samples, self.data_mean, self.data_std, self.data_u, self.data_s,
                                          self.data_bin_means, self.data_bin_edges, self.params)

            samples = np.concatenate([prior_samples[:n_samples, :13], samples[:, 13:]], axis=1)

        elif self.conditional and self.n_jets == 3:
            prior_prior_samples = self.prior_prior_model.sample_n(n_samples + 2*self.batch_size,
                                                         con_depth=self.con_depth)
            prior_samples = self.prior_model.sample_n(n_samples + self.batch_size, prior_samples=prior_prior_samples,
                                                 con_depth=self.con_depth)

            priors = np.concatenate([prior_prior_samples[:n_samples + self.batch_size_sample,3:12],prior_samples[:,2:6]], axis=1)
            samples = self.sample_n(n_samples, prior_samples=priors, con_depth=self.con_depth)
            prior_prior_samples = undo_preprocessing(prior_prior_samples, self.prior_prior_mean, self.prior_prior_std,
                                           self.prior_prior_u, self.prior_prior_s, self.prior_prior_bin_edges,
                                                     self.prior_prior_bin_means, self.prior_prior_params)
            prior_samples = undo_preprocessing(prior_samples, self.prior_mean, self.prior_std,
                                           self.prior_u, self.prior_s, self.prior_bin_edges, self.prior_bin_means, self.prior_params)
            samples = undo_preprocessing(samples, self.data_mean, self.data_std,
                                     self.data_u, self.data_s, self.data_bin_edges, self.data_bin_means, self.params)

            samples = np.concatenate([prior_prior_samples[:n_samples, 1:13], prior_samples[:n_samples, 13:17],
                                      samples[:,16:]], axis=1)

        else:
            samples = self.sample_n(n_samples, con_depth=self.con_depth)
            samples = undo_preprocessing(samples, self.data_mean, self.data_std, self.data_u, self.data_s,
                                         self.data_bin_edges, self.data_bin_means, self.params)

        return samples

    def plot_samples(self, samples, finished=False):
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

        plot_train = []
        plot_test = []
        plot_samples = []


        if self.conditional and self.n_jets !=3:
            for i in range(self.n_jets, 4):
                plot_train_jets = self.data_train[self.data_train[:, 0] == i]
                plot_train_jets = plot_train_jets[:,1:]
                plot_train.append(plot_train_jets)

                plot_test_jets = self.data_test[self.data_test[:, 0] == i]
                plot_test_jets = plot_test_jets[:,1:]
                plot_test.append(plot_test_jets)

                plot_samples_jets = samples[samples[:, 0] == i]
                plot_samples_jets = plot_samples_jets[:,1:]
                plot_samples.append(plot_samples_jets)

        else:
            plot_train.append(self.data_train)
            plot_test.append(self.data_test)
            plot_samples.append(samples)

        with PdfPages(f"{path}/1d_hist_epoch_{n_epochs}.pdf") as out:
            for j, _ in enumerate(plot_train):
                # Loop over the plot_channels
                for i, channel in enumerate(self.params["plot_channels"]):
                    # Get the train data, test data and generated data for the channel
                    obs_train = plot_train[j][:, channel]
                    obs_test = plot_test[j][:, channel]
                    obs_generated = plot_samples[j][:, channel]
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
                             weight_samples=iterations)

        if all(c in self.params["plot_channels"] for c in [9, 10, 13, 14]):
            if get(self.params,"plot_deltaR", True):
                obs_name = "\Delta R_{j_1 j_2}"
                with PdfPages(f"{path}/deltaR_jl_jm_epoch_{n_epochs}.pdf") as out:
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
                             range=[0, 8],
                             weight_samples=iterations)
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
                                 weight_samples=iterations)
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
                                 weight_samples=iterations)
            if get(self.params,"plot_Deta_Dphi", True):
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
        else:
            print("make_plots: Missing at least one required channel to plot DeltaR and/or dphi_deta")

        if get(self.params, "plot_Mll", False):
            with PdfPages(f"{path}/M_ll_epochs_{n_epochs}.pdf") as out:
                for j,_ in enumerate(plot_train):
                    obs_name = "M_{\ell \ell}"
                    obs_range = [75,110]
                    data_train = get_M_ll(plot_train[j])
                    data_test = get_M_ll(plot_test[j])
                    data_generated = get_M_ll(plot_samples[j])
                    plot_obs(pp=out,
                             obs_train=data_train,
                             obs_test=data_test,
                             obs_predict=data_generated,
                             name=obs_name,
                             n_epochs=n_epochs,
                             range=obs_range,
                             n_jets=j+self.n_jets,
                             weight_samples=iterations)

        if get(self.params,"plot_loss", False):
            out = f"{path}/loss_epoch_{n_epochs}.pdf"
            plot_loss(out, self.train_losses, self.regular_loss, self.kl_loss, self.regularizeGMM_loss, loss_log=get(self.params, "loss_log", True))

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
            for i in range(0, self.dim):
                obs_train = self.data_train[:,i]
                obs_test = self.data_test[:,i]
                obs_generated = samples[:,i]
                obs_name = self.obs_names[i]
                obs_range = None if self.obs_ranges==None else self.obs_ranges[i]
                plot_obs(pp=out,
                         obs_train=obs_train,
                         obs_test=obs_test,
                         obs_predict=obs_generated,
                         name=obs_name,
                         range=obs_range,
                         n_epochs=n_epochs,
                         n_jets=None,
                         weight_samples=iterations)
        if get(self.params, "plot_sigma", False) and iterations > 1:
            with PdfPages(f"{path}/binned_sigma_{n_epochs}.pdf") as out:
                for i in range(0, self.dim):
                    obs_generated = samples[:, i]
                    # Get the name and the range of the observable
                    obs_name = self.obs_names[i]
                    obs_range = None if self.obs_ranges == None else self.obs_ranges[i]
                    # Create the plot
                    if self.sigma_path is not None:
                        save_path = self.sigma_path + f"_{i}"
                    else:
                        save_path = None
                    plot_binned_sigma(pp=out,
                             obs_predict=obs_generated,
                             name=obs_name,
                             range=obs_range,
                             n_epochs=n_epochs,
                             weight_samples=iterations,
                             save_path=save_path)

        if get(self.params, "plot_mu_sigma",False) and iterations > 1:
            with PdfPages(f"{path}/mu_sigma_{n_epochs}.pdf") as out:
                for i in range(0, self.dim):
                    obs_generated = samples[:, i]
                    # Get the name and the range of the observable
                    obs_name = self.obs_names[i]
                    obs_range = None if self.obs_ranges == None else self.obs_ranges[i]
                    # Create the plot
                    plot_mu_sigma(pp=out,
                             obs_predict=obs_generated,
                             name=obs_name,
                             range=obs_range,
                             n_epochs=n_epochs,
                             weight_samples=iterations)


        if get(self.params, "toy_type", "ramp") == "gauss_sphere":
            with PdfPages(f"{path}/spherical_{n_epochs}.pdf") as out:
                R_train, phi_train = ToySimulator.getSpherical(self.data_train)
                R_test, phi_test = ToySimulator.getSpherical(self.data_test)
                R_gen, phi_gen = ToySimulator.getSpherical(samples)
                obs_name = "R"
                obs_range = [0,2]
                plot_obs(pp=out, obs_train=R_train, obs_test=R_test, obs_predict=R_gen,
                     name=obs_name, range=obs_range, weight_samples=iterations)

                for i in range(self.dim-1):
                    obs_name=f"\phi_{i}"
                    obs_range = [0, 2*np.pi] if i==self.dim-2 else [0, np.pi]
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

        if self.dim == 2 and get(self.params,"plot_Deta_Dphi", True):
            out = f"{path}/hist2d_{n_epochs}.pdf"
            plot_obs_2d(pp=out, data_train=self.data_train, data_test=self.data_test, data_generated=samples,
                        obs_ranges=self.obs_ranges, obs_names=self.obs_names, n_epochs=n_epochs)

        if get(self.params,"plot_loss", False):
            out = f"{path}/loss_epoch_{n_epochs}.pdf"
            plot_loss(out, self.train_losses, self.regular_loss, self.kl_loss, self.regularizeGMM_loss, loss_log=get(self.params, "loss_log", True))

    def toy_video(self, samples = None):
        n_epochs = self.epoch + get(self.params, "total_epochs", 0)
        path = f"videos/epoch_{n_epochs}"
        os.makedirs(path, exist_ok=True)

        video_dim = get(self.params, "video_dim", self.dim)
        for i in range(0, video_dim):
            image_folder = f"{path}/dim_{i}"
            os.makedirs(image_folder, exist_ok=True)
            obs_train = self.data_train[:, i]
            obs_test = self.data_test[:, i]
            obs_name = self.obs_names[i]
            obs_range = None if self.obs_ranges == None else self.obs_ranges[i]
            frames = samples.shape[-1]
            for t in range(frames):
                out = f"{path}/dim_{i}/timestep_{t}.png"
                obs_generated = samples[:,i, t]

                plot_obs(pp=out,
                         obs_train=obs_train,
                         obs_test=obs_test,
                         obs_predict=obs_generated,
                         name=obs_name,
                         range=obs_range)

            video_name = f"videos/epoch_{n_epochs}_dim_{i}_frames{frames}.mp4"

            images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
            images = natsorted(images)
            frame = cv2.imread(os.path.join(image_folder, images[0]))

            height, width, layers = frame.shape

            video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), 5, (width, height))

            for image in images:
                video.write(cv2.imread(os.path.join(image_folder, image)))

            cv2.destroyAllWindows()
            video.release()
