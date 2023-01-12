import numpy as np
import torch
from torch.utils.data import DataLoader
from Source.Models.inn import INN
from Source.Models.tbd import TBD
from Source.Models.ddpm import DDPM
from matplotlib.backends.backend_pdf import PdfPages
from Source.Util.plots import plot_obs, delta_r, plot_deta_dphi
from Source.Util.preprocessing import preprocess, undo_preprocessing
from Source.Util.datasets import Dataset
from Source.Util.util import get_device, save_params, get, load_params
from Source.Experiments.ExperimentBase import Experiment
import time
from datetime import datetime
import sys
import os
import h5py
import pandas
from torch.optim import Adam


class Z2_Experiment(Experiment):
    """
    Class to run Z+2jet generative modelling experiments

    TODO: Implement logging
    """

    def __init__(self, params):
        """
        The __init__ method reads in the parameters and saves them under self.params
        It also makes some useful definitions
        """
        super().__init__(params)

        self.n_jets = 2
        self.con_depth = 1
        self.n_dim = 4
        self.channels = get(self.params, "channels", None)

        if self.conditional:
            self.prior_path = get(self.params,"prior_path", None)
            self.prior_params = load_params(os.path.join(self.prior_path, "paramfile.yaml"))
            self.prior_channels = get(self.prior_params, "channels", None)

        self.starttime = time.time()

    def full_run(self):
        self.prepare_experiment()
        self.data_raw, self.prior_data_raw, _ = self.load_data(self.params)

        if self.conditional:
            self.prior_data, self.prior_mean, self.prior_std, self.prior_u, self.prior_s, self.prior_raw = \
                self.preprocess_data(self.prior_params,self.prior_data_raw, conditional=True)
            self.prior_data = self.prior_data[self.prior_data[:,-3]!=1]
            self.prior_raw = self.prior_raw[self.prior_raw[:,-1]!=1]
            self.new_data, self.data_mean, self.data_std, self.data_u, self.data_s, self.new_raw = \
                self.preprocess_data(self.params, self.data_raw, save_in_params=True, conditional=True)

            self.data = torch.concat([self.prior_data[:,:-3], self.new_data], dim=1)
            self.data_raw = np.concatenate([self.prior_raw[:,:12], self.new_raw[:,12:]], axis=1)

        else:
            self.data, self.data_mean, self.data_std, self.data_u, self.data_s, self.data_raw = \
                self.preprocess_data(self.params, self.data_raw, save_in_params=True)

        print(f"preprocess_data: input shape is {self.data.shape}")
        self.n_data = len(self.data)

        if self.warm_start:
            self.model = self.build_model(self.params, prior_path=self.warm_start_path, save_in_params=True)
        else:
            self.model = self.build_model(self.params, save_in_params=True)

        if self.conditional:
            self.prior_model = self.build_model(self.prior_params, self.prior_path)
        self.model.optimizer = self.build_optimizer(self.params, self.model.parameters())
        self.build_dataloaders(self.params)
        self.train_model(self.params)
        self.generate_samples()
        self.make_plots()
        self.finish_up()

    def generate_samples(self):
        """
        The generate_samples method uses the trained or loaded model to generate samples.
        Currently, the sampling code is hidden as part of the model classes to keep the ExperimentClass shorter.
        All models have a sample_n_parallel() method, that performs the sampling.

        New model classes implemented in this framework must have a sample_n_parallel(n_samples) method.
        See documentation of ModelBase class for more guidelines on how to implement new model classes.

        Overwrite this method if a different way of sampling is needed.
        """

        # Read in the "sample" parameter. If it is set to True, perform the sampling, otherwise skip it.
        sample = get(self.params, "sample", True)
        if sample:
            # Read in the "load_best_checkpoint" parameter.
            # If it is set to True, try to load the best validation checkpoint of the model.
            # Otherwise, just use the model state after the last epoch
            load_best_checkpoint = get(self.params, "load_best_checkpoint", True)
            if load_best_checkpoint:
                try:
                    state_dict = torch.load(self.out_dir + "/models/checkpoint.pt", map_location=self.device)
                    self.model.load_state_dict(state_dict)
                except Exception:
                    print(f"generate_samples: cannot load best checkpoint. Sampling with current model")

            # Read in the "n_samples" parameter specifying how many samples to generate
            # Call the model.sample_n_parallel(n_samples) method to perform the sampling
            n_samples = get(self.params, "n_samples", 1000000)
            print(f"generate_samples: Starting generation of {n_samples} samples")
            t0 = time.time()

            if self.conditional:
                self.prior_samples = self.prior_model.sample_n(n_samples+self.batch_size, con_depth=self.con_depth)
            else:
                self.prior_samples = None
            self.samples = self.model.sample_n(n_samples, prior_samples=self.prior_samples, con_depth=self.con_depth)
            t1 = time.time()
            sampletime = t1 - t0
            self.params["sampletime"] = sampletime

            # Undo the preprocessing of the samples
            # TODO: Currently they are mapped back to 16dim for this. Could be made more efficient
            if get(self.params, "preprocess", True):
                if self.conditional:
                    self.prior_samples = undo_preprocessing(self.prior_samples, self.prior_mean, self.prior_std,
                                                            self.prior_u, self.prior_s, self.prior_channels,
                                                            keep_all = True, conditional=self.conditional,
                                                            n_jets=1)
                    self.samples = undo_preprocessing(self.samples, self.data_mean, self.data_std,
                                                            self.data_u, self.data_s, self.channels,
                                                            keep_all=True, conditional=self.conditional,
                                                            n_jets=self.n_jets)
                    self.samples = np.concatenate([self.prior_samples[:n_samples,:12],self.samples[:,12:]], axis=1)
                else:
                    self.samples = undo_preprocessing(self.samples,
                                                  self.data_mean, self.data_std, self.data_u, self.data_s,
                                                  self.channels, keep_all=True, conditional=self.conditional,
                                                  n_jets=self.n_jets)
            print(f"generate_samples: Finished generation of {n_samples} samples after {sampletime} seconds")
            if get(self.params, "save_samples", False):
                os.makedirs('samples', exist_ok=True)
                np.save("samples/samples_final.npy", self.samples)
                print(f"save_samples: generated samples have been saved")
        else:
            print("generate_samples: sample set to False")

    def make_plots(self):
        """
        The make_plots method uses the train data, the test data and the generated samples to draw a range of plots.

        Overwrite this method if other plots are required.
        """

        # Read in the "plot" and "sample" parameters. If both are set to True, perform make the plots, otherwise skip it.
        plot = get(self.params, "plot", True)
        sample = get(self.params, "sample", True)
        if plot and sample:
            # Transform the raw_data from (E, p_x, p_y, p_z) to (p_T, phi, eta, m)
            # self.data_raw = EpppToPTPhiEta(self.data_raw, reduce_data=False, include_masses=True)
            # Create a directory to save the plots
            os.makedirs(f"plots/run{self.runs}", exist_ok=True)
            self.plot_channels = get(self.params, "plot_channels", None)

            # Read in the "plot_channels" parameter, specifying for which observables we want a 1d histogram
            # Default to all observables if not specified
            if self.plot_channels is None:
                self.plot_channels = self.prior_channels + self.channels
                print(f"make_plots: plot_channels not specified. Defaulting to all channels "
                      f"{self.prior_channels + self.channels}")

            # The cut between train and test data
            cut = int(self.n_data * (self.data_split[0] + self.data_split[1]))

            # Loop over the number of jets
            plot_train = []
            plot_test = []
            plot_samples = []
            if self.conditional:
                for i in range(self.n_jets, 4):
                    plot_train_jets = self.data_raw[:cut][self.data_raw[:cut, -1] == i]
                    plot_train.append(plot_train_jets)

                    plot_test_jets = self.data_raw[cut:][self.data_raw[cut:, -1] == i]
                    plot_test.append(plot_test_jets)

                    plot_samples_jets = self.samples[self.samples[:, -1] == i]
                    plot_samples.append(plot_samples_jets)

            else:
                plot_train.append(self.data_raw[cut:])
                plot_test.append(self.data_raw[:cut])
                plot_samples.append(self.samples)

            # Draw all 1d histograms into one PDF file
            with PdfPages(f"plots/run{self.runs}/1d_histograms") as out:
                # Loop over the plot_channels
                for j, _ in enumerate(plot_train):
                    for i, channel in enumerate(self.plot_channels):
                        # Get the train data, test data and generated data for the channel
                        obs_train = plot_train[j][:, channel]
                        obs_test = plot_test[j][:, channel]
                        obs_generated = plot_samples[j][:, channel]
                        # Get the name and the range of the observable
                        obs_name = self.obs_names[channel]
                        obs_range = self.obs_ranges[channel]
                        n_epochs = self.total_epochs -1
                        # Create the plot
                        plot_obs(pp=out,
                             obs_train=obs_train,
                             obs_test=obs_test,
                             obs_predict=obs_generated,
                             name=obs_name,
                             range=obs_range,
                             n_epochs=n_epochs,
                             n_jets=j+self.n_jets,
                             conditional=self.conditional)

            # Draw a 1d histogram for the DeltaR between the two jets
            # This requires that the channels 9,10,13,14 where part of the experiment
            plot_DeltaR = get(self.params, "plot_deltaR", True)
            if plot_DeltaR and self.n_jets > 1:
                if all(c in self.plot_channels for c in [9, 10, 13, 14]):
                    obs_name = "\Delta R_{j_1 j_2}"
                    with PdfPages(f"plots/run{self.runs}/deltaR_j1_j2.pdf") as out:
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
                                    n_jets=j+self.n_jets,
                                    conditional=self.conditional,
                                    range=[0, 8])



                else:
                    print("make_plots: plot_deltaR ist set to True, but missing at least one required channel")

            # Draw a 2d histogram DeltaEta vs DeltaPhi for the two jets
            # This requires that the channels 9,10,13,14 where part of the experiment
            plot_Deta_Dphi = get(self.params, "plot_Deta_Dphi", True)
            if plot_Deta_Dphi and self.n_jets > 1:
                for j, _ in enumerate(plot_train):
                    file_name = f"plots/run{self.runs}/deta_dphi_jets_{j + self.n_jets}.pdf"
                    plot_deta_dphi(file_name=file_name,
                                   data_train=plot_train[j],
                                   data_test=plot_test[j],
                                   data_generated=plot_samples[j],
                                   n_jets=j + self.n_jets,
                                   conditional=self.conditional,
                                   n_epochs=n_epochs)
            else:
                print("make_plots: plot_Deta_Dphi ist set to True, but missing at least one required channel")

            print("make_plots: Finished making plots")
        else:
            print("make_plots: plot set to False")

