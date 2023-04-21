import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from Source.Util.lr_scheduler import OneCycleLR
from Source.Models.inn import INN
from Source.Models.tbd import TBD
from Source.Models.ddpm import DDPM
from matplotlib.backends.backend_pdf import PdfPages
from Source.Util.plots import plot_obs, delta_r, plot_deta_dphi
from Source.Util.preprocessing import preprocess, undo_preprocessing
from Source.Util.datasets import Dataset
from Source.Util.util import get_device, save_params, get, load_params, magic_trafo
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

        self.channels = get(self.params, "channels", None)
        self.n_jets = get(self.params, "n_jets", 2)
        if self.channels is None:
            self.channels = np.array([i for i in range(self.n_jets * 4 + 8) if i not in [1, 3, 7]]).tolist()

        if self.conditional:
            self.n_con = 11
            self.params["n_con"] = self.n_con
            self.prior_path = get(self.params,"prior_path", None)
            self.prior_params = load_params(os.path.join(self.prior_path, "paramfile.yaml"))
            self.prior_channels = get(self.prior_params, "channels", None)
            if get(self.params,"plot_channels",None) is None:
                self.plot_channels = self.prior_channels + self.channels
                self.params["plot_channels"] = self.plot_channels
            else:
                self.plot_channels = get(self.params,"plot_channels",None)
        else:
            if get(self.params, "plot_channels", None) is None:
                self.plot_channels = self.channels
                self.params["plot_channels"] = self.channels
            else:
                self.plot_channels = get(self.params, "plot_channels", None)

        self.starttime = time.time()

    def full_run(self):
        self.prepare_experiment()
        self.load_data()
        if not self.load_dataset:
            self.data_raw = self.z_2

        if self.conditional:
            self.prior_raw = self.z_1
            print(self.prior_raw.shape)
            self.prior_data, self.prior_mean, self.prior_std, self.prior_u, self.prior_s, self.prior_bin_edges, self.prior_bin_means, self.prior_raw = \
                self.preprocess_data(self.prior_params,self.prior_raw, save_in_params=False,conditional=True)
            print(self.prior_raw.shape, self.prior_data.shape)
            self.prior_data = self.prior_data[self.prior_data[:,0]!=1]
            self.prior_raw = self.prior_raw[self.prior_raw[:,0]!=1]
            print(self.prior_raw.shape, self.prior_data.shape)
            self.new_data, self.data_mean, self.data_std, self.data_u, self.data_s, self.data_bin_edges, self.data_bin_means, self.new_raw = \
                self.preprocess_data(self.params, self.data_raw, conditional=True)
            print(self.new_data.shape)
            print(self.new_raw.shape)

            self.data = torch.concat([self.prior_data[:,1:12], self.new_data[:,2:]], dim=1)
            self.data_raw = np.concatenate([self.prior_raw[:,:13], self.new_raw[:,13:]], axis=1)

            print(self.data.shape)
            print(self.data_raw.shape)

            self.magic_transformation = get(self.params, "magic_transformation", False)
            if self.magic_transformation:
                deltaR12 = delta_r(self.data_raw[:, 1:], idx_phi1=9, idx_eta1=10, idx_phi2=13, idx_eta2=14)
                self.event_weights = magic_trafo(deltaR12)
                print(self.event_weights.shape)
                self.data = torch.cat([self.data, torch.from_numpy(self.event_weights[:, None]).to(self.device)],
                                      dim=1).float()
                print(f"preprocess_data: Using magic transformation")

                print(self.event_weights)
                print(self.event_weights.mean())

        else:
            self.data, self.data_mean, self.data_std, self.data_u, self.data_s, self.data_bin_edges, self.data_bin_means, self.data_raw = \
                self.preprocess_data(self.params, self.data_raw, save_in_params=True)

            self.magic_transformation = get(self.params, "magic_transformation", False)
            if self.magic_transformation:
                deltaR12 = delta_r(self.data_raw, idx_phi1=9, idx_eta1=10, idx_phi2=13, idx_eta2=14)
                self.event_weights = magic_trafo(deltaR12)
                print(self.event_weights.shape)
                self.data = torch.cat([self.data, torch.from_numpy(self.event_weights[:, None]).to(self.device)], dim=1).float()
                print(f"preprocess_data: Using magic transformation")

                print(self.event_weights)
                print(self.event_weights.mean())

        print(f"preprocess_data: input shape is {self.data.shape}")
        self.n_data = len(self.data)


        if self.warm_start:
            self.model = self.build_model(self.params, prior_path=self.warm_start_path, save_in_params=True)
        else:
            self.model = self.build_model(self.params, save_in_params=True)

        if self.conditional:
            self.prior_model = self.build_model(self.prior_params, self.prior_path)
            self.model.prior_mean, self.model.prior_std, self.model.prior_u, self.model.prior_s, self.model.prior_bin_edges, self.model.prior_bin_means \
                = self.prior_mean, self.prior_std, self.prior_u, self.prior_s, self.prior_bin_edges, self.prior_bin_means
            self.model.prior_channels = self.prior_channels
            self.model.prior_params = self.prior_params
        else:
            self.model.prior_mean, self.model.prior_std, self.model.prior_u, self.model.prior_s, self.model.prior_bin_edges, self.model.prior_bin_means \
                = None, None, None, None, None, None

        self.model.data_mean, self.model.data_std, self.model.data_u,self.model.data_s, self.model.data_bin_edges, self.model.data_bin_means  \
            = self.data_mean, self.data_std, self.data_u, self.data_s, self.data_bin_edges, self.data_bin_means

        self.model.obs_names = self.obs_names
        self.model.obs_ranges = self.obs_ranges
        self.build_optimizer()
        self.build_dataloaders()
        self.train_model()
        self.generate_samples()
        self.make_plots()
        self.finish_up()
