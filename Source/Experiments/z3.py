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


class Z3_Experiment(Experiment):
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
        self.n_jets = get(self.params, "n_jets", 3)
        if self.channels is None:
            self.channels = np.array([i for i in range(self.n_jets * 4 + 8) if i not in [1, 3, 7]]).tolist()

        if self.conditional:
            self.n_con = 13
            self.params["n_con"] = self.n_con
            self.prior_path = get(self.params,"prior_path", None)
            self.prior_params = load_params(os.path.join(self.prior_path, "paramfile.yaml"))
            self.prior_channels = get(self.prior_params, "channels", None)
            self.prior_prior_path = get(self.prior_params, "prior_path", None)
            self.prior_prior_params = load_params(os.path.join(self.prior_prior_path, "paramfile.yaml"))
            self.prior_prior_channels = get(self.prior_prior_params, "channels", None)
            if get(self.params,"plot_channels",None) is None:
                self.plot_channels = self.prior_prior_channels + self.prior_channels + self.channels
                self.params["plot_channels"] = self.plot_channels
        else:
            if get(self.params, "plot_channels", None) is None:
                self.plot_channels = self.channels
                self.params["plot_channels"] = self.channels

            self.n_con = 0
            self.params['n_con'] = self.n_con

        self.starttime = time.time()

    def full_run(self):
        self.prepare_experiment()
        self.load_data()
        if not self.load_dataset:
            self.data_raw = self.z_3

        if self.conditional:
            self.prior_prior_raw = self.z_1
            self.prior_raw = self.z_2

            self.prior_prior_data, self.prior_prior_mean, self.prior_prior_std, self.prior_prior_u, self.prior_prior_s, \
            self.prior_prior_bin_edges, self.prior_prior_bin_means, self.prior_prior_raw = \
                self.preprocess_data(self.prior_prior_params, self.prior_prior_raw,save_in_params=False, conditional=True)
            self.prior_prior_data = self.prior_prior_data[self.prior_prior_data[:, 2] == 1]
            self.prior_prior_raw = self.prior_prior_raw[self.prior_prior_raw[:, 0] == 3]

            self.prior_data, self.prior_mean, self.prior_std, self.prior_u, self.prior_s, self.prior_bin_edges, self.prior_bin_means, self.prior_raw = \
                self.preprocess_data(self.prior_params,self.prior_raw, save_in_params=False,conditional=True)
            self.prior_data = self.prior_data[self.prior_data[:,1]==1]
            self.prior_raw = self.prior_raw[self.prior_raw[:,0]==3]
            self.new_data, self.data_mean, self.data_std, self.data_u, self.data_s, self.data_bin_edges, self.data_bin_means, self.new_raw = \
                self.preprocess_data(self.params, self.data_raw, conditional=True)

            self.data = torch.concat([self.prior_prior_data[:,3:12], self.prior_data[:,2:6],self.new_data], dim=1)
            self.data_raw = np.concatenate([self.prior_prior_raw[:,1:13], self.prior_raw[:,13:17], self.new_raw[:,17:]], axis=1)

        else:
            self.data, self.data_mean, self.data_std, self.data_u, self.data_s, self.data_bin_edges, self.data_bin_means, self.data_raw = \
                self.preprocess_data(self.params, self.data_raw, save_in_params=True)

        print(f"preprocess_data: input shape is {self.data.shape}")
        self.n_data = len(self.data)

        if self.warm_start:
            self.model = self.build_model(self.params, prior_path=self.warm_start_path, save_in_params=True)
        else:
            self.model = self.build_model(self.params, save_in_params=True)

        if self.conditional:
            #Build first model for the first 12(9) dimensions
            self.prior_prior_model = self.build_model(self.prior_prior_params, self.prior_prior_path)
            self.model.prior_prior_mean, self.model.prior_prior_std, self.model.prior_prior_u, self.model.prior_prior_s, self.model.prior_prior_bin_edges, self.model.prior_prior_bin_means \
                = self.prior_prior_mean, self.prior_prior_std, self.prior_prior_u, self.prior_prior_s, self.prior_prior_bin_edges, self.prior_prior_bin_means
            self.model.prior_prior_channels = self.prior_prior_channels

            #Build second model for the following 4 dimensions
            self.prior_model = self.build_model(self.prior_params, self.prior_path)
            self.model.prior_mean, self.model.prior_std, self.model.prior_u, self.model.prior_s, self.model.prior_bin_edges, self.model.prior_bin_means \
                = self.prior_mean, self.prior_std, self.prior_u, self.prior_s, self.prior_bin_edges, self.prior_bin_means
            self.model.prior_channels = self.prior_channels

            self.model.prior_params = self.prior_params
            self.model.prior_prior_params = self.prior_prior_params
        else:
            self.model.prior_mean, self.model.prior_mean, self.model.prior_u, self.model.prior_s, self.model.prior_bin_edges,\
                self.model.prior_bin_means = None, None, None, None, None, None
            self.model.prior_prior_mean, self.model.prior_prior_mean, self.model.prior_prior_u, self.model.prior_prior_s,\
            self.model.prior_prior_bin_edges, self.model.prior_prior_bin_means = None, None, None, None, None, None

        self.model.data_mean, self.model.data_std, self.model.data_u,self.model.data_s, \
        self.model.data_bin_edges, self.model.data_bin_means = \
            self.data_mean, self.data_std, self.data_u, self.data_s, self.data_bin_edges, self.data_bin_means

        self.model.obs_names = self.obs_names
        self.model.obs_ranges = self.obs_ranges
        self.build_optimizer()
        self.build_dataloaders()
        self.train_model()
        self.generate_samples()
        self.make_plots()
        self.finish_up()