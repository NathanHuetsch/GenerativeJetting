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


class Z1_Experiment(Experiment):
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

        self.con_depth = get(self.params, "con_depth", 0)
        self.channels = get(self.params, "channels", None)
        self.n_jets = get(self.params, "n_jets", 1)
        if self.channels is None:
            self.channels = np.array([i for i in range(self.n_jets * 4 + 8) if i not in [1, 3, 7]]).tolist()

        if self.conditional:
            self.n_con = 3
            self.params['n_con'] = 3

        if get(self.params, "plot_channels", None) is None:
            self.plot_channels = self.channels
            self.params["plot_channels"] = self.plot_channels

        self.starttime = time.time()

    def full_run(self):
        self.prepare_experiment()
        self.load_data()
        if not self.load_dataset:
            self.data_raw = self.z_1

        self.data, self.data_mean, self.data_std, self.data_u, self.data_s, self.data_bin_edges, self.data_bin_means, self.data_raw = \
            self.preprocess_data(self.params, self.data_raw, save_in_params=True, conditional=self.conditional)

        print(f"preprocess_data: input shape is {self.data.shape}")
        self.n_data = len(self.data)

        if self.warm_start:
            self.model = self.build_model(self.params, prior_path=self.warm_start_path, save_in_params=True)
        else:
            self.model = self.build_model(self.params, save_in_params=True)

        self.model.data_mean, self.model.data_std, self.model.data_u, self.model.data_s, self.model.data_bin_edges, self.model.data_bin_means \
            = self.data_mean, self.data_std, self.data_u, self.data_s, self.data_bin_edges, self.data_bin_means
        self.model.obs_names = self.obs_names
        self.model.obs_ranges = self.obs_ranges
        self.build_optimizer()
        self.build_dataloaders()
        self.train_model()
        self.generate_samples()
        self.make_plots()
        self.finish_up()