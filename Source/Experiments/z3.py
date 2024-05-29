import numpy as np
import torch
from torch.utils.data import DataLoader
from Source.Models.cfm import CFM
from matplotlib.backends.backend_pdf import PdfPages
from Source.Util.plots import plot_obs, delta_r, plot_deta_dphi
from Source.Util.preprocessing import preprocess, undo_preprocessing
from Source.Util.util import get_device, save_params, get, load_params, magic_trafo
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

    def full_run(self):
        self.prepare_experiment()
        self.load_data()

        self.data, self.data_mean, self.data_std, self.data_raw = self.preprocess_data(self.params, self.data_raw)

        self.magic_transformation = get(self.params, "magic_transformation", False)
        if self.magic_transformation:
            R_minus = get(self.params, "R_minus", 0.2)
            R_plus = get(self.params, "R_plus", 1.5)
            deltaR12 = delta_r(self.data_raw, idx_phi1=9, idx_eta1=10, idx_phi2=13, idx_eta2=14)
            deltaR13 = delta_r(self.data_raw, idx_phi1=9, idx_eta1=10, idx_phi2=17, idx_eta2=18)
            deltaR23 = delta_r(self.data_raw, idx_phi1=13, idx_eta1=14, idx_phi2=17, idx_eta2=18)
            self.event_weights = magic_trafo(deltaR12, R_minus=R_minus, R_plus=R_plus)\
                                 *magic_trafo(deltaR13, R_minus=R_minus, R_plus=R_plus)\
                                 *magic_trafo(deltaR23, R_minus=R_minus, R_plus=R_plus)
            self.data = torch.cat([self.data, torch.from_numpy(self.event_weights[:, None]).to(self.device)], dim=1).float()
            print(f"preprocess_data: Using magic transformation")

        print(f"preprocess_data: input shape is {self.data.shape}")
        self.n_data = len(self.data)

        self.model = self.build_model(self.params)

        self.model.data_mean, self.model.data_std = self.data_mean, self.data_std

        self.model.obs_names = self.obs_names
        self.model.obs_ranges = self.obs_ranges
        self.build_optimizer()
        self.build_dataloaders()
        self.train_model()
        self.generate_samples()
        self.make_plots()
        self.finish_up()