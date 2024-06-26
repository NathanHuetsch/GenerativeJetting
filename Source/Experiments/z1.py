import numpy as np
import torch
from torch.utils.data import DataLoader
from Source.Models.cfm import CFM
from Source.Models.cm import CM

from matplotlib.backends.backend_pdf import PdfPages
from Source.Util.plots import plot_obs, delta_r, plot_deta_dphi
from Source.Util.preprocessing import preprocess, undo_preprocessing
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
    Class to run Z+1jet generative modelling experiments
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

        self.data_raw = undo_preprocessing(self.data.detach().cpu().numpy(), self.data_mean, self.data_std, self.params)

        print(f"preprocess_data: input shape is {self.data.shape}")
        self.n_data = len(self.data)

        self.model = self.build_model(self.params)

        # Load teacher model
        teacher_model_params = get(self.params, "teacher_model_params", None)


        self.teacher_model = None # solves problem with undefined variable fast. not optimal solution
        if  get(self.params, "model", "CM") == "CM": 
            self.teacher_model_params = load_params(teacher_model_params)
            self.teacher_model = self.load_model(self.teacher_model_params)

        self.model.data_mean, self.model.data_std = self.data_mean, self.data_std
        self.model.obs_names = self.obs_names
        self.model.obs_ranges = self.obs_ranges
        self.build_optimizer()
        self.build_dataloaders()
        self.train_model(self.teacher_model)
        self.generate_samples()
        self.make_plots()
        self.finish_up()
