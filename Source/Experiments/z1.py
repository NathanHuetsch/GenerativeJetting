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

        self.n_jets = 1
        self.con_depth = get(self.params, "con_depth", 0)
        self.channels = get(self.params, "channels", None)
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
        self.data_raw = self.z_1

        self.data, self.data_mean, self.data_std, self.data_u, self.data_s, self.data_raw = \
            self.preprocess_data(self.params, self.data_raw, save_in_params=True, conditional=self.conditional)

        print(f"preprocess_data: input shape is {self.data.shape}")
        self.n_data = len(self.data)

        if self.warm_start:
            self.model = self.build_model(self.params, prior_path=self.warm_start_path, save_in_params=True)
        else:
            self.model = self.build_model(self.params, save_in_params=True)

        self.model.data_mean, self.model.data_std, self.model.data_u, self.model.data_s \
            = self.data_mean, self.data_std, self.data_u, self.data_s
        self.model.obs_names = self.obs_names
        self.model.obs_ranges = self.obs_ranges
        self.build_optimizer()
        self.build_dataloaders()
        self.train_model()
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

            # Read in the "n_samples" parameter specifying how many samples to generate
            # Call the model.sample_n_parallel(n_samples) method to perform the sampling
            n_samples = get(self.params, "n_samples", 1000000)
            print(f"generate_samples: Starting generation of {n_samples} samples")
            t0 = time.time()
            self.samples = self.model.sample_and_undo(n_samples, n_jets=self.n_jets)
            t1 = time.time()
            sampletime = t1 - t0
            self.params["sampletime"] = sampletime
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
            self.model.plot_samples(self.samples, finished=True)