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
from Source.Util.simulateToyData import ToySimulator

class Toy_Experiment(Experiment):

    def __init__(self, params):

        super().__init__(params)

        self.istoy = get(self.params,"istoy", True)
        self.params['istoy'] = self.istoy
        self.n_data = get(self.params, "n_data", 1000000)
        self.iterations = get(self.params, "iterations", 1)
        self.bayesian = get(self.params, "bayesian",False)
        if get(params, "toy_type", "ramp")=="ramp":
            self.n_dim = get(params, "n_flat", 1)+get(params, "n_lin", 1)+get(params, "n_quad", 0)
        else:
            self.n_dim = get(params, "n_dim", 2)

        self.samples = []



    def full_run(self):
        self.prepare_experiment()
        self.load_data()
        self.obs_names = [f"x_{i}" for i in range(self.n_dim)]
        #self.obs_range = get(self.params, "obs_ranges", [0, 1])
        self.obs_ranges = None #[self.obs_range] * self.dim
        self.data_raw = self.data.detach().cpu().numpy()
        self.model = self.build_model(self.params, save_in_params=True)

        self.model.obs_names = self.obs_names
        self.model.obs_ranges = self.obs_ranges
        self.model.data = self.data

        self.build_optimizer()
        self.build_dataloaders()
        if self.iterations > 1 and not self.bayesian:
            for i in range(self.iterations):
                self.train_model()
                self.generate_samples()
                self.make_plots()
                self.runs += 1

        else:
            self.train_model()
            self.generate_samples()
            self.make_plots()

        self.finish_up()

    def load_data(self):
        load_dataset = get(self.params, "load_dataset", True)
        # Read in the "data_path" parameter. Raise and error if it is not specified or does not exist
        data_path = get(self.params, "data_path", None)
        if load_dataset:
            # Read in the "data_type" parameter, defaulting to np if not specified. Try to read in the data accordingly
            data_type = get(self.params, "data_type", "np")
            if data_type == "np":
                self.data = np.load(data_path)
                print(f"load_data: Loaded data with shape {self.data.shape} from ", data_path)
            else:
                raise ValueError(f"load_data: Cannot load data from {data_path}")
        else:
            self.data = ToySimulator(self.params).data

        self.dim = self.data.shape[1]
        self.params["dim"] = self.dim
        print(f"load_data: Simulated data with shape {self.data.shape} following a "
              f"{self.dim}-dimensional {get(self.params, 'toy_type', 'ramp')} distribution")

        if not isinstance(self.data, torch.Tensor):
            self.data = torch.from_numpy(self.data)
        self.data = self.data.to(self.device).float()
        print(f"load_data: Moved data to {self.data.device}")

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

        if self.bayesian:
            iterations = self.iterations
        else:
            iterations = 1

        if sample:
            for i in range(0, iterations):
                # Read in the "n_samples" parameter specifying how many samples to generate
                # Call the model.sample_n_parallel(n_samples) method to perform the sampling
                n_samples = get(self.params, "n_samples", 1000000)
                print(f"generate_samples: Starting generation of {n_samples} samples")
                t0 = time.time()
                self.sample = self.model.sample_n(n_samples)
                t1 = time.time()
                sampletime = t1 - t0
                self.params["sampletime"] = sampletime
                self.samples.append(self.sample)

                print(f"generate_samples: Finished generation of {n_samples} samples after {sampletime} seconds")
                if get(self.params, "save_samples", False):
                    os.makedirs('samples', exist_ok=True)
                    np.save(f"samples/samples_final_{i}.npy", self.samples)
                    print(f"save_samples: generated samples have been saved")
        else:
            print("generate_samples: sample set to False")

    def make_plots(self):
        plot = get(self.params, "plot", True)
        sample = get(self.params, "sample", True)

        if self.iterations>1:
            to_plot = self.samples
        else:
            to_plot = self.sample


        if plot and sample:
            self.model.plot_toy(to_plot, finished=True)

            print("make_plots: Finished making plots")
        else:
            print("make_plots: plot set to False")
