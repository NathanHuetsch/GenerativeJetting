import numpy as np
import torch
from Source.Util.datasets import Dataset
from Source.Util.util import get_device, save_params, get, load_params
from Source.Util.discretize import discretize, undo_discretize
from Source.Experiments.ExperimentBase import Experiment
import time
from datetime import datetime
import os, sys
from Source.Util.simulateToyData import ToySimulator

class Toy_Experiment(Experiment):

    def __init__(self, params):

        super().__init__(params)

        self.params["istoy"] = True
        self.n_data = get(self.params, "n_data", 1000000)

    def full_run(self):
        self.prepare_experiment()
        self.load_data()
        self.preprocess_data()

        if get(self.params, "toy_type", "ramp")=="ramp":
            self.n_dim = get(self.params, "n_flat", 1)+get(self.params, "n_lin", 1)+get(self.params, "n_quad", 0)
            if get(self.params, "obs_ranges",None) is None:
                self.obs_ranges = [[-0.1, 1.1]] * self.dim
            else:
                self.obs_ranges = self.params['obs_ranges'] * self.dim
        else:
            self.n_dim = get(self.params, "n_dim", 2)
            self.obs_ranges = [[-1.5, 1.5]] * self.dim
        self.obs_names = ["x_{"+str(i)+"}" for i in range(self.n_dim)]

        if self.iterations > 1 and not self.bayesian:
            det_samples = []
            for i in range(self.iterations):
                self.total_epochs = 0
                self.model = self.build_model(self.params, save_in_params=True)
                print(f"build_model: Building Bayesian model is set to {self.bayesian}")

                self.model.obs_names = self.obs_names
                self.model.obs_ranges = self.obs_ranges
                self.model.data = self.data
                self.model.bin_edges, self.model.bin_means = self.bin_edges, self.bin_means

                self.build_optimizer()
                self.build_dataloaders()

                self.train_model()
                self.generate_samples()
                det_samples.append(self.samples)
                self.runs += 1
                self.model.runs += 1

            self.samples = np.concatenate(det_samples)
            self.make_plots()
        else:
            self.model = self.build_model(self.params, prior_path=self.warm_start_path, save_in_params=True)
            print(f"build_model: Building Bayesian model is set to {self.bayesian}")
            self.model.obs_names = self.obs_names
            self.model.obs_ranges = self.obs_ranges
            self.model.data = self.data
            self.model.bin_edges, self.model.bin_means = self.bin_edges, self.bin_means

            self.build_optimizer()
            self.build_dataloaders()

            self.train_model()
            self.generate_samples()
            self.make_plots()

        self.make_video()
        self.finish_up()

    def load_data(self):
        load_dataset = get(self.params, "load_dataset", True)
        # Read in the "data_path" parameter. Raise and error if it is not specified or does not exist
        data_path = get(self.params, "data_path", None)
        if load_dataset:
            # Read in the "data_type" parameter, defaulting to np if not specified. Try to read in the data accordingly
            data_type = get(self.params, "data_type", "np")
            if data_type == "np":
                self.data_raw = np.load(data_path)
                self.dim = self.data_raw.shape[1]
                print(f"load_data: Loaded data with shape {self.data_raw.shape} from ", data_path)
            else:
                raise ValueError(f"load_data: Cannot load data from {data_path}")
        else:
            self.data_raw = ToySimulator(self.params).data
            self.dim = self.data_raw.shape[1]
            print(f"load_data: Simulated data with shape {self.data_raw.shape} following a "
                  f"{self.dim}-dimensional {get(self.params, 'toy_type', 'ramp')} distribution")
        self.params["dim"] = self.dim

    def preprocess_data(self):
        if self.params["model"] == "AutoRegBinned":
            self.data, self.bin_edges, self.bin_means = discretize(self.data_raw.copy(), self.params)
        else:
            self.data = self.data_raw
            self.bin_edges, self.bin_means = None, None

        if not isinstance(self.data_raw, torch.Tensor):
            self.data = torch.from_numpy(self.data)
        if self.params["model"] == "AutoRegBinned":
            self.data = self.data.long()
        else:
            self.data = self.data.float()
        self.data = self.data.to(self.device)
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
            bay_samples = []
            for i in range(0, iterations):
                # Read in the "n_samples" parameter specifying how many samples to generate
                # Call the model.sample_n_parallel(n_samples) method to perform the sampling
                n_samples = get(self.params, "n_samples", 1000000)
                print(f"generate_samples: {i}.Starting generation of {n_samples} samples")
                t0 = time.time()
                sample = self.model.sample_n(n_samples)
                t1 = time.time()
                sampletime = t1 - t0
                self.params["sampletime"] = sampletime
                bay_samples.append(sample)

                print(f"generate_samples: {i}.Finished generation of {n_samples} samples after {sampletime} seconds")
                if get(self.params, "save_samples", False):
                    os.makedirs('samples', exist_ok=True)
                    np.save(f"samples/samples_final_{i}.npy", sample)
                    print(f"save_samples: generated samples have been saved")

            self.samples = np.concatenate(bay_samples)
        else:
            print("generate_samples: sample set to False")

    def make_plots(self):
        plot = get(self.params, "plot", True)
        sample = get(self.params, "sample", True)

        if plot and sample:
            self.model.plot_toy(self.samples, finished=True)

            print("make_plots: Finished making plots")
        else:
            print("make_plots: plot set to False")

    def make_video(self):
        video = get(self.params, "video", False)

        if video:
            n_samples = get(self.params, "n_samples", 1000000)
            print(f"make_video: Starting generation of video frame samples")
            t0 = time.time()
            sample = self.model.sample_n_evolution(n_samples)
            t1 = time.time()
            videosampletime = t1 - t0
            self.params["videosampletime"] = videosampletime
            print(f"make_video: Drawing videos")
            t0 = time.time()
            self.model.toy_video(sample)
            t1 = time.time()
            videotime = t1 - t0
            self.params["videotime"] = videotime
