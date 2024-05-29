import numpy as np
import torch
from torch.utils.data import DataLoader
from Source.Models.cfm import CFM
from matplotlib.backends.backend_pdf import PdfPages
from Source.Util.plots import plot_obs, delta_r, plot_deta_dphi
from torch.optim.lr_scheduler import CosineAnnealingLR
from Source.Util.preprocessing import preformat, preprocess, undo_preprocessing
from Source.Util.util import get_device, save_params, get, load_params, magic_trafo
import time
from datetime import datetime
import sys
import os
import h5py
import pandas
from torch.optim import Adam, AdamW, RAdam


class Experiment:
    """
    Base Class for generative modelling experiment classes to inherit from.
    Children classes should overwrite the individual methods as needed.
    Depending on the details, some methods might not be needed, e.g. if the dataset is already preprocessed
    or if we just want to generate samples and plots without retraining a model.

    See z2.py for an example of a fully implemented ExperimentClass

    Structure:

    __init__(params)      : Read in parameters
    prepare_experiment()  : Create out_dir and cwd into chdir into it
    load_data()           : Read in the dataset
    preprocess_data()     : Preprocess the data and move it to device
    build_model()         : Build the model and define its architecture
    build_optimizer()     : Build the optimizer and define its parameters
    build_dataloaders()   : Build the dataloaders for model training
    train_model()         : Run the model training
    generate_samples()    : Use the model to generate samples
    make_plots()          : Make plots to compare the generated samples to the test set data
    finish_up()           : Finish the experiment and save some information

    full_run()            : Call all of the above methods to perform a full experiment
    """

    def __init__(self, params):
        # Names of the observables for plotting
        self.obs_names = ["p_{T,l1}", "\phi_{l1}", "\eta_{l1}", "\mu_{l1}",
                          "p_{T,l2}", "\phi_{l2}", "\eta_{l2}", "\mu_{l2}",
                          "p_{T,j1}", "\phi_{j1}", "\eta_{j1}", "\mu_{j1}",
                          "p_{T,j2}", "\phi_{j2}", "\eta_{j2}", "\mu_{j2}",
                          "p_{T,j3}", "\phi_{j3}", "\eta_{j3}", "\mu_{j3}"]

        self.obs_units = ["GeV", None, None, "GeV",
                          "GeV", None, None, "GeV",
                          "GeV", None, None, "GeV",
                          "GeV", None, None, "GeV",
                          "GeV", None, None, "GeV"]

        # Ranges of the observables for plotting
        self.obs_ranges = [[0.5, 150], [-4, 4], [-6, 6], [0, 50],
                           [0.5, 150], [-4, 4], [-6, 6], [0, 50],
                           [17,  157], [-4, 4], [-6, 6], [0, 50],
                           [17,  82], [-4, 4], [-6, 6], [0, 50],
                           [17,  82], [-4, 4], [-6, 6], [0, 50]]

        self.params = params
        self.conditional = get(self.params, "conditional", False)
        if not self.conditional:
            self.params["n_con"]=0
        self.con_depth = get(self.params, "con_depth", 0)

        self.device = get(self.params, "device", get_device())
        self.batch_size = get(self.params, "batch_size", 1024)
        self.data_split = get(self.params, "data_split", [0.6, 0, 0.4])

        self.load_dataset = get(self.params, "load_dataset", False)
        self.bayesian = get(self.params, "bayesian", False)

        self.n_jets = get(self.params, "n_jets", 2)
        self.starttime = time.time()

    def prepare_experiment(self):
        """
                The prepare_experiment method gets the necessary parameters and sets up an out_dir directory for the experiment.
                All results will be saved in this directory.
                """
        # If we start fresh, we read in the "runs_dir" and "run_name" parameters and set up an out_dir
        # All out_dir names get a random number added to avoid unintentionally overwriting old experiments
        runs_dir = get(self.params, "runs_dir", None)
        if runs_dir is None:
            runs_dir = os.path.join(os.getcwd(), "runs")
            print("prepare_experiment: runs_dir not specified. Working in ", runs_dir)
        run_name = get(self.params, "run_name", None)
        rnd_number = np.random.randint(low=1000, high=9999)
        if run_name is None:
            self.out_dir = os.path.join(runs_dir, str(rnd_number))
            print("prepare_experiment: run_name not specified. Using random number")
        else:
            self.out_dir = os.path.join(runs_dir, run_name + str(rnd_number))
        os.makedirs(self.out_dir)
        os.chdir(self.out_dir)
        self.params["out_dir"] = self.out_dir
        save_params(self.params, "paramfile.yaml")

        # The "redirect_console" parameter controls weather or not we redirect console outputs and errors into text files
        # This is usefull when working on the cluster
        if get(self.params, "redirect_console", True):
            sys.stdout = open("stdout.txt", "w", buffering=1)
            sys.stderr = open("stderr.txt", "w", buffering=1)
            print(f"prepare_experiment: Redirecting console output to out_dir")

        print(f"prepare_experiment: Using out_dir {self.out_dir}")
        print(f"prepare_experiment: Using device {self.device}")

    def load_data(self):
        """
        The load_data method gets the necessary parameters and reads in the data
        Currently supported are datasets of type *.npy ; *.torch ; *.h5

        Overwrite this method if other ways of reading in data are needed
        This method should place the data under self.data_raw
        """
        # Read in the "data_path" parameter. Raise and error if it is not specified or does not exist
        data_path = get(self.params, "data_path", None)
        if data_path is None:
            raise ValueError("load_data: data_path is None. Please specify in params")
        assert os.path.exists(data_path), f"load_data: data_path {data_path} does not exist"

        self.data_raw = np.load(data_path, allow_pickle=True)
        print(f"load_data: Loaded data with shape {self.data_raw.shape} from {data_path}")

    def preprocess_data(self, p, data_raw):
        """
        The preprocess_data method gets the necessary parameters and preprocesses the data
        """

        self.channels = get(self.params, "channels", None)
        if self.channels is None:
            print(f"preprocess_data: channels and dim not specified. Using all")
            self.channels = np.array([i for i in range(self.n_jets * 4 + 8) if i not in [1, 3, 7]]).tolist()
            self.params["channels"] = self.channels

            print(f"preprocess_data: channels {self.channels}")
        else:

            print(f"preprocess_data: channels {self.channels} specified.")
        self.params["dim_x"] = len(self.channels)
        self.plot_channels = self.channels
        self.params["plot_channels"] = self.plot_channels
        # Do the preprocessing
        data_raw = preformat(data_raw)

        data, data_mean, data_std = preprocess(data_raw, p)
        print("preprocess_data: Finished preprocessing")

        # Make sure the data is a torch.Tensor and move it to device
        data = torch.from_numpy(data).float().to(self.device)
        print(f"preprocess_data: Moved data to {data.device}")

        return data, data_mean, data_std, data_raw

    def build_model(self,p):
        """
        The build_model method gets the necessary parameters and defines and builds the model.
        """

        # Read in the model class and try to build the model. Raise an error if it is not specified
        model_type = get(p, "model", None)
        if model_type is None:
            raise ValueError("build_model: model not specified")
        model = eval(model_type)(p)

        # Keep track of the total number of trainable model parameters
        model_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.params["model_parameters"] = model_parameters
        print(f"build_model: Built model {model_type}. Total number of parameters: {model_parameters}")

        return model

    def build_optimizer(self):
        """
        The build_optimizer method gets the necessary parameters and builds the training optimizer.
        """

        # Read in the "train" parameter. If it is set to True, build the optimizer, otherwise skip it.
        train = get(self.params, "train", True)
        if train:
            optim = get(self.params, "optimizer", None)
            if optim is None:
                optim = "Adam"
                print(f"build_optimizer: optimizer not specified. Defaulting to {optim}")

            lr = get(self.params, "lr", 0.0001)
            betas = get(self.params, "betas", [0.9, 0.999])
            weight_decay = get(self.params, "weight_decay", 0)
            if optim == "Adam":
                self.model.optimizer = \
                    Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
            elif optim == "AdamW":
                self.model.optimizer = \
                    AdamW(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
            elif optim == "RAdam":
                self.model.optimizer = \
                    RAdam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
            else:
                raise ValueError(f"build_optimizer: optimizer {optim} not implemented")
            print(
                f"build_optimizer: Built optimizer {optim} with lr {lr}, betas {betas}, weight_decay {weight_decay}")
        else:
            self.model.optimizer = None
            print("build_optimizer: train set to False. Not building optimizer")

    def build_dataloaders(self):
        """
        The build_dataloaders methods gets the necessary parameters and builds the train_loader, val_loader and test_loader
        TODO: test_loader used for nothing atm
        """

        # Read in the "train" parameter. If it is set to True, build the dataloaders, otherwise skip it.
        train = get(self.params, "train", True)
        n_data = get(self.params, "n_data", 1000000)
        # Read in the "data_split" parameter, specifying which parts of the data to use for training, validation and test
        cut1 = int(n_data * self.data_split[0])
        cut2 = int(self.n_data * (self.data_split[0] + self.data_split[1]))
        self.model.data_train = self.data_raw[:cut1]
        self.model.data_test = self.data_raw[cut2:]


        if train:
            # Read in the "batch_size" parameter and calculate the cuts between traindata, valdata and testdata
            # Define the loaders


            self.model.train_loader = \
                DataLoader(dataset=self.data[:cut1],
                           batch_size=self.batch_size,
                           shuffle=True)
            self.model.test_loader = \
                DataLoader(dataset=self.data[cut2:],
                           batch_size=self.batch_size,
                           shuffle=True)

            self.sample_periodically = get(self.params, "sample_periodically", False)
            print(
                f"build_dataloaders: Built dataloaders with data_split {self.data_split} and batch_size {self.batch_size}")

            use_scheduler = get(self.params, "use_scheduler", False)
            if use_scheduler:
                lr_scheduler = get(self.params, "lr_scheduler", "OneCycle")
                if lr_scheduler == "CosineAnnealing":
                    n_epochs = get(self.params, "n_epochs", 100)
                    eta_min = get(self.params, "eta_min", 0)
                    self.model.scheduler = CosineAnnealingLR(
                        optimizer=self.model.optimizer,
                        T_max=n_epochs*len(self.model.train_loader),
                        eta_min=eta_min
                    )
                    print(f"build_dataloaders: Using CosineAnnealing lr scheduler with eta_min {eta_min}")
                else:
                    print(f"build_dataloaders: lr_scheduler {lr_scheduler} not recognised. Not using it")
                    self.params["use_scheduler"]=False
        else:
            print("build_dataloaders: train set to False. Not building dataloaders")

    def train_model(self):
        """
        The train_model method performs the model training.
        Currently the training code is hidden as part of the model classes to keep the ExperimentClass shorter.
        """

        # Read in the "train" parameter. If it is set to True, perform the training, otherwise skip it.
        train = get(self.params, "train", True)
        if train:
            # Create a folder to save model checkpoints
            os.makedirs("models", exist_ok=True)
            # Keep track of the time and perform the model training
            # See the model classes for documentation on the run_training() method
            t0 = time.time()
            self.model.run_training()
            t1 = time.time()
            traintime = t1 - t0
            n_epochs = get(self.params,"n_epochs",100)
            print(f"train_model: Finished training {n_epochs} epochs after {traintime:.2f} s = {traintime/60:.2f} min = {traintime/60**2:.2f} h.")

            # Save the final model. Update the total amount of training epochs the model has been trained for
            torch.save(self.model.state_dict(), f"models/model.pt")

            self.params["traintime"] = traintime
            print(f"train_model: Model has been trained for {n_epochs} epochs")
        else:
            print("train_model: train set to False. Not training")


    def generate_samples(self):
        """
        The generate_samples method uses the trained or loaded model to generate samples.
        Currently, the sampling code is hidden as part of the model classes to keep the ExperimentClass shorter.
        """

        # Read in the "sample" parameter. If it is set to True, perform the sampling, otherwise skip it.
        sample = get(self.params, "sample", True)
        iterations = self.iterations if self.bayesian else 1

        if sample:
            bay_samples = []
            for i in range(0, iterations):
                # Read in the "n_samples" parameter specifying how many samples to generate
                # Call the model.sample_n_parallel(n_samples) method to perform the sampling
                n_samples = get(self.params, "n_samples", 1000000)
                print(f"generate_samples: Starting generation of {n_samples} samples")
                t0 = time.time()
                sample = self.model.sample_and_undo(n_samples)
                t1 = time.time()
                sampletime = t1 - t0
                self.params["sampletime"] = sampletime
                bay_samples.append(sample)

                print(f"generate_samples: Finished generation of {n_samples} samples after {sampletime:.2f} s = {sampletime/60:.2f} min.")
                if get(self.params, "save_samples", False):
                    os.makedirs('samples', exist_ok=True)
                    np.save(f"samples/samples_final_{i}.npy", sample)
                    print(f"save_samples: generated samples have been saved")

            self.samples = np.concatenate(bay_samples)
            #print(self.samples.mean(0), self.samples.std(0))
            #print(self.data_raw.mean(0), self.data_raw.std(0))
        else:
            print("generate_samples: sample set to False")

    def make_plots(self):
        """
        The make_plots method uses the train data, the test data and the generated samples to draw a range of plots.
        """

        # Read in the "plot" and "sample" parameters. If both are set to True, perform make the plots, otherwise skip it.
        plot = get(self.params, "plot", True)
        sample = get(self.params, "sample", True)
        if plot and sample:
            print(f"make_plots: plotting {self.plot_channels}")
            self.model.plot_samples(self.samples, finished=True)

            print("make_plots: Finished making plots")
        else:
            print("make_plots: plot set to False")

    def finish_up(self):
        """
        The finish_up method just writes some additional information into the parameters and saves them in the out_dir
        """
        self.params["datetime"] = str(datetime.now())
        self.params["experimenttime"] = time.time() - self.starttime
        save_params(self.params, "paramfile.yaml")
        print("finish_up: Finished experiment.")

        if get(self.params, "redirect_console", True):
            sys.stdout.close()
            sys.stderr.close()

    def full_run(self):
        pass
