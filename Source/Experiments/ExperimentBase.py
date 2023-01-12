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
import time
from datetime import datetime
import sys
import os
import h5py
import pandas
from torch.optim import Adam


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

        # Ranges of the observables for plotting
        self.obs_ranges = [[0.5, 150], [-4, 4], [-6, 6], [0, 100],
                           [0.5, 150], [-4, 4], [-6, 6], [0, 100],
                           [0.5, 150], [-4, 4], [-6, 6], [0, 100],
                           [0.5, 150], [-4, 4], [-6, 6], [0, 100],
                           [0.5, 150], [-4, 4], [-6, 6], [0, 100]]
        self.params = params
        self.conditional = get(self.params, "conditional", False)
        self.warm_start = get(self.params, "warm_start", False)
        self.warm_start_path = get(self.params, "warm_start_path", None)
        self.device = get(self.params, "device", get_device())
        self.batch_size = get(self.params, "batch_size", 1024)
        self.data_split = get(self.params, "data_split", [0.55, 0.05, 0.4])
        self.runs = get(self.params, "runs", 0)
        self.total_epochs = get(self.params, "total_epochs", 0)
        self.con_depth = get(self.params, "con_depth", 0)
        self.model = None
        self.prior_model= None
        self.prior_prior_model = None
        self.data = None
        self.data_raw = None
        self.data_mean = 0
        self.data_std = 0
        self.data_u = 0
        self.data_s = 0
        self.n_data = 0

        self.starttime = time.time()

    def prepare_experiment(self):
        """
                The prepare_experiment method gets the necessary parameters and sets up an out_dir directory for the experiment.
                All results will be saved in this directory.
                """
        # Read in the "warm_start" parameter to establish weather or not we start from a pretrained model
        # If we start from a pretrained model, we set the "warm_start_path" as the out_dir

        if self.warm_start:

            assert self.warm_start_path is not None, \
                f"prepare_experiment: warm_start set to True, but warm_start_path not specified"
            assert os.path.exists(self.warm_start_path), \
                f"prepare_experiment: warm_start set to True, but warm_start_path {self.warm_start_path} does not exist"
            self.out_dir = self.warm_start_path
            os.chdir(self.out_dir)
            print("prepare_experiment: Using warm_start_path as out_dir ", self.out_dir)

        # If we start fresh, we read in the "runs_dir" and "run_name" parameters and set up an out_dir
        # All out_dir names get a random number added to avoid unintentionally overwriting old experiments
        else:
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

        print("prepare_experiment: Using out_dir ", self.out_dir)
        print(f"prepare_experiment: Using device {self.device}")

    def load_data(self, p):
        """
        The load_data method gets the necessary parameters and reads in the data
        Currently supported are datasets of type *.npy ; *.torch ; *.h5

        Overwrite this method if other ways of reading in data are needed
        This method should place the data under self.data_raw
                """
        load_dataset = get(p, "load_dataset", True)
        # Read in the "data_path" parameter. Raise and error if it is not specified or does not exist
        data_path = get(p, "data_path", None)
        if data_path is None:
            raise ValueError("load_data: data_path is None. Please specify in params")
        assert os.path.exists(data_path), f"load_data: data_path {data_path} does not exist"

        prior_data_raw = None
        prior_prior_data_raw = None
        data_raw = None

        if load_dataset:
            # Read in the "data_type" parameter, defaulting to np if not specified. Try to read in the data accordingly
            data_type = get(p, "data_type", "np")
            if data_type == "np":
                data_raw = np.load(data_path)
                print(f"load_data: Loaded data with shape {data_raw.shape} from ", data_path)
            elif data_type == "torch":
                data_raw = torch.load(data_path)
                print(f"load_data: Loaded data with shape {data_raw.shape} from ", data_path)
            elif data_type == "h5":
                data_path_internal = get(p, "data_path_internal", None)
                if data_path_internal is not None:
                    with h5py.File(data_path, "r") as f:
                        data_raw = f[data_path_internal][:]
                else:
                    try:
                        data_raw = pandas.read_hdf(data_path).values
                    except Exception as e:
                        raise ValueError("load_data: Failed to read h5 file in data_path")
            else:
                raise ValueError(f"load_data: Cannot load data from {data_path}")
        else:
            data_all = Dataset(data_path, conditional=self.conditional)
            n_jets = get(p, "n_jets", 2)
            if n_jets == 1:
                data_raw = data_all.z_1
                prior_data_raw = None
            elif n_jets == 2:
                data_raw = data_all.z_2
                prior_data_raw = data_all.z_1
            elif n_jets == 3:
                data_raw = data_all.z_3
                prior_data_raw = data_all.z_2
                prior_prior_data_raw = data_all.z_1

        return data_raw, prior_data_raw, prior_prior_data_raw

    def preprocess_data(self, p, data_raw, save_in_params=False, conditional=False):
        """
        The preprocess_data method gets the necessary parameters and preprocesses the data
        Currently preprocessing is only implemented for Z2 jet data.

        Overwrite this method if other preprocessing is required
        This method should include moving the data to device.
        The final result should be placed under self.data
        The dimensionality of the data should be placed under self.dim
        """

        # Read in the "preprocess" parameter to specify weather or not the data should be preprocessed
        # Read in the "channels" and "dim" parameters specifiying which channels of the data should be used
        # If "channels" is specified, "dim" will be ignored and will be inferred from channels
        # If "channels" is not specified, only "dim" 4,6 and None are valid
        preprocess_data = get(p, "preprocess", True)
        channels = get(p, "channels", None)
        n_jets = get(p, "n_jets", 2)
        fraction = get(p, "fraction", None)
        n_con = get(p, "n_con", 0)
        if channels is None:
            print(f"preprocess_data: channels and dim not specified. Defaulting to {5 + 4 * n_jets} channels.")
            channels = np.array([i for i in range(n_jets * 4 + 8) if i not in [1, 3, 7]])
        else:
            print(f"preprocess_data: channels {channels} specified.")
        # Do the preprocessing
        # Currently using already preprocessed data is not implemented
        if not preprocess_data:
            print("preprocess_data: preprocess set to False")
            data = data_raw[:, channels]
            raise ValueError("preprocess_data: preprocess set to False. Not implemented properly")
        else:
            data, data_mean, data_std, data_u, data_s \
                = preprocess(data_raw, channels, fraction, conditional=conditional,
                             n_jets=n_jets)

            print("preprocess_data: Finished preprocessing")

        n_data = len(data)
        data_raw = undo_preprocessing(data, data_mean, data_std, data_u, data_s, channels, keep_all=True,
                                      conditional=conditional, n_jets=n_jets)

        # Make sure the data is a torch.Tensor and move it to device
        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(data)
        data = data.to(self.device).float()
        print(f"preprocess_data: Moved data to {data.device}")

        if save_in_params:
            if get(p, "dim", None) is None:
                self.params["dim"] = len(channels)

            self.params["channels"] = list(channels)
            self.params["n_data"] = n_data
        return data, data_mean, data_std, data_u, data_s, data_raw

    def build_model(self,p, prior_path=None, save_in_params=False):
        """
        The build_model method gets the necessary parameters and defines and builds the model.
        Currently, supported models are INN, TBD and DDPM.
        Currently, all diffusion models are based on a Resnet.

        New model classes implemented in this framework must take a param file as input.
        See documentation of ModelBase class for more guidelines on how to implement new model classes.

        Overwrite this method if a completely different model structure is required
        This method should place the model under self.model
        """

        # Read in the model class and try to build the model. Raise an error if it is not specified
        model_type = get(p, "model", None)
        if model_type is None:
            raise ValueError("build_model: model not specified")
        # Read in the parameters required to build the model. Raise an error if one of them is not specified
        n_blocks = get(p, "n_blocks", None)
        assert n_blocks is not None, "build_model: n_blocks not specified"
        layers_per_block = get(p, "layers_per_block", None)
        assert layers_per_block is not None, "build_model: layers_per_block not specified"
        intermediate_dim = get(p, "intermediate_dim", None)
        assert intermediate_dim is not None, "build_model: intermediate_dim not specified"
        encode_t = get(p, "encode_t", False)
        print(f"build_model: Trying to build model {model_type} "
              f"with n_blocks {n_blocks}, layers_per_block {layers_per_block}, "
              f"intermediate dim {intermediate_dim} and encode_t {encode_t}")
        if model_type == "INN":
            model = INN(p)
        elif model_type == "TBD":
            model = TBD(p)
        elif model_type == "DDPM":
            model = DDPM(p)
        else:
            raise ValueError(f"build_model: model class {model_type} not recognised. Use INN, TBD or DDPM")

        # Keep track of the total number of trainable model parameters
        model_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"build_model: Built model {model_type}. Total number of parameters: {model_parameters}")

        # If "warm_start", load the model parameters from the specified directory.
        # It is expected that they are found under warm_start_dir/models/checkpoint
        if prior_path is not None:
            try:
                state_dict = torch.load(prior_path + "/models/model_run0.pt", map_location=self.device)
            except FileNotFoundError:
                raise ValueError(f"build_model: cannot load model for prior_path")

            model.load_state_dict(state_dict)
            print(f"build_model: Loaded state_dict from prior_path {prior_path}")

        if save_in_params:
            self.params["model_parameters"] = model_parameters

            if get(self.params, "sample_periodically", False):
                # The Model needs these values to make intermediate plots
                # TODO: Can we make this nicer?
                self.intermediate_samples = model.samples
        return model

    def build_optimizer(self,p, parameters):
        """
        The build_optimizer method gets the necessary parameters and builds the training optimizer.
        Currently only vanilla Adam is implemented.
        TODO: Implement SGD
        TODO: Implement LR scheduling

        Overwrite or extend this method if a different optimizer is needed.
        The method should place the optimizer under self.model.optimizer
        """

        # Read in the "train" parameter. If it is set to True, build the optimizer, otherwise skip it.
        train = get(p, "train", True)
        if train:
            # Read in the "optimizer" parameter and build the specified optimizer
            # TODO: Not very nice
            optim = get(p, "optimizer", None)
            if optim is None:
                optim = "Adam"
                print(f"build_optimizer: optimizer not specified. Defaulting to {optim}")

            if optim == "Adam":
                lr = get(p, "lr", 0.0001)
                betas = get(p, "betas", [0.9, 0.999])
                weight_decay = get(p, "weight_decay", 0)
                optimizer = \
                    Adam(parameters, lr=lr, betas=betas, weight_decay=weight_decay)
                print(
                    f"build_optimizer: Built optimizer {optim} with lr {lr}, betas {betas}, weight_decay {weight_decay}")
            else:
                raise ValueError(f"build_optimizer: optimizer {optim} not implemented")

        else:
            optimizer = None
            print("build_optimizer: train set to False. Not building optimizer")

        return optimizer

    def build_dataloaders(self,p):
        """
        The build_dataloaders methods gets the necessary parameters and builds the train_loader, val_loader and test_loader
        TODO: test_loader used for nothing atm

        Overwrite this method if necessary.
        This method should place the loaders under
        self.model.train_loader, self.model.val_loader and self.model.test_loader respectively
        """

        # Read in the "train" parameter. If it is set to True, build the dataloaders, otherwise skip it.
        train = get(p, "train", True)
        n_data = get(p, "n_data", 1000000)
        # Read in the "data_split" parameter, specifying which parts of the data to use for training, validation and test

        if train:
            # Read in the "batch_size" parameter and calculate the cuts between traindata, valdata and testdata
            # Define the loaders

            cut1 = int(n_data * self.data_split[0])
            cut2 = int(n_data * (self.data_split[0] + self.data_split[1]))
            self.model.train_loader = \
                DataLoader(dataset=self.data[:cut1],
                           batch_size=self.batch_size,
                           shuffle=True)
            self.model.val_loader = \
                DataLoader(dataset=self.data[cut1: cut2],
                           batch_size=self.batch_size,
                           shuffle=True)
            self.model.test_loader = \
                DataLoader(dataset=self.data[cut2:],
                           batch_size=self.batch_size,
                           shuffle=True)
            print(
                f"build_dataloaders: Built dataloaders with data_split {self.data_split} and batch_size {self.batch_size}")
        else:
            print("build_dataloaders: train set to False. Not building dataloaders")

    def train_model(self, p, save_in_params=True):
        """
        The train_model method performs the model training.
        Currently the training code is hidden as part of the model classes to keep the ExperimentClass shorter.
        All models have a run_training() method, that performs the training.

        New model classes implemented in this framework must have a run_training() method.
        See documentation of ModelBase class for more guidelines on how to implement new model classes.

        TODO: Implement sample_every
        TODO: Implement force_checkpoint_every

        Overwrite this method if a different way of performing the training is needed.
        """

        # Read in the "train" parameter. If it is set to True, perform the training, otherwise skip it.
        train = get(p, "train", True)
        if train:
            # Create a folder to save model checkpoints
            os.makedirs("models", exist_ok=True)
            # Keep track of the time and perform the model training
            # See the model classes for documentation on the run_training() method
            t0 = time.time()
            sample_every_n_samples = get(p, "sample_every_n_samples", 100000)
            self.model.run_training(prior_model=self.prior_model, prior_prior_model=self.prior_prior_model)
            t1 = time.time()
            traintime = t1 - t0
            n_epochs = get(p,"n_epochs",100)
            print(f"train_model: Finished training {n_epochs} epochs after {traintime} seconds.")

            # Save the final model. Update the total amount of training epochs the model has been trained for
            torch.save(self.model.state_dict(), f"models/model_run{self.runs}.pt")

            if save_in_params:
                self.params["total_epochs"] = self.total_epochs
                self.params["traintime"] = traintime
                self.total_epochs = self.total_epochs + n_epochs

        else:
            print("train_model: train set to False. Not training")
        print(f"train_model: Model has been trained for a total of {self.total_epochs} epochs")


    def generate_samples(self):
        pass

    def make_plots(self):
        pass

    def finish_up(self):
        """
        The finish_up method just writes some additional information into the parameters and saves them in the out_dir
        """
        self.params["runs"] = self.runs + 1
        self.params["datetime"] = str(datetime.now())
        self.params["experimenttime"] = time.time() - self.starttime
        save_params(self.params, "paramfile.yaml")
        print("finish_up: Finished experiment.")

        if get(self.params, "redirect_console", True):
            sys.stdout.close()
            sys.stderr.close()

    def full_run(self):
        pass
