import numpy as np
import torch
from torch.utils.data import DataLoader
from Source.Models.inn import INN
from Source.Models.tbd import TBD
from Source.Models.ddpm import DDPM
from Source.Models.gpt import GPTwrapper
from matplotlib.backends.backend_pdf import PdfPages
from Source.Util.plots import plot_obs, delta_r, plot_deta_dphi
from Source.Util.preprocessing import preprocess, undo_preprocessing
from Source.Util.util import get_device, save_params, get
from Source.Util.discretize import create_bins, discretize_wrapper
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
        self.params = params
        self.device = get(self.params, "device", get_device())

        # Read in the model class. Raise an error if it is not specified
        self.modelname = get(self.params, "model", None)
        if self.modelname is None:
            raise ValueError("build_model: model not specified")

        # Names of the observables for plotting
        self.obs_names = ["p_{T,l1}", "\phi_{l1}", "\eta_{l1}", "\mu_{l1}",
                          "p_{T,l2}", "\phi_{l2}", "\eta_{l2}", "\mu_{l2}",
                          "p_{T,j1}", "\phi_{j1}", "\eta_{j1}", "\mu_{j1}",
                          "p_{T,j2}", "\phi_{j2}", "\eta_{j2}", "\mu_{j2}"]

        # Ranges of the observables for plotting
        self.obs_ranges = [[0, 150], [-4, 4], [-6, 6], [0, 100],
                           [0, 150], [-4, 4], [-6, 6], [0, 100],
                           [0, 150], [-4, 4], [-6, 6], [0, 100],
                           [0, 150], [-4, 4], [-6, 6], [0, 100]]

        # Keep track of how many runs and epochs of training the model got before
        self.runs = get(self.params, "runs", 0)
        self.total_epochs = get(self.params, "total_epochs", 0)

        self.starttime = time.time()

    def prepare_experiment(self):
        """
        The prepare_experiment method gets the necessary parameters and sets up an out_dir directory for the experiment.
        All results will be saved in this directory.
        """
        # Read in the "warm_start" parameter to establish weather or not we start from a pretrained model
        # If we start from a pretrained model, we set the "warm_start_path" as the out_dir
        self.warm_start = get(self.params, "warm_start", False)
        if self.warm_start:
            self.warm_start_path = get(self.params, "warm_start_path", None)
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
            sys.stdout = open("stdout.txt", "w")
            sys.stderr = open("stderr.txt", "w")
            print(f"prepare_experiment: Redirecting console output to out_dir")

        print("prepare_experiment: Using out_dir ", self.out_dir)
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

        # Read in the "data_type" parameter, defaulting to np if not specified. Try to read in the data accordingly
        data_type = get(self.params, "data_type", "np")
        if data_type == "np":
            self.data_raw = np.load(data_path)
            print(f"load_data: Loaded data with shape {self.data_raw.shape} from ", data_path)
        elif data_type == "torch":
            self.data_raw = torch.load(data_path)
            print(f"load_data: Loaded data with shape {self.data_raw.shape} from ", data_path)
        elif data_type == "h5":
            data_path_internal = get(self.params, "data_path_internal", None)
            if data_path_internal is not None:
                with h5py.File(data_path, "r") as f:
                    self.data_raw = f[data_path_internal][:]
            else:
                try:
                    self.data_raw = pandas.read_hdf(data_path).values
                except Exception as e:
                    raise ValueError("load_data: Failed to read h5 file in data_path")
        else:
            raise ValueError(f"load_data: Cannot load data from {data_path}")

    def preprocess_data(self):
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
        self.preprocess = get(self.params, "preprocess", True)
        self.channels = get(self.params, "channels", None)
        self.dim = get(self.params, "dim", None)
        self.fraction = get(self.params, "fraction", None)
        if self.channels is None and self.dim is None:
            print("preprocess_data: channels and dim not specified. Defaulting to 13 channels")
            self.dim = 13
            self.channels = np.array([i for i in range(self.data_raw.shape[1]) if i not in [1, 3, 7]])
            self.params["dim"] = len(self.channels)
        elif self.channels is None and self.dim == 4:
            print("preprocess_data: channels not specified and dim=4. Using [9,10,13,14]")
            self.channels = [9, 10, 13, 14]
        elif self.channels is None and self.dim == 6:
            print("preprocess_data: channels not specified and dim=6. Using [8,9,10,12,13,14]")
            self.channels = [8, 9, 10, 12, 13, 14]
        elif self.channels is None:
            raise ValueError(f"preprocess: channels not specified and dim is not 4,6,None")
        else:
            self.params["dim"] = len(self.channels)
            print(f"preprocess_data: channels {self.channels} specified. Ignoring dim")
        self.params["channels"] = self.channels
        # Do the preprocessing
        # Currently using already preprocessed data is not implemented
        if not self.preprocess:
            print("preprocess_data: preprocess set to False")
            self.data = self.data_raw[:, self.channels]
            raise ValueError("preprocess_data: preprocess set to False. Not implemented properly")
        else:
            self.data, self.data_mean, self.data_std, self.data_u, self.data_s \
                = preprocess(self.data_raw, self.channels, self.fraction)
            print("preprocess_data: Finished preprocessing")

        print(f"preprocess_data: input shape is {self.data.shape}")
        self.n_data = len(self.data)
        self.data_raw = undo_preprocessing(self.data, self.data_mean, self.data_std, self.data_u, self.data_s,
                                           self.channels, keep_all=True)

        if not isinstance(self.data, torch.Tensor):
            self.data = torch.from_numpy(self.data)
        # do discretization for GPT-style models
        if self.modelname == "GPT":
            n_bins = self.params["n_bins"]
            assert n_bins is not None, "preprocess_data: n_bins not specified"
            batchsize_discretize = 100000
            
            for i in range(self.data.size(1)):
                self.data[:, i] = discretize_wrapper(self.data, i, n_bins, batchsize_discretize)

    def build_model(self):
        """
        The build_model method gets the necessary parameters and defines and builds the model.
        Currently, supported models are INN, TBD and DDPM.
        Currently, all diffusion models are based on a Resnet.

        New model classes implemented in this framework must take a param file as input.
        See documentation of ModelBase class for more guidelines on how to implement new model classes.

        Overwrite this method if a completely different model structure is required
        This method should place the model under self.model
        """

        # Read in the parameters required to build the model. Raise an error if one of them is not specified
        n_blocks = get(self.params, "n_blocks", None)
        assert n_blocks is not None, "build_model: n_blocks not specified"
        intermediate_dim = get(self.params, "intermediate_dim", None)
        assert intermediate_dim is not None, "build_model: intermediate_dim not specified"
        if (self.modelname == "INN" or self.modelname == "TBD" or self.modelname == "DDPM"):
            layers_per_block = get(self.params, "layers_per_block", None)
            assert layers_per_block is not None, "build_model: layers_per_block not specified"
            encode_t = get(self.params, "encode_t", False)
            print(f"build_model: Trying to build model {self.model} "
                  f"with n_blocks {n_blocks}, layers_per_block {layers_per_block}, "
                  f"intermediate dim {intermediate_dim} and encode_t {encode_t}")
            if self.modelname == "INN":
                self.modelname = INN(self.params)
            elif self.modelname == "TBD":
                self.modelname = TBD(self.params)
            elif self.modelname == "DDPM":
                self.modelname = DDPM(self.params)
        elif self.modelname == "GPT":
            self.params["vocab_size"] = 2*self.params["n_bins"]
            self.params["block_size"] = self.data.size(1)-1
            
            self.model = GPTwrapper(self.params)
        else:
            raise ValueError(f"build_model: model class {self.modelname} not recognised. Use INN, TBD, DDPM or GPT")

        # Keep track of the total number of trainable model parameters
        model_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.params["model_parameters"] = model_parameters
        print(f"build_model: Built model {self.modelname}. Total number of parameters: {model_parameters}")

        # If "warm_start", load the model parameters from the specified directory.
        # It is expected that they are found under warm_start_dir/models/checkpoint
        if get(self.params, "warm_start", False):
            try:
                state_dict = torch.load(self.warm_start_path + "/models/checkpoint.pt", map_location=self.device)
            except FileNotFoundError:
                raise ValueError(f"build_model: cannot load model for warm_start")

            self.model.load_state_dict(state_dict)
            print(f"build_model: Loaded state_dict from warm_start_path {self.warm_start_path}")
        else:
            if not get(self.params, "train", True):
                print("build_model: CARE !!! train set to False and warm_start set to False")

        if get(self.params, "sample_periodically", False):
            # The Model needs these values to make intermediate plots
            # TODO: Can we make this nicer?
            self.model.data_mean = self.data_mean
            self.model.data_std = self.data_std
            self.model.data_u = self.data_u
            self.model.data_s = self.data_s
            self.model.obs_names = self.obs_names
            self.model.obs_ranges = self.obs_ranges

    def build_optimizer(self):
        """
        The build_optimizer method gets the necessary parameters and builds the training optimizer.
        Currently only vanilla Adam is implemented.
        TODO: Implement SGD
        TODO: Implement LR scheduling

        Overwrite or extend this method if a different optimizer is needed.
        The method should place the optimizer under self.model.optimizer
        """

        # Read in the "train" parameter. If it is set to True, build the optimizer, otherwise skip it.
        train = get(self.params, "train", True)
        if train:
            # Read in the "optimizer" parameter and build the specified optimizer
            # TODO: Not very nice
            optim = get(self.params, "optimizer", None)
            if optim is None:
                optim = "Adam"
                print(f"build_optimizer: optimizer not specified. Defaulting to {optim}")

            if optim == "Adam":
                lr = get(self.params, "lr", 0.0001)
                betas = get(self.params, "betas", [0.9, 0.999])
                weight_decay = get(self.params, "weight_decay", 0)
                self.model.optimizer = \
                    Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
                print(
                    f"build_optimizer: Built optimizer {optim} with lr {lr}, betas {betas}, weight_decay {weight_decay}")
            else:
                raise ValueError(f"build_optimizer: optimizer {optim} not implemented")

        else:
            print("build_optimizer: train set to False. Not building optimizer")

    def build_dataloaders(self):
        """
        The build_dataloaders methods gets the necessary parameters and builds the train_loader, val_loader and test_loader
        TODO: test_loader used for nothing atm

        Overwrite this method if necessary.
        This method should place the loaders under
        self.model.train_loader, self.model.val_loader and self.model.test_loader respectively
        """

        # Make sure the data is a torch.Tensor of correct type and move it to device
        if self.modelname == "GPT":
            self.data = self.data.long()
        else:
            self.data = self.data.float()
        self.data = self.data.to(self.device)
        print(f"preprocess_data: Moved data to {self.data.device}")

        # Read in the "train" parameter. If it is set to True, build the dataloaders, otherwise skip it.
        train = get(self.params, "train", True)
        # Read in the "data_split" parameter, specifying which parts of the data to use for training, validation and test
        self.data_split = get(self.params, "data_split", [0.55, 0.05, 0.4])
        if train:
            # Read in the "batch_size" parameter and calculate the cuts between traindata, valdata and testdata
            # Define the loaders
            self.batch_size = get(self.params, "batch_size", 1024)
            cut1 = int(self.n_data * self.data_split[0])
            cut2 = int(self.n_data * (self.data_split[0] + self.data_split[1]))
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

    def train_model(self):
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
            self.params["traintime"] = traintime
            n_epochs = self.params["n_epochs"]
            print(f"train_model: Finished training {n_epochs} epochs after {traintime} seconds.")

            if get(self.params, "validate", True):
                best_val_loss = self.params["best_val_loss"]
                best_val_epoch = self.params["best_val_epoch"]
                print(f"train_model: Best val_loss {best_val_loss} at epoch {best_val_epoch}")

            # Save the final model. Update the total amount of training epochs the model has been trained for
            torch.save(self.model.state_dict(), f"models/model_run{self.runs}.pt")
            self.total_epochs = self.total_epochs + get(self.params, "n_epochs", 1000)
            self.params["total_epochs"] = self.total_epochs

        else:
            print("train_model: train set to False. Not training")
        print(f"train_model: Model has been trained for a total of {self.total_epochs} epochs")

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
            # Read in the "load_best_checkpoint" parameter.
            # If it is set to True, try to load the best validation checkpoint of the model.
            # Otherwise, just use the model state after the last epoch
            load_best_checkpoint = get(self.params, "load_best_checkpoint", True)
            if load_best_checkpoint:
                try:
                    state_dict = torch.load(self.out_dir + "/models/checkpoint.pt", map_location=self.device)
                    self.model.load_state_dict(state_dict)
                except Exception:
                    print(f"generate_samples: cannot load best checkpoint. Sampling with current model")

            # Read in the "n_samples" parameter specifying how many samples to generate
            # Call the model.sample_n_parallel(n_samples) method to perform the sampling
            n_samples = get(self.params, "n_samples", 1000000)
            print(f"generate_samples: Starting generation of {n_samples} samples")
            t0 = time.time()
            self.samples = np.concatenate([self.model.sample_n(n_samples//200) for _ in range(200)], 0)
            t1 = time.time()
            sampletime = t1 - t0
            self.params["sampletime"] = sampletime

            #undo discretization
            if self.modelname == "GPT":
                events = np.zeros_like(self.samples, dtype="float")
                n_bins = self.params["n_bins"]
                for i in range(self.data.size(1)):
                    bin_means = create_bins(self.data[:, i].cpu().numpy(), n_bins)
                    nDiff = 0
                    if(i%2==1): #do this better
                        nDiff=1
                    events[:,i] = bin_means[np.clip(self.samples[:, i] - nDiff * n_bins, 0, n_bins-1)]
                self.samples = events  

            # Undo the preprocessing of the samples
            # TODO: Currently they are mapped back to 16dim for this. Could be made more efficient
            if get(self.params, "preprocess", True):
                self.samples = undo_preprocessing(self.samples,
                                                  self.data_mean, self.data_std, self.data_u, self.data_s,
                                                  self.channels, keep_all=True)

            if get(self.params, "save_samples", False):
                np.save("samples_final.npy", self.samples)
            print(f"generate_samples: Finished generation of {n_samples} samples after {sampletime} seconds")
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
            # Transform the raw_data from (E, p_x, p_y, p_z) to (p_T, phi, eta, m)
            # self.data_raw = EpppToPTPhiEta(self.data_raw, reduce_data=False, include_masses=True)
            # Create a directory to save the plots
            os.makedirs(f"plots/run{self.runs}", exist_ok=True)
            self.plot_channels = get(self.params, "plot_channels", None)

            # Read in the "plot_channels" parameter, specifying for which observables we want a 1d histogram
            # Default to all observables if not specified
            if self.plot_channels is None:
                self.plot_channels = self.channels
                print(f"make_plots: plot_channels not specified. Defaulting to all channels {self.channels}")

            # The cut between train and test data
            cut = int(self.n_data * (self.data_split[0] + self.data_split[1]))

            # Draw all 1d histograms into one PDF file
            with PdfPages(f"plots/run{self.runs}/1d_histograms") as out:
                # Loop over the plot_channels
                for i, channel in enumerate(self.plot_channels):
                    # Get the train data, test data and generated data for the channel
                    obs_train = self.data_raw[:cut, channel]
                    obs_test = self.data_raw[cut:, channel]
                    obs_generated = self.samples[:, channel]
                    # Get the name and the range of the observable
                    obs_name = self.obs_names[channel]
                    obs_range = self.obs_ranges[channel]
                    # Create the plot
                    plot_obs(pp=out,
                             obs_train=obs_train,
                             obs_test=obs_test,
                             obs_predict=obs_generated,
                             name=obs_name,
                             range=obs_range)

            # Draw a 1d histogram for the DeltaR between the two jets
            # This requires that the channels 9,10,13,14 where part of the experiment
            plot_DeltaR = get(self.params, "plot_deltaR", True)
            if plot_DeltaR:
                if all(c in self.channels for c in [9, 10, 13, 14]):
                    file_name = f"plots/run{self.runs}/deltaR_j1_j2.pdf"
                    obs_name = "\Delta R_{j_1 j_2}"
                    obs_train = delta_r(self.data_raw[:cut])
                    obs_test = delta_r(self.data_raw[cut:])
                    obs_generated = delta_r(self.samples[:])
                    plot_obs(pp=file_name,
                             obs_train=obs_train,
                             obs_test=obs_test,
                             obs_predict=obs_generated,
                             name=obs_name,
                             range=[0, 8])
                else:
                    print("make_plots: plot_deltaR ist set to True, but missing at least one required channel")

            # Draw a 2d histogram DeltaEta vs DeltaPhi for the two jets
            # This requires that the channels 9,10,13,14 where part of the experiment
            plot_Deta_Dphi = get(self.params, "plot_Deta_Dphi", True)
            if plot_Deta_Dphi:
                if all(c in self.channels for c in [9, 10, 13, 14]):
                    file_name = f'plots/run{self.runs}/deta_dphi.png'
                    plot_deta_dphi(file_name=file_name,
                                   data_train=self.data_raw[:cut],
                                   data_test=self.data_raw[cut:],
                                   data_generated=self.samples[:])
                else:
                    print("make_plots: plot_Deta_Dphi ist set to True, but missing at least one required channel")

            print("make_lots: Finished making plots")
        else:
            print("make_plots: plot set to False")

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
