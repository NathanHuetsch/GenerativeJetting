import numpy as np
import torch
from Source.Util.datasets import Dataset
from Source.Util.preprocessing import preformat, preprocess, undo_preprocessing
from Source.Util.util import get_device, save_params, get, load_params
from Source.Experiments.ExperimentBase import Experiment
import os, sys, time


class Jet_Experiment(Experiment):
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
        super().__init__(params)
        # Names of the observables for plotting
        self.obs_names = ["p_{T,l1}", "\phi_{l1}", "\eta_{l1}", "\mu_{l1}",
                          "p_{T,l2}", "\phi_{l2}", "\eta_{l2}", "\mu_{l2}",
                          "p_{T,j1}", "\phi_{j1}", "\eta_{j1}", "\mu_{j1}",
                          "p_{T,j2}", "\phi_{j2}", "\eta_{j2}", "\mu_{j2}",
                          "p_{T,j3}", "\phi_{j3}", "\eta_{j3}", "\mu_{j3}"]

        # Ranges of the observables for plotting
        self.obs_ranges = [[0.5, 150], [-4, 4], [-6, 6], [0, 50],
                           [0.5, 150], [-4, 4], [-6, 6], [0, 50],
                           [0.5, 150], [-4, 4], [-6, 6], [0, 50],
                           [0.5, 150], [-4, 4], [-6, 6], [0, 50],
                           [0.5, 150], [-4, 4], [-6, 6], [0, 50]]

        self.params["istoy"] = False
        self.con_depth = get(self.params, "con_depth", 0)
        self.n_jets = get(self.params, "n_jets", 1)
        self.channels = get(self.params, "channels", None)
        if self.channels is None:
            self.channels = np.array([i for i in range(self.n_jets * 4 + 8) if i not in [1, 3, 7]]).tolist()
            self.params["channels"] = self.channels
        
        self.prior_model = None
        self.prior_prior_model = None

        if not self.bayesian and self.iterations>1:
            raise ValueError("jet.__init__: Trying to train an ensemble on the jet dataset, but we do not do this here.")

    def load_data(self):
        """
        The load_data method gets the necessary parameters and reads in the data
        Currently supported are datasets of type *.npy ; *.torch ; *.h5

        Overwrite this method if other ways of reading in data are needed
        This method should place the data under self.data_raw
                """
        # Read in the "data_path" parameter. Raise and error if it is not specified or does not exist
        data_path = get(self.params, "data_path", None)
        fraction = get(self.params,"fraction",None)
        if data_path is None:
            raise ValueError("load_data: data_path is None. Please specify in params")
        assert os.path.exists(data_path), f"load_data: data_path {data_path} does not exist"

        if self.load_dataset:
            # Read in the "data_type" parameter, defaulting to np if not specified. Try to read in the data accordingly
            data_type = get(self.params, "data_type", "np")
            if data_type == "np":
                self.data_raw = np.load(data_path)
                print(f"load_data: Loaded data with shape {self.data_raw.shape} from ", data_path)
            else:
                raise ValueError(f"load_data: Cannot load data from {data_path}")
        else:
            data_all = Dataset(data_path, fraction=fraction, conditional=self.conditional)
            self.z_1 = data_all.z_1
            self.z_2 = data_all.z_2
            self.z_3 = data_all.z_3

    def preprocess_data(self, p, data_raw, save_in_params=True, conditional=False):
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
        channels = get(p, "channels", None)
        n_jets = get(p, "n_jets", 2)
        if channels is None:
            print(f"preprocess_data: channels and dim not specified. Defaulting to {5 + 4 * n_jets} channels.")
            channels = np.array([i for i in range(n_jets * 4 + 8) if i not in [1, 3, 7]]).tolist()
            p["channels"] = channels
            print(f"preprocess_data: channels {channels}")
        else:
            print(f"preprocess_data: channels {channels} specified.")
        # Do the preprocessing
        data_raw = preformat(data_raw, p)

        data, data_mean, data_std, data_u, data_s, bin_edges, bin_means = preprocess(data_raw, p)
        print("preprocess_data: Finished preprocessing")

        n_data = len(data)

        # Quick optional check whether preprocessing works as intended (data_raw = data_raw2?)
        # data_raw2 = undo_preprocessing(data, data_mean, data_std, data_u, data_s, bin_edges, bin_means, p)

        # Make sure the data is a torch.Tensor and move it to device
        data = data.to(self.device)
        print(f"preprocess_data: Moved data to {data.device}")

        if save_in_params:
            if get(p, "dim", None) is None:
                self.params["dim"] = len(channels)

            self.params["channels"] = channels
            self.params["n_data"] = n_data
        return data, data_mean, data_std, data_u, data_s, bin_edges, bin_means, data_raw
