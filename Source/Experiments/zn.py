import numpy as np
import torch
from torch.utils.data import DataLoader
from Source.Util.datasets import Dataset
from Source.Util.util import get_device, save_params, get, load_params
from Source.Util.preprocessing import preformat, preprocess, undo_preprocessing
from Source.Util.plots import plot_obs, delta_r, plot_deta_dphi, plot_obs_2d, plot_loss, plot_binned_sigma, plot_mu_sigma
from Source.Util.physics import get_M_ll
from matplotlib.backends.backend_pdf import PdfPages
from Source.Experiments.jet import Jet_Experiment
from torch.optim.lr_scheduler import CosineAnnealingLR
from Source.Util.lr_scheduler import OneCycleLR
import os, sys, time, random

class Zn_Experiment(Jet_Experiment):
    """
    Class to run Z+njet generative modelling experiments
    """

    def __init__(self, params):
        """
        The __init__ method reads in the parameters and saves them under self.params
        It also makes some useful definitions
        """
        super().__init__(params)
        
        if get(self.params, "plot_channels", None) is None:
            self.plot_channels = self.channels
            self.params["plot_channels"] = self.channels


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

        data_all = Dataset(data_path, fraction=fraction, conditional=self.conditional)
        self.z_1 = data_all.z_1
        self.z_2 = data_all.z_2
        self.z_3 = data_all.z_3

        print(f"load_data: Loaded data with shapes {self.z_1.shape}, {self.z_2.shape}, {self.z_3.shape} from {data_path}")

    def preprocess_data_njets(self):
        channels = get(self.params, "channels", None)
        n_jets = 3
        if channels is None:
            print(f"preprocess_data: channels and dim not specified. Defaulting to {5 + 4 * n_jets} channels.")
            channels = np.array([i for i in range(n_jets * 4 + 8) if i not in [1, 3, 7]]).tolist()
            self.params["channels"] = channels
            print(f"preprocess_data: channels {channels}")
        else:
            print(f"preprocess_data: channels {channels} specified.")
        self.params["dim"] = max(channels)+1

        channels_3jet = np.array(self.params["channels"], dtype="int")
        channels_1jet = channels_3jet[~np.isin(channels_3jet, np.array([1, 3, 7, 12, 13, 14, 15, 16, 17, 18, 19])).astype("bool")]
        channels_2jet = channels_3jet[~np.isin(channels_3jet, np.array([1, 3, 7, 16, 17, 18, 19])).astype("bool")]
        self.channels_njet = [channels_1jet, channels_2jet, channels_3jet]

        def preprocess_for_njets(data_raw, n_jets, p):
            p["channels"] = self.channels_njet[n_jets-1]
            
            data_raw = preformat(data_raw, p)
            data, data_mean, data_std, data_u, data_s, bin_edges, bin_means = preprocess(data_raw, p)

            data = data.to(self.device)
            return data_raw, data, data_mean, data_std, data_u, data_s, bin_edges, bin_means

        self.data_raw_1, self.data_1, self.data_mean_1, self.data_std_1, self.data_u_1, self.data_s_1, \
                     self.bin_edges_1, self.bin_means_1 = preprocess_for_njets(self.z_1, 1, self.params)
        self.data_raw_2, self.data_2, self.data_mean_2, self.data_std_2, self.data_u_2, self.data_s_2, \
                     self.bin_edges_2, self.bin_means_2 = preprocess_for_njets(self.z_2, 2, self.params)
        self.data_raw_3, self.data_3, self.data_mean_3, self.data_std_3, self.data_u_3, self.data_s_3, \
                     self.bin_edges_3, self.bin_means_3 = preprocess_for_njets(self.z_3, 3, self.params)

    def build_dataloaders(self):
        train = get(self.params, "train", True)

        cut_1_1 = int(len(self.data_1) * self.data_split[0])
        cut_1_2 = int(len(self.data_1) * (self.data_split[0] + self.data_split[1]))
        self.data_train_1 = self.data_raw_1[:cut_1_1]
        self.data_test_1 = self.data_raw_1[cut_1_2:]

        cut_2_1 = int(len(self.data_2) * self.data_split[0])
        cut_2_2 = int(len(self.data_2) * (self.data_split[0] + self.data_split[1]))
        self.data_train_2 = self.data_raw_2[:cut_2_1]
        self.data_test_2 = self.data_raw_2[cut_2_2:]

        cut_3_1 = int(len(self.data_3) * self.data_split[0])
        cut_3_2 = int(len(self.data_3) * (self.data_split[0] + self.data_split[1]))
        self.data_train_3 = self.data_raw_3[:cut_3_1]
        self.data_test_3 = self.data_raw_3[cut_3_2:]

        self.model.data_train = self.data_train_3 #for KL loss (this is technically cheating, should fix this)

        if train:
            train_set = Conditional_Jet_Dataset(self.data_1[:cut_1_1], self.data_2[:cut_2_1], self.data_3[:cut_3_1])
            self.model.train_loader =Conditional_Jet_DataLoader(dataset=train_set,
                                                 batch_size=self.batch_size, shuffle=True)

            self.sample_periodically = get(self.params, "sample_periodically", False)
            print(
                f"build_dataloaders: Built dataloaders with data_split {self.data_split} and batch_size {self.batch_size}")

            use_scheduler = get(self.params, "use_scheduler", False)
            if use_scheduler:
                lr_scheduler = get(self.params, "lr_scheduler", "OneCycle")
                if lr_scheduler == "OneCycle":
                    lr = get(self.params, "lr", 0.0001)
                    n_epochs = get(self.params, "n_epochs", 100)
                    self.model.scheduler = OneCycleLR(
                        self.model.optimizer,
                        lr * 10,
                        epochs=n_epochs,
                        steps_per_epoch=len(self.model.train_loader))
                    print("build_dataloaders: Using one-cycle lr scheduler")
                elif lr_scheduler == "CosineAnnealing":
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

    def prepare_training(self):
        print("train_model: Preparing model training")
        self.use_scheduler = get(self.params, "use_scheduler", False)
        self.model.train_losses = np.array([])
        self.model.train_losses_epoch = np.array([])
        self.n_trainbatches = len(self.model.train_loader)

        self.sample_periodically = get(self.params, "sample_periodically", False)
        if self.sample_periodically:
            self.sample_every = get(self.params, "sample_every", 1)
            self.sample_every_n_samples = get(self.params, "sample_every_n_samples", 100000)
            print(f'train_model: sample_periodically set to True. Sampling {self.sample_every_n_samples} every'
                  f' {self.sample_every} epochs. This may significantly slow down training!')

    def train_model(self):
        train = get(self.params, "train", True)
        if train:
            os.makedirs("models", exist_ok=True)
            t0 = time.time()
            self.run_training()
            t1 = time.time()
            traintime = t1 - t0
            n_epochs = get(self.params,"n_epochs",100)
            print(f"train_model: Finished training {n_epochs} epochs after {traintime:.2f} s = {traintime/60:.2f} min = {traintime/60**2:.2f} h.")

            torch.save(self.model.state_dict(), f"models/model_run{self.runs}.pt")

            self.params["total_epochs"] = self.total_epochs
            self.params["traintime"] = traintime
            self.total_epochs = self.total_epochs + n_epochs

        else:
            print("train_model: train set to False. Not training")
        print(f"train_model: Model has been trained for a total of {self.total_epochs} epochs")

    def run_training(self):
        self.prepare_training()
        samples = []
        n_epochs = get(self.params, "n_epochs", 100)
        past_epochs = get(self.params, "total_epochs", 0)
        print(f"train_model: Model has been trained for {past_epochs} epochs before.")
        print(f"train_model: Beginning training. n_epochs set to {n_epochs}")
        for e in range(n_epochs):
            t0 = time.time()
            self.model.epoch = past_epochs + e
            self.model.train()
            self.train_one_epoch()

            if self.sample_periodically:
                if (self.model.epoch + 1) % self.sample_every == 0:
                    self.model.eval()
                    samples_1, samples_2, samples_3 = self.sample_and_undo(self.sample_every_n_samples)
                    self.plot_samples(samples_1, samples_2, samples_3)

            if get(self.params,"save_periodically",False):
                if (self.model.epoch + 1) % get(self.params,"save_every",10) == 0 or self.model.epoch==0:
                    torch.save(self.model.state_dict(), f"models/model_epoch_{e+1}.pt")

            if e==0:
                t1 = time.time()
                dtEst= (t1-t0) * n_epochs
                print(f"Training time estimate: {dtEst/60:.2f} min = {dtEst/60**2:.2f} h")

    def train_one_epoch(self):
        train_losses = np.array([])
        for batch_id, x in enumerate(self.model.train_loader):
            x1, x2, x3 = x
            self.model.optimizer.zero_grad()
            loss1 = self.model.batch_loss(x1, n_jets=1, pos=self.channels_njet[0])
            loss2 = self.model.batch_loss(x2, n_jets=2, pos=self.channels_njet[1])
            loss3 = self.model.batch_loss(x3, n_jets=3, pos=self.channels_njet[2])
            loss = (loss1 * 17/9 + loss2 * 17/13 + loss3)/3 #do a proper mean

            if np.isfinite(loss.item()): 
                loss.backward()
                self.model.optimizer.step()
                train_losses = np.append(train_losses, loss.item())
                if self.use_scheduler:
                    self.model.scheduler.step()
            else:
                print(f"train_model: Unstable loss. Skipped backprop for epoch {self.model.epoch}, batch_id {batch_id}")
        self.model.train_losses_epoch = np.append(self.model.train_losses_epoch, train_losses.mean())
        self.model.train_losses = np.concatenate([self.model.train_losses, train_losses], axis=0)

    def generate_samples(self):
        sample = get(self.params, "sample", True)

        if self.bayesian:
            iterations = self.iterations
        else:
            iterations = 1

        if sample:
            bay_samples_1 = []
            bay_samples_2 = []
            bay_samples_3 = []
            for i in range(0, iterations):
                n_samples = get(self.params, "n_samples", 1000000)
                print(f"generate_samples: Starting generation of {n_samples} samples")
                t0 = time.time()
                sample_1, sample_2, sample_3 = self.sample_and_undo(n_samples)
                t1 = time.time()
                sampletime = t1 - t0
                self.params["sampletime"] = sampletime
                bay_samples_1.append(sample_1)
                bay_samples_2.append(sample_2)
                bay_samples_3.append(sample_3)

                print(f"generate_samples: Finished generation of {n_samples} samples after {sampletime:.2f} s = {sampletime/60:.2f} min.")
                if get(self.params, "save_samples", False):
                    os.makedirs('samples', exist_ok=True)
                    np.save(f"samples/samples_final_{i}.npy", sample)
                    print(f"save_samples: generated samples have been saved")

            self.samples_1 = np.concatenate(bay_samples_1)
            self.samples_2 = np.concatenate(bay_samples_2)
            self.samples_3 = np.concatenate(bay_samples_3)
        else:
            print("generate_samples: sample set to False")

    def sample_and_undo(self, n_samples):
        samples_1 = self.model.sample_n(n_samples, n_jets=1, pos=self.channels_njet[0])
        p = self.params.copy()
        p["channels"] = self.channels_njet[0]
        samples_1 = undo_preprocessing(samples_1, self.data_mean_1, self.data_std_1, self.data_u_1,
                                       self.data_s_1, self.bin_edges_1, self.bin_means_1, p)

        samples_2 = self.model.sample_n(n_samples, n_jets=2, pos=self.channels_njet[1])
        p = self.params.copy()
        p["channels"] = self.channels_njet[1]
        samples_2 = undo_preprocessing(samples_2, self.data_mean_2, self.data_std_2, self.data_u_2,
                                       self.data_s_2, self.bin_edges_2, self.bin_means_2, p)

        samples_3 = self.model.sample_n(n_samples, n_jets=3, pos=self.channels_njet[2])
        p = self.params.copy()
        p["channels"] = self.channels_njet[2] #technically change nothing (but want to keep things symmetric)
        samples_3 = undo_preprocessing(samples_3, self.data_mean_3, self.data_std_3, self.data_u_3,
                                       self.data_s_3, self.bin_edges_3, self.bin_means_3, p)

        return samples_1, samples_2, samples_3

    def make_plots(self):
        plot = get(self.params, "plot", True)
        sample = get(self.params, "sample", True)
        if plot and sample:
            print(f"make_plots: plotting {self.plot_channels}")
            self.plot_samples(self.samples_1, self.samples_2, self.samples_3, finished=True)

            print("make_plots: Finished making plots")
        else:
            print("make_plots: plot set to False")

    def plot_samples(self, samples_1, samples_2, samples_3, finished=False):
        os.makedirs(f"plots", exist_ok=True)
        if finished:
            path = f"plots/run{self.runs}"
            os.makedirs(path, exist_ok=True)
            iterations = self.iterations
        else:
            path = "plots"
            iterations = 1

        n_epochs = self.model.epoch + get(self.params, "total_epochs", 0)

        plot_train_1 = self.data_train_1
        plot_test_1 = self.data_test_1
        plot_samples_1 = samples_1
        plot_train_2 = self.data_train_2
        plot_test_2 = self.data_test_2
        plot_samples_2 = samples_2
        plot_train_3 = self.data_train_3
        plot_test_3 = self.data_test_3
        plot_samples_3 = samples_3

        with PdfPages(f"{path}/1d_hist_epoch_{n_epochs}.pdf") as out:
            for n_jets in [1,2,3]:
                plot_channels = np.array([i for i in range(n_jets * 4 + 8) if i not in [1, 3, 7]]).tolist()
                for _, channel in enumerate(plot_channels):
                    obs_train = eval(f"plot_train_{n_jets}")[:, channel]
                    obs_test = eval(f"plot_test_{n_jets}")[:, channel]
                    obs_generated = eval(f"plot_samples_{n_jets}")[:, channel]
                    # Get the name and the range of the observable
                    obs_name = self.obs_names[channel]
                    obs_range = self.obs_ranges[channel]
                    # Create the plot
                    plot_obs(pp=out,
                             obs_train=obs_train,
                             obs_test=obs_test,
                             obs_predict=obs_generated,
                             name=obs_name,
                             range=obs_range,
                             n_epochs=n_epochs,
                             n_jets=n_jets,
                             weight_samples=iterations)


        if get(self.params,"plot_deltaR", True):
            with PdfPages(f"{path}/deltaR_jl_jm_epoch_{n_epochs}.pdf") as out:
                obs_name = "\Delta R_{j_1 j_2}"
                obs_train = delta_r(plot_train_2)
                obs_test = delta_r(plot_test_2)
                obs_generated = delta_r(plot_samples_2)
                plot_obs(pp=out,
                             obs_train=obs_train,
                             obs_test=obs_test,
                             obs_predict=obs_generated,
                             name=obs_name,
                             n_epochs=n_epochs,
                             n_jets=2,
                             range=[0, 8],
                             weight_samples=iterations)

                obs_name = "\Delta R_{j_1 j_2}"
                obs_train = delta_r(plot_train_3)
                obs_test = delta_r(plot_test_3)
                obs_generated = delta_r(plot_samples_3)
                plot_obs(pp=out,
                             obs_train=obs_train,
                             obs_test=obs_test,
                             obs_predict=obs_generated,
                             name=obs_name,
                             n_epochs=n_epochs,
                             n_jets=3,
                             range=[0, 8],
                             weight_samples=iterations)
                
                obs_name = "\Delta R_{j_1 j_3}"
                obs_train = delta_r(plot_train_3, idx_phi1=9, idx_eta1=10, idx_phi2=17, idx_eta2=18)
                obs_test = delta_r(plot_test_3, idx_phi1=9, idx_eta1=10, idx_phi2=17, idx_eta2=18)
                obs_generated = delta_r(plot_samples_3, idx_phi1=9, idx_eta1=10, idx_phi2=17,
                                        idx_eta2=18)
                plot_obs(pp=out,
                                 obs_train=obs_train,
                                 obs_test=obs_test,
                                 obs_predict=obs_generated,
                                 name=obs_name,
                                 n_epochs=n_epochs,
                                 n_jets=3,
                                 range=[0, 8],
                                 weight_samples=iterations)

                obs_name = "\Delta R_{j_2 j_3}"
                obs_train = delta_r(plot_train_3, idx_phi1=13, idx_eta1=14, idx_phi2=17, idx_eta2=18)
                obs_test = delta_r(plot_test_3, idx_phi1=13, idx_eta1=14, idx_phi2=17, idx_eta2=18)
                obs_generated = delta_r(plot_samples_3, idx_phi1=13, idx_eta1=14, idx_phi2=17,
                                                idx_eta2=18)
                plot_obs(pp=out,
                                 obs_train=obs_train,
                                 obs_test=obs_test,
                                 obs_predict=obs_generated,
                                 name=obs_name,
                                 n_epochs=n_epochs,
                                 n_jets=3,
                                 range=[0, 8],
                                 weight_samples=iterations)


        if get(self.params,"plot_Deta_Dphi", True):
            with PdfPages(f"{path}/deta_dphi_jets_epoch_{n_epochs}.pdf") as out:
                plot_deta_dphi(pp=out,
                               data_train=plot_train_2,
                               data_test=plot_test_2,
                               data_generated=plot_samples_2,
                               n_jets=2,
                               n_epochs=n_epochs)


                plot_deta_dphi(pp=out,
                               data_train=plot_train_3,
                               data_test=plot_test_3,
                               data_generated=plot_samples_3,
                               n_jets=3,
                               n_epochs=n_epochs)                

                plot_deta_dphi(pp=out,
                                   data_train=plot_train_3,
                                   data_test=plot_test_3,
                                   data_generated=plot_samples_3,
                                   idx_phi1=9,
                                   idx_phi2=17,
                                   idx_eta1=10,
                                   idx_eta2=18,
                                   n_jets=3,
                                   n_epochs=n_epochs)

                plot_deta_dphi(pp=out,
                                   data_train=plot_train_3,
                                   data_test=plot_test_3,
                                   data_generated=plot_samples_3,
                                   idx_phi1=13,
                                   idx_phi2=17,
                                   idx_eta1=14,
                                   idx_eta2=18,
                                   n_jets=3,
                                   n_epochs=n_epochs)

        if get(self.params, "plot_deltaR_all", True):
            with PdfPages(f"{path}/deltaR_all_epoch_{n_epochs}.pdf") as out:
                # 1-jet
                obs_name = "\Delta R_{l_1 l_2}"
                obs_train = delta_r(plot_train_1, idx_phi1=1, idx_eta1=2, idx_phi2=5, idx_eta2=6)
                obs_test = delta_r(plot_test_1, idx_phi1=1, idx_eta1=2, idx_phi2=5, idx_eta2=6)
                obs_generated = delta_r(plot_samples_1, idx_phi1=1, idx_eta1=2, idx_phi2=5, idx_eta2=6)
                plot_obs(pp=out,
                             obs_train=obs_train,
                             obs_test=obs_test,
                             obs_predict=obs_generated,
                             name=obs_name,
                             n_epochs=n_epochs,
                             n_jets=1,
                             range=[0, 8],
                             weight_samples=iterations)

                obs_name = "\Delta R_{l_1 j_1}"
                obs_train = delta_r(plot_train_1, idx_phi1=1, idx_eta1=2, idx_phi2=9, idx_eta2=10)
                obs_test = delta_r(plot_test_1, idx_phi1=1, idx_eta1=2, idx_phi2=9, idx_eta2=10)
                obs_generated = delta_r(plot_samples_1, idx_phi1=1, idx_eta1=2, idx_phi2=9, idx_eta2=10)
                plot_obs(pp=out,
                             obs_train=obs_train,
                             obs_test=obs_test,
                             obs_predict=obs_generated,
                             name=obs_name,
                             n_epochs=n_epochs,
                             n_jets=1,
                             range=[0, 8],
                             weight_samples=iterations)

                obs_name = "\Delta R_{l_2 j_1}"
                obs_train = delta_r(plot_train_1, idx_phi1=5, idx_eta1=6, idx_phi2=9, idx_eta2=10)
                obs_test = delta_r(plot_test_1, idx_phi1=5, idx_eta1=6, idx_phi2=9, idx_eta2=10)
                obs_generated = delta_r(plot_samples_1, idx_phi1=5, idx_eta1=6, idx_phi2=9, idx_eta2=10)
                plot_obs(pp=out,
                             obs_train=obs_train,
                             obs_test=obs_test,
                             obs_predict=obs_generated,
                             name=obs_name,
                             n_epochs=n_epochs,
                             n_jets=1,
                             range=[0, 8],
                             weight_samples=iterations)

                # 2-jet
                obs_name = "\Delta R_{l_1 l_2}"
                obs_train = delta_r(plot_train_2, idx_phi1=1, idx_eta1=2, idx_phi2=5, idx_eta2=6)
                obs_test = delta_r(plot_test_2, idx_phi1=1, idx_eta1=2, idx_phi2=5, idx_eta2=6)
                obs_generated = delta_r(plot_samples_2, idx_phi1=1, idx_eta1=2, idx_phi2=5, idx_eta2=6)
                plot_obs(pp=out,
                             obs_train=obs_train,
                             obs_test=obs_test,
                             obs_predict=obs_generated,
                             name=obs_name,
                             n_epochs=n_epochs,
                             n_jets=2,
                             range=[0, 8],
                             weight_samples=iterations)

                obs_name = "\Delta R_{l_1 j_1}"
                obs_train = delta_r(plot_train_2, idx_phi1=1, idx_eta1=2, idx_phi2=9, idx_eta2=10)
                obs_test = delta_r(plot_test_2, idx_phi1=1, idx_eta1=2, idx_phi2=9, idx_eta2=10)
                obs_generated = delta_r(plot_samples_2, idx_phi1=1, idx_eta1=2, idx_phi2=9, idx_eta2=10)
                plot_obs(pp=out,
                             obs_train=obs_train,
                             obs_test=obs_test,
                             obs_predict=obs_generated,
                             name=obs_name,
                             n_epochs=n_epochs,
                             n_jets=2,
                             range=[0, 8],
                             weight_samples=iterations)

                obs_name = "\Delta R_{l_2 j_1}"
                obs_train = delta_r(plot_train_2, idx_phi1=5, idx_eta1=6, idx_phi2=9, idx_eta2=10)
                obs_test = delta_r(plot_test_2, idx_phi1=5, idx_eta1=6, idx_phi2=9, idx_eta2=10)
                obs_generated = delta_r(plot_samples_2, idx_phi1=5, idx_eta1=6, idx_phi2=9, idx_eta2=10)
                plot_obs(pp=out,
                             obs_train=obs_train,
                             obs_test=obs_test,
                             obs_predict=obs_generated,
                             name=obs_name,
                             n_epochs=n_epochs,
                             n_jets=2,
                             range=[0, 8],
                             weight_samples=iterations)

                obs_name = "\Delta R_{l_1 j_2}"
                obs_train = delta_r(plot_train_2, idx_phi1=1, idx_eta1=2, idx_phi2=13, idx_eta2=14)
                obs_test = delta_r(plot_test_2, idx_phi1=1, idx_eta1=2, idx_phi2=13, idx_eta2=14)
                obs_generated = delta_r(plot_samples_2, idx_phi1=1, idx_eta1=2, idx_phi2=13, idx_eta2=14)
                plot_obs(pp=out,
                                 obs_train=obs_train,
                                 obs_test=obs_test,
                                 obs_predict=obs_generated,
                                 name=obs_name,
                                 n_epochs=n_epochs,
                                 n_jets=2,
                                 range=[0, 8],
                                 weight_samples=iterations)

                obs_name = "\Delta R_{l_2 j_2}"
                obs_train = delta_r(plot_train_2, idx_phi1=5, idx_eta1=6, idx_phi2=13, idx_eta2=14)
                obs_test = delta_r(plot_test_2, idx_phi1=5, idx_eta1=6, idx_phi2=13, idx_eta2=14)
                obs_generated = delta_r(plot_samples_2, idx_phi1=5, idx_eta1=6, idx_phi2=13, idx_eta2=14)
                plot_obs(pp=out,
                                 obs_train=obs_train,
                                 obs_test=obs_test,
                                 obs_predict=obs_generated,
                                 name=obs_name,
                                 n_epochs=n_epochs,
                                 n_jets=2,
                                 range=[0, 8],
                                 weight_samples=iterations)

                # 3-jet
                obs_name = "\Delta R_{l_1 l_2}"
                obs_train = delta_r(plot_train_3, idx_phi1=1, idx_eta1=2, idx_phi2=5, idx_eta2=6)
                obs_test = delta_r(plot_test_3, idx_phi1=1, idx_eta1=2, idx_phi2=5, idx_eta2=6)
                obs_generated = delta_r(plot_samples_3, idx_phi1=1, idx_eta1=2, idx_phi2=5, idx_eta2=6)
                plot_obs(pp=out,
                             obs_train=obs_train,
                             obs_test=obs_test,
                             obs_predict=obs_generated,
                             name=obs_name,
                             n_epochs=n_epochs,
                             n_jets=3,
                             range=[0, 8],
                             weight_samples=iterations)

                obs_name = "\Delta R_{l_1 j_1}"
                obs_train = delta_r(plot_train_3, idx_phi1=1, idx_eta1=2, idx_phi2=9, idx_eta2=10)
                obs_test = delta_r(plot_test_3, idx_phi1=1, idx_eta1=2, idx_phi2=9, idx_eta2=10)
                obs_generated = delta_r(plot_samples_3, idx_phi1=1, idx_eta1=2, idx_phi2=9, idx_eta2=10)
                plot_obs(pp=out,
                             obs_train=obs_train,
                             obs_test=obs_test,
                             obs_predict=obs_generated,
                             name=obs_name,
                             n_epochs=n_epochs,
                             n_jets=3,
                             range=[0, 8],
                             weight_samples=iterations)

                obs_name = "\Delta R_{l_2 j_1}"
                obs_train = delta_r(plot_train_3, idx_phi1=5, idx_eta1=6, idx_phi2=9, idx_eta2=10)
                obs_test = delta_r(plot_test_3, idx_phi1=5, idx_eta1=6, idx_phi2=9, idx_eta2=10)
                obs_generated = delta_r(plot_samples_3, idx_phi1=5, idx_eta1=6, idx_phi2=9, idx_eta2=10)
                plot_obs(pp=out,
                             obs_train=obs_train,
                             obs_test=obs_test,
                             obs_predict=obs_generated,
                             name=obs_name,
                             n_epochs=n_epochs,
                             n_jets=3,
                             range=[0, 8],
                             weight_samples=iterations)

                obs_name = "\Delta R_{l_1 j_2}"
                obs_train = delta_r(plot_train_3, idx_phi1=1, idx_eta1=2, idx_phi2=13, idx_eta2=14)
                obs_test = delta_r(plot_test_3, idx_phi1=1, idx_eta1=2, idx_phi2=13, idx_eta2=14)
                obs_generated = delta_r(plot_samples_3, idx_phi1=1, idx_eta1=2, idx_phi2=13, idx_eta2=14)
                plot_obs(pp=out,
                                 obs_train=obs_train,
                                 obs_test=obs_test,
                                 obs_predict=obs_generated,
                                 name=obs_name,
                                 n_epochs=n_epochs,
                                 n_jets=3,
                                 range=[0, 8],
                                 weight_samples=iterations)

                obs_name = "\Delta R_{l_2 j_2}"
                obs_train = delta_r(plot_train_3, idx_phi1=5, idx_eta1=6, idx_phi2=13, idx_eta2=14)
                obs_test = delta_r(plot_test_3, idx_phi1=5, idx_eta1=6, idx_phi2=13, idx_eta2=14)
                obs_generated = delta_r(plot_samples_3, idx_phi1=5, idx_eta1=6, idx_phi2=13, idx_eta2=14)
                plot_obs(pp=out,
                                 obs_train=obs_train,
                                 obs_test=obs_test,
                                 obs_predict=obs_generated,
                                 name=obs_name,
                                 n_epochs=n_epochs,
                                 n_jets=3,
                                 range=[0, 8],
                                 weight_samples=iterations)
                
                obs_name = "\Delta R_{l_1 j_3}"
                obs_train = delta_r(plot_train_3, idx_phi1=1, idx_eta1=2, idx_phi2=17, idx_eta2=18)
                obs_test = delta_r(plot_test_3, idx_phi1=1, idx_eta1=2, idx_phi2=17, idx_eta2=18)
                obs_generated = delta_r(plot_samples_3, idx_phi1=1, idx_eta1=2, idx_phi2=17, idx_eta2=18)
                plot_obs(pp=out,
                                 obs_train=obs_train,
                                 obs_test=obs_test,
                                 obs_predict=obs_generated,
                                 name=obs_name,
                                 n_epochs=n_epochs,
                                 n_jets=3,
                                 range=[0, 8],
                                 weight_samples=iterations)

                obs_name = "\Delta R_{l_2 j_3}"
                obs_train = delta_r(plot_train_3, idx_phi1=5, idx_eta1=6, idx_phi2=17, idx_eta2=18)
                obs_test = delta_r(plot_test_3, idx_phi1=5, idx_eta1=6, idx_phi2=17, idx_eta2=18)
                obs_generated = delta_r(plot_samples_3, idx_phi1=5, idx_eta1=6, idx_phi2=17, idx_eta2=18)
                plot_obs(pp=out,
                                 obs_train=obs_train,
                                 obs_test=obs_test,
                                 obs_predict=obs_generated,
                                 name=obs_name,
                                 n_epochs=n_epochs,
                                 n_jets=3,
                                 range=[0, 8],
                                 weight_samples=iterations)

        if get(self.params, "plot_Mll", False):
            with PdfPages(f"{path}/M_ll_epochs_{n_epochs}.pdf") as out:        
                for n_jets in [1,2,3]:
                    obs_name = "M_{\ell \ell}"
                    obs_range = [75,110]
                    data_train = get_M_ll(eval(f"plot_train_{n_jets}"))
                    data_test = get_M_ll(eval(f"plot_test_{n_jets}"))
                    data_generated = get_M_ll(eval(f"plot_samples_{n_jets}"))
                    plot_obs(pp=out,
                             obs_train=data_train,
                             obs_test=data_test,
                             obs_predict=data_generated,
                             name=obs_name,
                             n_epochs=n_epochs,
                             range=obs_range,
                             n_jets=n_jets,
                             weight_samples=iterations)

        plot_1d_differences = get(self.params, "plot_1d_differences", True)
        if plot_1d_differences:
            with PdfPages(f"{path}/1d_differences_{n_epochs}.pdf") as out:
                # 1-jet
                differences = [[2, 6], [2, 10], [6, 10], [5, 9]]
                for channels in differences:
                    channel1 = channels[0]
                    channel2 = channels[1]
                    obs_name = self.obs_names[channel1] + " - " + self.obs_names[channel2]
                    
                    obs_train = plot_train_1[:, channel1] - plot_train_1[:, channel2]
                    obs_test = plot_test_1[:, channel1] - plot_test_1[:, channel2]
                    obs_generated = plot_samples_1[:, channel1] - plot_samples_1[:, channel2]
                    plot_obs(pp=out,
                                 obs_train=obs_train,
                                 obs_test=obs_test,
                                 obs_predict=obs_generated,
                                 name=obs_name,
                                 n_epochs=n_epochs,
                                 n_jets=1,
                                 weight_samples=iterations)
                # 2-jet
                differences = [[2, 6], [2, 10], [2, 14], [6, 10], [6, 14], [10, 14], [5, 9], [5, 13], [9, 13], [1,5], [1,9], [1,13]]
                for channels in differences:
                    channel1 = channels[0]
                    channel2 = channels[1]
                    obs_name = self.obs_names[channel1] + " - " + self.obs_names[channel2]
                    
                    obs_train = plot_train_2[:, channel1] - plot_train_2[:, channel2]
                    obs_test = plot_test_2[:, channel1] - plot_test_2[:, channel2]
                    obs_generated = plot_samples_2[:, channel1] - plot_samples_2[:, channel2]
                    plot_obs(pp=out,
                                 obs_train=obs_train,
                                 obs_test=obs_test,
                                 obs_predict=obs_generated,
                                 name=obs_name,
                                 n_epochs=n_epochs,
                                 n_jets=1,
                                 weight_samples=iterations)
                # 3-jet
                differences = [[2, 6], [2, 10], [2, 14], [6, 10], [6, 14], [10, 14], [5, 9], [5, 13], [9, 13],
                               [2, 18], [6, 18], [10, 18], [14, 18], [5, 17], [9, 17], [13, 17]]
                for channels in differences:
                    channel1 = channels[0]
                    channel2 = channels[1]
                    obs_name = self.obs_names[channel1] + " - " + self.obs_names[channel2]
                    
                    obs_train = plot_train_3[:, channel1] - plot_train_3[:, channel2]
                    obs_test = plot_test_3[:, channel1] - plot_test_3[:, channel2]
                    obs_generated = plot_samples_3[:, channel1] - plot_samples_3[:, channel2]
                    plot_obs(pp=out,
                                 obs_train=obs_train,
                                 obs_test=obs_test,
                                 obs_predict=obs_generated,
                                 name=obs_name,
                                 n_epochs=n_epochs,
                                 n_jets=3,
                                 weight_samples=iterations)

        if get(self.params,"plot_loss", False):
            out = f"{path}/loss_epoch_{n_epochs}.pdf"
            plot_loss(out, self.model.train_losses, self.model.regular_loss, self.model.kl_loss, self.model.regularizeGMM_loss, loss_log=get(self.params, "loss_log", True))
    
    def full_run(self):
        self.prepare_experiment()
        self.load_data()
        self.preprocess_data_njets()
        
        self.model = self.build_model(self.params)
        self.model.obs_names = self.obs_names
        self.model.obs_ranges = self.obs_ranges

        self.build_optimizer()
        self.build_dataloaders()
        self.train_model()
        
        self.generate_samples()
        self.make_plots()
        self.finish_up()

class Conditional_Jet_Dataset(Dataset):
    def __init__(self, data_1, data_2, data_3):
        self.data_1 = data_1
        self.data_2 = data_2
        self.data_3 = data_3
        self.lenghts = [len(self.data_1), len(self.data_2), len(self.data_3)]

        # define placeholders for effective datasets 1, 2
        # (reduced size, they are randomly redefined in each epoch)
        self.data_1_eff = data_1[:self.lenghts[-1]]
        self.data_2_eff = data_2[:self.lenghts[-1]]

    def __len__(self):
        return self.lenghts[-1]

    def __getitem__(self, index):
        x1 = self.data_1_eff[index]
        x2 = self.data_2_eff[index]
        x3 = self.data_3[index]
        return (x1, x2, x3)

class Conditional_Jet_DataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle=True, num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, pre_epoch_callback=None):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory, drop_last=drop_last)
        
    def __iter__(self):
        # recreate effective datasets 1 and 2
        # (so that they have same size as 3)
        #idx1 = torch.randperm(self.dataset.lenghts[0])[:self.dataset.lenghts[-1]]
        #self.dataset.data_1_eff = self.dataset.data_1[idx1]
        #idx2 = torch.randperm(self.dataset.lenghts[1])[:self.dataset.lenghts[-1]]
        #self.dataset.data_2_eff = self.dataset.data_2[idx2]

        # then just return the mother iteraretor
        return super().__iter__()
