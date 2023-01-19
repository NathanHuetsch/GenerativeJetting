import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from matplotlib.backends.backend_pdf import PdfPages
from Source.Util.plots import plot_obs, delta_r, plot_deta_dphi
from Source.Util.preprocessing import preprocess, undo_preprocessing
from Source.Util.util import get_device, save_params, get, load_params
from Source.Experiments.ExperimentBase import Experiment
import torch.nn.functional as F
import Source.Networks
import time
from datetime import datetime
import sys
import os
from torch.optim import Adam
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


class Classifier_Experiment():
    def __init__(self, params):
        #super().__init__(params)
        self.params = params
        self.device = get(self.params, "device", get_device())

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

        self.starttime = time.time()

    def prepare_experiment(self):
        self.experiment_dir = get(self.params, "experiment_dir", None)
        if self.experiment_dir is None:
            raise ValueError("prepare_experiment: experiment_dir not specified")
        run_name = get(self.params, "run_name", "Classifier") + str(np.random.randint(low=1000, high=9999))
        self.out_dir = os.path.join(self.experiment_dir, run_name)
        os.makedirs(self.out_dir)
        os.chdir(self.out_dir)
        self.params["out_dir"] = self.out_dir
        save_params(self.params, "paramfile.yaml")

        experiment_params_path = os.path.join(self.experiment_dir, "paramfile.yaml")
        self.experiment_params = load_params(experiment_params_path)

        if get(self.params, "redirect_console", True):
            sys.stdout = open("stdout.txt", "w", buffering=1)
            sys.stderr = open("stderr.txt", "w", buffering=1)
            print(f"prepare_experiment: Redirecting console output to out_dir")

        print("prepare_experiment: Using out_dir ", self.out_dir)
        print(f"prepare_experiment: Using device {self.device}")

    def load_data(self):
        data_path = get(self.params, "data_path", "/remote/gpu07/huetsch/data/Z_2.npy")
        assert os.path.exists(data_path), f"load_data_true: data_path {data_path} does not exist"
        try:
            self.data_true_raw = np.load(data_path)
            print(f"load_data_true: Loaded data with shape {self.data_true_raw.shape} from ", data_path)
        except Exception:
            raise ValueError(f"load_data_true: Cannot load data from {data_path}")

        samples_path = get(self.params, "samples_path", None)
        if samples_path is None:
            samples_path = os.path.join(self.experiment_dir, "samples/run0.npy")
        assert os.path.exists(samples_path), f"load_data_generated: data_path {samples_path} does not exist"

        try:
            self.data_generated_raw = np.load(samples_path)
            mask = np.where(self.data_generated_raw > 10 ** 10)[0]
            self.data_generated_raw = np.delete(self.data_generated_raw, mask, axis=0)
            print(f"load_data_generated: Loaded data with shape {self.data_generated_raw.shape} from ", samples_path)
        except Exception:
            raise ValueError(f"load_data_generated: Cannot load data from {samples_path}")

    def preprocess_data(self):

        self.channels = get(self.experiment_params, "channels", None)
        self.preprocess = get(self.params, "preprocess", True)
        if self.channels is None:
            print("preprocess_data: channels and dim not specified. Defaulting to 13 channels")
            self.channels = [0, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
        else:
            print(f"preprocess_data: channels {self.channels} specified")
        self.dim = len(self.channels)
        self.params["dim"] = self.dim
        self.params["channels"] = self.channels


        self.data_true, self.data_mean, self.data_std, self.data_u, self.data_s \
            = preprocess(self.data_true_raw, self.channels)
        self.data_true_raw = undo_preprocessing(self.data_true, self.data_mean, self.data_std,
                                                self.data_u, self.data_s, self.channels, keep_all=True)

        #self.data_generated, a, b, c, d \
        #    = preprocess(self.data_generated_raw, self.channels)
        #self.data_generated_raw = undo_preprocessing(self.data_generated, self.data_mean, self.data_std,
        #                                             self.data_u, self.data_s, self.channels, keep_all=True)


        if not self.preprocess:
            self.data_true = self.data_true_raw[:, self.channels]
            self.data_generated = self.data_generated_raw[:, self.channels]



        self.add_dR = get(self.params, "add_dR", False)
        if self.add_dR:
            #Unfinished !!
            dR = delta_r(self.data_true_raw)
            self.data_true = np.concatenate([self.data_true, dR[:, None]], axis=1)
            self.data_true_raw = np.concatenate([self.data_true_raw, dR[:, None]], axis=1)

            dR = delta_r(self.data_generated_raw)
            self.data_generated = np.concatenate([self.data_generated, dR[:, None]], axis=1)
            self.data_generated_raw = np.concatenate([self.data_generated_raw, dR[:, None]], axis=1)

            self.dim = self.dim + 1
            self.params["dim"] = self.dim

    def prepare_training(self):

        network = get(self.params, "network", "Classifier")
        self.network = getattr(Source.Networks, network)(self.params).to(self.device)
        print(f"prepare_training: Built network {network}")

        lr = get(self.params, "lr", 0.0001)
        betas = get(self.params, "betas", [0.9, 0.99])
        weight_decay = get(self.params, "weight_decay", 0)
        self.optimizer = \
            Adam(self.network.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        print(f"prepare_training: Built optimizer Adam with lr {lr}, betas {betas}, weight_decay {weight_decay}")

        self.data_split = get(self.params, "data_split", 0.8)
        self.n_data_true_train = int(len(self.data_true)*self.data_split)
        self.n_data_generated_train = int(len(self.data_generated)*self.data_split)
        self.n_data_true_test = len(self.data_true) - self.n_data_true_train
        self.n_data_generated_test = len(self.data_generated) - self.n_data_generated_train

        train_events = np.concatenate([self.data_true[:self.n_data_true_train],
                                       self.data_generated[:self.n_data_generated_train]], axis=0)
        train_target = np.concatenate([np.ones(self.n_data_true_train),
                                       np.zeros(self.n_data_generated_train)], axis=0)
        train_data = np.concatenate([train_events, train_target[:, None]], axis=1)
        self.train_data = torch.from_numpy(train_data).float().to(self.device)

        test_events = np.concatenate([self.data_true[self.n_data_true_train:],
                                      self.data_generated[self.n_data_generated_train:]], axis=0)
        test_target = np.concatenate([np.ones(self.n_data_true_test),
                                      np.zeros(self.n_data_generated_test)], axis=0)
        test_data = np.concatenate([test_events, test_target[:, None]], axis=1)
        self.test_data = torch.from_numpy(test_data).float().to(self.device)

        self.batch_size = get(self.params, "batch_size", 1024)
        self.train_loader = \
            DataLoader(dataset=self.train_data,
                       batch_size=self.batch_size,
                       shuffle=True)
        self.test_loader = \
            DataLoader(dataset=self.test_data,
                       batch_size=self.batch_size,
                       shuffle=True)
        print(f"prepare_training: Built dataloaders")
    def train_model(self):

        log_dir = os.path.join(self.params["out_dir"], "logs")
        self.logger = SummaryWriter(log_dir)
        print(f"train_model: Logging to log_dir {log_dir}")

        self.n_epochs = get(self.params, "n_epochs", 100)
        T0 = time.time()
        self.network.train()

        l_trainloader = len(self.train_loader)
        print(f"train_model: Beginning training for {self.n_epochs} epochs")
        for e in range(self.n_epochs):
            train_losses = np.array([])
            t0 = time.time()
            epoch_loss = 0
            for i, batch in enumerate(self.train_loader):

                self.optimizer.zero_grad()

                data = batch[:, :-1]
                target = batch[:, -1]

                prediction = self.network(data)
                loss = F.binary_cross_entropy(prediction, target.unsqueeze(1))
                loss.backward()
                self.optimizer.step()
                train_losses = np.append(train_losses, loss.item())

                self.logger.add_scalar("train_losses", train_losses[-1], e*l_trainloader + i)
                epoch_loss += loss.item()
            t1 = time.time()
            self.logger.add_scalar("train_losses_epoch", epoch_loss/l_trainloader, e)
            print(f"Finished epoch {e} in {round(t1 - t0)} seconds with average loss", epoch_loss / l_trainloader)

        T1 = time.time()
        traintime = round(T1-T0)
        self.params["traintime"] = traintime
        torch.save(self.network.state_dict(), "classifier.pt")
        print(f"train_model: Finished training for {self.n_epochs} epochs after {traintime} seconds.")

    def predict(self):
        self.true_predictions = np.zeros(len(self.data_true))
        self.generated_predictions = np.zeros(len(self.data_generated))
        self.network.eval()
        with torch.no_grad():
            print("predict: Beginning predictions for true events")
            events = self.data_true
            n_batches = int(len(events)/self.batch_size)
            for i in range(n_batches):
                batch = torch.from_numpy(events[i*self.batch_size:(i+1)*self.batch_size])
                prediction = self.network(batch.float().to(self.device))
                self.true_predictions[i*self.batch_size:(i+1)*self.batch_size]=prediction.cpu().squeeze().numpy()
            batch = torch.from_numpy(events[n_batches*self.batch_size:])
            print(batch.shape)
            prediction = self.network(batch.float().to(self.device))
            self.true_predictions[n_batches*self.batch_size:]=prediction.cpu().squeeze().numpy()

            print("predict: Beginning predictions for generated events")
            events = self.data_generated
            n_batches = int(len(events)/self.batch_size)
            for i in range(n_batches):
                batch = torch.from_numpy(events[i*self.batch_size:(i+1)*self.batch_size])
                prediction = self.network(batch.float().to(self.device))
                self.generated_predictions[i*self.batch_size:(i+1)*self.batch_size]=prediction.cpu().squeeze().numpy()
            batch = torch.from_numpy(events[n_batches*self.batch_size:])
            print(batch.shape)
            prediction = self.network(batch.float().to(self.device))
            self.generated_predictions[n_batches*self.batch_size:]=prediction.cpu().squeeze().numpy()
            """
            print("predict: Beginning predictions for true test events")
            events = self.test_data[:self.n_data_true_test]
            n_batches = int(len(events) / self.batch_size)
            for i in range(n_batches):
                batch = events[i * self.batch_size:i * self.batch_size + self.batch_size]
                data = batch[:, :-1]
                prediction = self.network(data)
                self.true_test_predictions.append(prediction.cpu().squeeze().numpy())
            batch = events[n_batches*self.batch_size:]
            data = batch[:, :-1]
            prediction = self.network(data)
            self.true_test_predictions.append(prediction.cpu().squeeze().numpy())

            print("predict: Beginning predictions for generated test events")
            events = self.test_data[self.n_data_true_test:]
            n_batches = int(len(events) / self.batch_size)
            for i in range(n_batches):
                batch = events[i * self.batch_size:i * self.batch_size + self.batch_size]
                data = batch[:, :-1]
                prediction = self.network(data)
                self.generated_test_predictions.append(prediction.cpu().squeeze().numpy())
            batch = events[n_batches*self.batch_size:]
            data = batch[:, :-1]
            prediction = self.network(data)
            self.generated_test_predictions.append(prediction.cpu().squeeze().numpy())

            print("predict: Beginning predictions for test events")
            for i, batch in enumerate(self.test_loader):
                data = batch[:, :-1].float()
                prediction = self.network(data)

                target = batch[:, -1].cpu().numpy()
                true_ind = np.where(target == 1)[0]
                generated_ind = np.where(target == 0)[0]

                self.true_test_predictions.append(prediction[true_ind].cpu().squeeze().numpy())
                self.generated_test_predictions.append(prediction[generated_ind].cpu().squeeze().numpy())
            
        self.true_train_predictions = np.concatenate(self.true_train_predictions, axis=0)
        self.true_test_predictions = np.concatenate(self.true_test_predictions, axis=0)
        self.generated_train_predictions = np.concatenate(self.generated_train_predictions, axis=0)
        self.generated_test_predictions = np.concatenate(self.generated_test_predictions, axis=0)
        print("pred shapes: ", self.true_train_predictions.shape, self.true_test_predictions.shape,
              self.generated_train_predictions.shape, self.generated_test_predictions.shape)
                """
        self.true_train_predictions = self.true_predictions[:self.n_data_true_train]
        self.true_test_predictions = self.true_predictions[self.n_data_true_train:]
        self.generated_train_predictions = self.generated_predictions[:self.n_data_generated_train]
        self.generated_test_predictions = self.generated_predictions[self.n_data_generated_train:]

        print("predict: Saving predictions")
        np.save("true_train_predictions", self.true_train_predictions)
        np.save("true_test_predictions", self.true_test_predictions)
        np.save("generated_train_predictions", self.generated_train_predictions)
        np.save("generated_test_predictions", self.generated_test_predictions)

        np.save("true_predictions", self.true_predictions)
        np.save("generated_predictions", self.generated_predictions)
    def make_plots(self):
        os.makedirs("plots", exist_ok=True)
        print("make_plots: Drawing ClassifierPrediction plots")
        with PdfPages(f"plots/LogClassifierPredictions") as out:
            plt.hist(self.true_test_predictions, range=[0, 1], bins=100, density=True)
            plt.yscale("log")
            plt.title("LogHistogram ClassifierPredictions TrueData Testset")
            plt.xlabel("Probability")
            plt.ylabel("Hist")
            out.savefig(bbox_inches='tight', pad_inches=0.05)
            plt.close()

            plt.hist(self.generated_test_predictions, range=[0, 1], bins=100, density=True)
            plt.yscale("log")
            plt.title("LogHistogram ClassifierPredictions GeneratedData Testset")
            plt.xlabel("Probability")
            plt.ylabel("Hist")
            out.savefig(bbox_inches='tight', pad_inches=0.05)
            plt.close()

            plt.hist(self.true_train_predictions, range=[0, 1], bins=100, density=True)
            plt.yscale("log")
            plt.title("LogHistogram ClassifierPredictions TrueData Trainset")
            plt.xlabel("Probability")
            plt.ylabel("Hist")
            out.savefig(bbox_inches='tight', pad_inches=0.05)
            plt.close()

            plt.hist(self.generated_train_predictions, range=[0, 1], bins=100, density=True)
            plt.yscale("log")
            plt.title("LogHistogram ClassifierPredictions GeneratedData Trainset")
            plt.xlabel("Probability")
            plt.ylabel("Hist")
            out.savefig(bbox_inches='tight', pad_inches=0.05)
            plt.close()

            plt.hist(self.true_train_predictions, range=[0, 1], bins=100, density=True, color="red", label="True", alpha=0.5)
            plt.hist(self.generated_train_predictions, range=[0, 1], bins=100, density=True, color="blue", label="Generated", alpha=0.5)
            plt.yscale("log")
            plt.legend()
            plt.title("LogHistogram ClassifierPredictions Trainset")
            plt.xlabel("Probability")
            plt.ylabel("Hist")
            out.savefig(bbox_inches='tight', pad_inches=0.05)
            plt.close()

            plt.hist(self.true_test_predictions, range=[0, 1], bins=100, density=True, color="red", label="True", alpha=0.5)
            plt.hist(self.generated_test_predictions, range=[0, 1], bins=100, density=True, color="blue", label="Generated", alpha=0.5)
            plt.yscale("log")
            plt.legend()
            plt.title("LogHistogram ClassifierPredictions Testset")
            plt.xlabel("Probability")
            plt.ylabel("Hist")
            out.savefig(bbox_inches='tight', pad_inches=0.05)
            plt.close()

        with PdfPages(f"plots/ClassifierPredictions") as out:
            plt.hist(self.true_test_predictions, range=[0, 1], bins=100, density=True)
            plt.title("Histogram ClassifierPredictions TrueData Testset")
            plt.xlabel("Probability")
            plt.ylabel("Hist")
            out.savefig(bbox_inches='tight', pad_inches=0.05)
            plt.close()

            plt.hist(self.generated_test_predictions, range=[0, 1], bins=100, density=True)
            plt.title("Histogram ClassifierPredictions GeneratedData Testset")
            plt.xlabel("Probability")
            plt.ylabel("Hist")
            out.savefig(bbox_inches='tight', pad_inches=0.05)
            plt.close()

            plt.hist(self.true_train_predictions, range=[0, 1], bins=100, density=True)
            plt.title("Histogram ClassifierPredictions TrueData Trainset")
            plt.xlabel("Probability")
            plt.ylabel("Hist")
            out.savefig(bbox_inches='tight', pad_inches=0.05)
            plt.close()

            plt.hist(self.generated_train_predictions, range=[0, 1], bins=100, density=True)
            plt.title("Histogram ClassifierPredictions GeneratedData Trainset")
            plt.xlabel("Probability")
            plt.ylabel("Hist")
            out.savefig(bbox_inches='tight', pad_inches=0.05)
            plt.close()

            plt.hist(self.true_train_predictions, range=[0, 1], bins=100, density=True, color="red", label="True", alpha=0.5)
            plt.hist(self.generated_train_predictions, range=[0, 1], bins=100, density=True, color="blue", label="Generated", alpha=0.5)
            plt.legend()
            plt.title("Histogram ClassifierPredictions Trainset")
            plt.xlabel("Probability")
            plt.ylabel("Hist")
            out.savefig(bbox_inches='tight', pad_inches=0.05)
            plt.close()

            plt.hist(self.true_test_predictions, range=[0, 1], bins=100, density=True, color="red", label="True", alpha=0.5)
            plt.hist(self.generated_test_predictions, range=[0, 1], bins=100, density=True, color="blue", label="Generated", alpha=0.5)
            plt.legend()
            plt.title("Histogram ClassifierPredictions Testset")
            plt.xlabel("Probability")
            plt.ylabel("Hist")
            out.savefig(bbox_inches='tight', pad_inches=0.05)
            plt.close()

        self.true_weights = np.clip(self.true_predictions/(1-self.true_predictions), 0, 4)
        self.generated_weights = np.clip(self.generated_predictions/(1-self.generated_predictions), 0, 4)

        self.true_train_weights = self.true_weights[:self.n_data_true_train]
        self.true_test_weights = self.true_weights[self.n_data_true_train:]
        self.generated_train_weights = self.generated_weights[:self.n_data_generated_train]
        self.generated_test_weights = self.generated_weights[self.n_data_generated_train:]

        print("make_plots: Drawing ClassifierWeights plots")
        with PdfPages(f"plots/LogClassifierWeights") as out:
            plt.hist(self.true_test_weights, range=[0, 4], bins=100, density=True)
            plt.yscale("log")
            plt.title("LogHistogram ClassifierWeights TrueData Testset")
            plt.xlabel("Weights")
            plt.ylabel("Hist")
            out.savefig(bbox_inches='tight', pad_inches=0.05)
            plt.close()

            plt.hist(self.generated_test_weights, range=[0, 4], bins=100, density=True)
            plt.yscale("log")
            plt.title("LogHistogram ClassifierWeights GeneratedData Testset")
            plt.xlabel("Weights")
            plt.ylabel("Hist")
            out.savefig(bbox_inches='tight', pad_inches=0.05)
            plt.close()

            plt.hist(self.true_train_weights, range=[0, 4], bins=100, density=True)
            plt.yscale("log")
            plt.title("LogHistogram ClassifierWeights TrueData Trainset")
            plt.xlabel("Weights")
            plt.ylabel("Hist")
            out.savefig(bbox_inches='tight', pad_inches=0.05)
            plt.close()

            plt.hist(self.generated_train_weights, range=[0, 4], bins=100, density=True)
            plt.yscale("log")
            plt.title("LogHistogram ClassifierWeights GeneratedData Trainset")
            plt.xlabel("Weights")
            plt.ylabel("Hist")
            out.savefig(bbox_inches='tight', pad_inches=0.05)
            plt.close()

            plt.hist(self.true_train_weights, range=[0, 4], bins=100, density=True, color="red", label="True", alpha=0.5)
            plt.hist(self.generated_train_weights, range=[0, 4], bins=100, density=True, color="blue", label="Generated",
                     alpha=0.5)
            plt.yscale("log")
            plt.legend()
            plt.title("LogHistogram ClassifierWeights Trainset")
            plt.xlabel("Weights")
            plt.ylabel("Hist")
            out.savefig(bbox_inches='tight', pad_inches=0.05)
            plt.close()

            plt.hist(self.true_test_weights, range=[0, 4], bins=100, density=True, color="red", label="True", alpha=0.5)
            plt.hist(self.generated_test_weights, range=[0, 4], bins=100, density=True, color="blue", label="Generated",
                     alpha=0.5)
            plt.yscale("log")
            plt.legend()
            plt.title("LogHistogram ClassifierWeights Testset")
            plt.xlabel("Weights")
            plt.ylabel("Hist")
            out.savefig(bbox_inches='tight', pad_inches=0.05)
            plt.close()

        with PdfPages(f"plots/ClassifierWeights") as out:
            plt.hist(self.true_test_weights, range=[0, 4], bins=100, density=True)
            plt.title("Histogram ClassifierWeights TrueData Testset")
            plt.xlabel("Weights")
            plt.ylabel("Hist")
            out.savefig(bbox_inches='tight', pad_inches=0.05)
            plt.close()

            plt.hist(self.generated_test_weights, range=[0, 4], bins=100, density=True)
            plt.title("Histogram ClassifierWeights GeneratedData Testset")
            plt.xlabel("Weights")
            plt.ylabel("Hist")
            out.savefig(bbox_inches='tight', pad_inches=0.05)
            plt.close()

            plt.hist(self.true_train_weights, range=[0, 4], bins=100, density=True)
            plt.title("Histogram ClassifierWeights TrueData Trainset")
            plt.xlabel("Weights")
            plt.ylabel("Hist")
            out.savefig(bbox_inches='tight', pad_inches=0.05)
            plt.close()

            plt.hist(self.generated_train_weights, range=[0, 4], bins=100, density=True)
            plt.title("Histogram ClassifierWeights GeneratedData Trainset")
            plt.xlabel("Weights")
            plt.ylabel("Hist")
            out.savefig(bbox_inches='tight', pad_inches=0.05)
            plt.close()

            plt.hist(self.true_train_weights, range=[0, 4], bins=100, density=True, color="red", label="True", alpha=0.5)
            plt.hist(self.generated_train_weights, range=[0, 4], bins=100, density=True, color="blue", label="Generated",
                     alpha=0.5)
            plt.legend()
            plt.title("Histogram ClassifierWeights Trainset")
            plt.xlabel("Weights")
            plt.ylabel("Hist")
            out.savefig(bbox_inches='tight', pad_inches=0.05)
            plt.close()

            plt.hist(self.true_test_weights, range=[0, 4], bins=100, density=True, color="red", label="True", alpha=0.5)
            plt.hist(self.generated_test_weights, range=[0, 4], bins=100, density=True, color="blue", label="Generated",
                     alpha=0.5)
            plt.legend()
            plt.title("Histogram ClassifierWeights Testset")
            plt.xlabel("Weights")
            plt.ylabel("Hist")
            out.savefig(bbox_inches='tight', pad_inches=0.05)
            plt.close()


        experiment_data_split = self.experiment_params["data_split"]
        cut = int(len(self.data_true_raw) * (experiment_data_split[0] + experiment_data_split[1]))
        self.data_train_raw = self.data_true_raw
        self.data_train = self.data_true

        print("make_plots: Drawing histograms")
        with PdfPages(f"plots/1d_histograms") as out:
            # Loop over the plot_channels
            for i, channel in enumerate(self.channels):
                obs_train = self.data_train_raw[:cut, channel]
                obs_test = self.data_true_raw[cut:, channel]
                obs_generated = self.data_generated_raw[:, channel]
                # Get the name and the range of the observable
                obs_name = self.obs_names[channel]
                obs_range = self.obs_ranges[channel]
                # Create the plot
                plot_obs(pp=out,
                         obs_train=obs_train,
                         obs_test=obs_test,
                         obs_predict=obs_generated,
                         name=obs_name,
                         range=obs_range, n_epochs=self.n_epochs)
            if self.add_dR:
                obs_train = self.data_train_raw[:cut, -1]
                obs_test = self.data_true_raw[cut:, -1]
                obs_generated = self.data_generated_raw[:, -1]
                # Get the name and the range of the observable
                obs_name = "dR"
                obs_range = [0,8]
                # Create the plot
                plot_obs(pp=out,
                         obs_train=obs_train,
                         obs_test=obs_test,
                         obs_predict=obs_generated,
                         name=obs_name,
                         range=obs_range, n_epochs=self.n_epochs)

        with PdfPages(f"plots/1d_histograms_preprocessed") as out:
            # Loop over the plot_channels
            for i, channel in enumerate(range(6)):
                obs_train = self.data_train[:cut, channel]
                obs_test = self.data_true[cut:, channel]
                obs_generated = self.data_generated[:, channel]
                # Get the name and the range of the observable
                obs_name = self.obs_names[channel]
                obs_range = [-3, 3]
                # Create the plot
                plot_obs(pp=out,
                         obs_train=obs_train,
                         obs_test=obs_test,
                         obs_predict=obs_generated,
                         name=obs_name,
                         range=obs_range, n_epochs=self.n_epochs)


        with PdfPages(f"plots/1d_histograms_reweighted") as out:
            # Loop over the plot_channels
            # weights = np.concatenate([self.generated_train_weights, self.generated_test_weights], axis=0)
            weights = self.generated_weights
            for i, channel in enumerate(self.channels):
                obs_train = self.data_train_raw[:cut, channel]
                obs_test = self.data_true_raw[cut:, channel]
                obs_generated = self.data_generated_raw[:, channel]
                obs_name = self.obs_names[channel]
                obs_range = self.obs_ranges[channel]
                plot_obs(pp=out,
                         obs_train=obs_train,
                         obs_test=obs_test,
                         obs_predict=obs_generated,
                         weights=weights,
                         name=obs_name,
                         range=obs_range, n_epochs=self.n_epochs)
            if self.add_dR:
                obs_train = self.data_train_raw[:cut, -1]
                obs_test = self.data_true_raw[cut:, -1]
                obs_generated = self.data_generated_raw[:, channel]
                obs_name = "dR"
                obs_range = [0,8]
                plot_obs(pp=out,
                         obs_train=obs_train,
                         obs_test=obs_test,
                         obs_predict=obs_generated,
                         weights=weights,
                         name=obs_name,
                         range=obs_range, n_epochs=self.n_epochs)


    def finish_up(self):
        self.params["datetime"] = str(datetime.now())
        self.params["experimenttime"] = time.time() - self.starttime
        save_params(self.params, "paramfile.yaml")
        print("finish_up: Finished experiment.")

        if get(self.params, "redirect_console", True):
            sys.stdout.close()
            sys.stderr.close()

    def full_run(self):
        self.prepare_experiment()
        self.load_data()
        self.preprocess_data()
        self.prepare_training()
        self.train_model()
        self.predict()
        self.make_plots()
        self.finish_up()
