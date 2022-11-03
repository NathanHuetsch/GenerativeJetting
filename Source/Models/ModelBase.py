import numpy as np
import torch
import torch.nn as nn
import math
import os
from torch.utils.tensorboard import SummaryWriter
from Source.Util.util import get, get_device
from Source.Networks.inn_net import build_INN


class GenerativeModel(nn.Module):
    """
    Base Class for Generative Models to inherit from.
    Children classes should overwrite the individual methods as needed.
    Every child class MUST overwrite the methods:

    def build_net(self): should register some NN architecture as self.net
    def batch_loss(self, x): takes a batch of samples as input and returns the loss
    def sample_n_parallel(self, n_samples): generates and returns n_samples new samples

    See tbd.py for an example of child class

    Structure:

    __init__(params)      : Read in parameters and register the important ones
    build_net()           : Create the NN and register it as self.net
                            HAS TO BE OVERWRITTEN IN CHILD CLASS
    prepare_training()    : Read in the appropriate parameters and prepare the model for training
                            Currently this is called from run_training(), so it should not be called on its own
    run_training()        : Run the actual training.
                            Necessary parameters are read in and the training is performed.
                            This calls on the methods train_one_epoch() and validate_one_epoch()
    train_one_epoch()     : Performs one epoch of model training.
                            This calls on the method batch_loss(x)
    validate_one_epoch()  : Performs one epoch of validation.
                            This calls on the method batch_loss(x)
    batch_loss(x)         : Takes one batch of samples as input and returns the loss.
                            HAS TO BE OVERWRITTEN IN CHILD CLASS
    sample_n(n_samples)   : Generates and returns n_samples new samples as a numpy array
                            HAS TO BE OVERWRITTEN IN CHILD CLASS
    save()                : Saves the model and all relevant information. TODO
    load()                : Loads a saved model and all relevant information. TODO
    """
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.device = get(self.params, "device", get_device())
        self.dim = self.params["dim"]
        self.net = self.build_net()

    def build_net(self):
        pass

    def prepare_training(self):
        print("train_model: Preparing model training")
        self.train_losses = np.array([])
        self.train_losses_epoch = np.array([])
        self.n_trainbatches = len(self.train_loader)

        self.validate = get(self.params, "validate", True)
        if self.validate:
            self.n_valbatches = len(self.val_loader)
            self.val_losses_epoch = np.array([])
            self.validate_every = get(self.params, "validate_every", 10)
            self.best_val_loss = get(self.params, "best_val_loss", 1e30)
            self.best_val_epoch = get(self.params, "best_val_loss", 0)
            self.no_improvements = get(self.params, "no_improvements", 0)
            print(f"train_model: validate set to True. Validating every {self.validate_every} epochs")
        else:
            print("train_model: validate set to False. No checkpoints will be created")

        self.log = get(self.params, "log", True)
        if self.log:
            log_dir = get(self.params, "log_dir",
                          os.path.join(self.params["out_dir"], "logs"))
            self.logger = SummaryWriter(log_dir)
            print(f"train_model: Logging to log_dir {log_dir}")
        else:
            print("train_model: log set to False. No logs will be written")

    def run_training(self):

        self.prepare_training()
        n_epochs = get(self.params, "n_epochs", 1000)
        past_epochs = get(self.params, "total_epochs", 0)
        print(f"train_model: Model has been trained for {past_epochs} epochs before.")
        print(f"train_model: Beginning training. n_epochs set to {n_epochs}")
        for e in range(n_epochs):
            self.epoch = past_epochs + e
            self.train()
            self.train_one_epoch()

            if self.validate:
                if (self.epoch + 1) % self.validate_every == 0:
                    self.eval()
                    self.validate_one_epoch()

        if self.validate:
            self.params["no_improvements"] = self.no_improvements
            self.params["best_val_epoch"] = self.best_val_epoch
            self.params["best_val_loss"] = float(self.best_val_loss)

    def train_one_epoch(self):
        train_losses = np.array([])
        for batch_id, x in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            loss = self.batch_loss(x)

            loss_m = self.train_losses[-1000:].mean()
            loss_s = self.train_losses[-1000:].std()

            if np.isfinite(loss.item()) and (abs(loss.item() - loss_m) / loss_s < 5 or len(self.train_losses_epoch) == 0):
                loss.backward()
                self.optimizer.step()
                train_losses = np.append(train_losses, loss.item())
                if self.log:
                    self.logger.add_scalar("train_losses", train_losses[-1], self.epoch*self.n_trainbatches + batch_id)
            else:
                print(f"train_model: Unstable loss. Skipped backprop for epoch {self.epoch}, batch_id {batch_id}")

        self.train_losses_epoch = np.append(self.train_losses_epoch, train_losses.mean())
        self.train_losses = np.concatenate([self.train_losses, train_losses], axis=0)
        if self.log:
            self.logger.add_scalar("train_losses_epoch", self.train_losses_epoch[-1], self.epoch)

    def validate_one_epoch(self):
        val_losses = np.array([])
        for batch_id, x in enumerate(self.val_loader):
            with torch.no_grad():
                loss = self.batch_loss(x)
            val_losses = np.append(val_losses, loss.item())
            if self.log:
                self.logger.add_scalar("val_losses", val_losses[-1],
                                       self.n_valbatches*len(self.val_losses_epoch) + batch_id)
        self.val_losses_epoch = np.append(self.val_losses_epoch, val_losses.mean())
        if self.log:
            self.logger.add_scalar("val_losses_epoch", self.val_losses_epoch[-1], len(self.val_losses_epoch))
        print(f"train_model: Validated after epoch {self.epoch}. Current val_loss is {val_losses.mean()}")
        if val_losses.mean() < self.best_val_loss:
            self.no_improvements = 0
            self.best_val_epoch = self.epoch
            self.best_val_loss = val_losses.mean()
            torch.save(self.state_dict(), "models/checkpoint.pt")
        else:
            self.no_improvements = self.no_improvements + 1
            print(f"train_model: val_loss has not improved for {self.no_improvements} consecutive validations")

    def batch_loss(self, x):
        pass

    def sample_n(self, n_samples):
        pass

    def save(self, epoch=""):
        # Deprecated TOD0
        os.makedirs(self.doc.get_file("model", False), exist_ok=True)
        torch.save({"opt": self.optim.state_dict(),
                    "net": self.net.state_dict(),
                    "epoch": self.epoch}, self.doc.get_file(f"model/model{epoch}", False))

    def load(self, epoch=""):
        # Deprecated TOD0
        name = self.doc.get_file(f"model/model{epoch}", False)
        state_dicts = torch.load(name, map_location=self.device)
        self.net.load_state_dict(state_dicts["net"])

        try:
            self.epoch = state_dicts["epoch"]
        except:
            self.epoch = 0
            print(f"Warning: Epoch number not provided in save file, setting to {self.epoch}")
        try:
            self.optim.load_state_dict(state_dicts["opt"])
        except ValueError as e:
            print(e)
        self.net.to(self.device)