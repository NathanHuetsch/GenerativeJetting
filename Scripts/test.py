import numpy as np
import torch
from torch.utils.data import DataLoader
from matplotlib.backends.backend_pdf import PdfPages
from Source.Util.plots import plot_obs, delta_r, plot_deta_dphi
from Source.Util.util import get_device, save_params, get, load_params
import sys
import os
from torch.optim import Adam
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time

from Source.Networks.classifier import Classifier

device = get_device()

print("Loading data")

path_events = "/remote/gpu07/huetsch/data/events_undone.npy"

experiment_path = "/remote/gpu07/huetsch/old_results/test_Attention/6d_TBD_evenSmaller7526"
path_samples = os.path.join(experiment_path, "samples/run0.npy")

events = np.load(path_events)[:1000000]
samples = np.load(path_samples)[:, [8,9,10,12,13,14]]
"""
dphi = events[:, 1] - events[:, 4]
deta = events[:, 2] - events[:, 5]
dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
dR = dphi**2 + deta**2
events = np.concatenate([events, dR[:, None]], axis=1)

dphi = samples[:, 1] - samples[:, 4]
deta = samples[:, 2] - samples[:, 5]
dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
dR = dphi**2 + deta**2
samples = np.concatenate([samples, dR[:, None]], axis=1)
"""
data = np.concatenate([events, samples], axis=0)
target = np.concatenate([np.ones((events.shape[0])), np.zeros((samples.shape[0]))], axis=0)

traindata = torch.from_numpy(np.hstack([data, target[:, None]])).float()
testdata = torch.from_numpy(np.hstack([data, target[:, None]])).float()

print("Building model")

p = {
    "dim": traindata.shape[1]-1,
    "n_layers": 6,
    "intermediate_dim": 128,
    "dropout": None,
    "normalization": None,
    "activation": "ReLU"
}

classifier = Classifier(p).to(device)
print(classifier)

n_epochs = 20
batch_size = 1024
lr = 0.001
betas = (0.5, 0.9)

trainloader = DataLoader(traindata.to(device), batch_size=batch_size, shuffle=True)
optimizer = Adam(classifier.parameters(), lr=lr, betas=betas)

classifier.train()
train_losses = []
print("Starting training")
for e in range(n_epochs):
    t0 = time.time()
    epoch_loss = 0
    for i, batch in enumerate(trainloader):
        optimizer.zero_grad()

        data = batch[:, :-1].float()
        target = batch[:, -1].float()

        prediction = classifier(data)
        loss = F.binary_cross_entropy(prediction, target.unsqueeze(1))
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        epoch_loss += loss.item()
    t1 = time.time()
    print(f"Finished epoch {e} in {round(t1-t0)} seconds with average loss", epoch_loss / len(trainloader))

predictions_events = []
predictions_samples = []

print("Starting predictions")
for i in range(1000):
    event = events[i*1000:i*1000+1000]
    pred = classifier(torch.from_numpy(event).float().to(device)).detach().cpu().squeeze().numpy()
    predictions_events.append(pred)
    sample = samples[i*1000:i*1000+1000]
    pred = classifier(torch.from_numpy(sample).float().to(device)).detach().cpu().squeeze().numpy()
    predictions_samples.append(pred)
predictions_events = np.concatenate(np.array(predictions_events))
predictions_samples = np.concatenate(np.array(predictions_samples))

path_out = os.path.join(experiment_path, "classifier_pred")
os.makedirs(path_out, exist_ok=True)

np.save(os.path.join(path_out, "predictions_events"), predictions_events)
np.save(os.path.join(path_out, "predictions_samples"), predictions_samples)