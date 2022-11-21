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

events = np.load("/remote/gpu07/huetsch/data/events_undone.npy")[:1000000]
samples = np.load("/remote/gpu07/huetsch/data/samples_undone.npy")

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

preds_events = np.load("/remote/gpu07/huetsch/data/classifier_predictions_events.npy")
preds_samples = np.load("/remote/gpu07/huetsch/data/classifier_predictions_samples.npy")

weights = preds_samples/(1-preds_samples)

fig = plt.figure(figsize=(12, 6))
r = [0.5, 2]
y_weights, x_weights = np.histogram(weights, bins=100, range=r, density=True)
y_weights = y_weights/np.sum(y_weights)
plt.step(x_weights[:100], y_weights, label="Weights")
plt.legend()
plt.yscale("log")
plt.title("Weights")
plt.savefig("WeightHisto")

fig.add_subplot(1, 3, 1)
dphi = events[:, 1] - events[:, 4]
deta = events[:, 2] - events[:, 5]
dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
plt.hist2d(deta, dphi, bins=100, range=[[-5, 5], [-np.pi, np.pi]])
plt.xlabel('$\Delta \eta$')
plt.ylabel('$\Delta \phi$')
plt.title('GT events')

fig.add_subplot(1, 3, 2)
dphi = samples[:, 1] - samples[:, 4]
deta = samples[:, 2] - samples[:, 5]
dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
plt.hist2d(deta, dphi, bins=100, range=[[-5, 5], [-np.pi, np.pi]])
plt.xlabel('$\Delta \eta$')
plt.ylabel('$\Delta \phi$')
plt.title('Samples')

fig.add_subplot(1, 3, 3)
plt.hist2d(deta, dphi, bins=100, weights=weights, range=[[-5, 5], [-np.pi, np.pi]])
plt.xlabel('$\Delta \eta$')
plt.ylabel('$\Delta \phi$')
plt.title('Weighted')
plt.savefig("plot.png")
plt.close()

i=6
r = [-1, 8]
y_e, x_e = np.histogram(events[:, i], bins=60, range=r, density=True)
y_s, x_s = np.histogram(samples[:, i], bins=60, range=r, density=True)
y_w, x_w = np.histogram(samples[:, i], bins=60, range=r, density=True, weights=weights)
y_e, y_s, y_w = y_e/np.sum(y_e), y_s/np.sum(y_s), y_w/np.sum(y_w)
plt.step(x_e[:60], y_e, label="events")
plt.step(x_s[:60], y_s, label="samples")
plt.step(x_w[:60], y_w, label="Weighted")
plt.legend()
plt.title(f"dim {i}")
plt.savefig("histo")