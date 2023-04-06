import numpy as np
import torch
from torch.utils.data import DataLoader
from matplotlib.backends.backend_pdf import PdfPages
from Source.Util.plots import plot_obs, delta_r, plot_deta_dphi
from Source.Util.util import get_device, save_params, get, load_params
from Source.Util.preprocessing import preprocess, undo_preprocessing
import sys
import os
from torch.optim import Adam
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time

from Source.Networks.classifier import Classifier

device = get_device()

obs_names = ["p_{T,l1}", "\phi_{l1}", "\eta_{l1}", "\mu_{l1}",
                  "p_{T,l2}", "\phi_{l2}", "\eta_{l2}", "\mu_{l2}",
                  "p_{T,j1}", "\phi_{j1}", "\eta_{j1}", "\mu_{j1}",
                  "p_{T,j2}", "\phi_{j2}", "\eta_{j2}", "\mu_{j2}"]

# Ranges of the observables for plotting
obs_ranges = [[0, 150], [-4, 4], [-6, 6], [0, 100],
                   [0, 150], [-4, 4], [-6, 6], [0, 100],
                   [0, 150], [-4, 4], [-6, 6], [0, 100],
                   [0, 150], [-4, 4], [-6, 6], [0, 100]]

path = "/remote/gpu07/huetsch/old_results/test_Attention/6d_TBD_evenSmaller7526/Classifier_base_dR3083/test"
os.makedirs(path, exist_ok=True)

true_preds_path = "/remote/gpu07/huetsch/old_results/test_Attention/6d_TBD_evenSmaller7526/Classifier_base_dR3083/true_predictions.npy"
gen_preds_path = "/remote/gpu07/huetsch/old_results/test_Attention/6d_TBD_evenSmaller7526/Classifier_base_dR3083/generated_predictions.npy"

true_preds = np.load(true_preds_path)
gen_preds = np.load(gen_preds_path)

experiment_path = "/remote/gpu07/huetsch/old_results/test_Attention/6d_TBD_evenSmaller7526"
path_data_true = "/remote/gpu07/huetsch/data/Z_2.npy"
path_data_generated = os.path.join(experiment_path, "samples/run0.npy")

data_true_raw = np.load(path_data_true)
data_generated_raw = np.load(path_data_generated)

channels = [8, 9, 10, 12, 13, 14]

data_true, data_mean, data_std, data_u, data_s \
    = preprocess(data_true_raw, channels)
data_true_raw = undo_preprocessing(data_true, data_mean, data_std,
                                        data_u, data_s, channels, keep_all=False)

data_generated, data_mean, data_std, data_u, data_s \
    = preprocess(data_generated_raw, channels, convert=False)
data_generated_raw = undo_preprocessing(data_generated, data_mean, data_std,
                                             data_u, data_s, channels, keep_all=False)

#data_true = data_true_raw[:, channels]
#data_generated = data_generated_raw[:, channels]
"""
dphi = data_true[:, 1] - data_true[:, 4]
deta = data_true[:, 2] - data_true[:, 5]
dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
dR = dphi**2 + deta**2
data_true = np.concatenate([data_true, dR[:, None]], axis=1)

dphi = data_generated[:, 1] - data_generated[:, 4]
deta = data_generated[:, 2] - data_generated[:, 5]
dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
dR = dphi**2 + deta**2
data_generated = np.concatenate([data_generated, dR[:, None]], axis=1)
"""
plt.hist(true_preds, range=[0, 1], bins=100, density=True, color="red", label="True", alpha=0.5)
plt.hist(gen_preds, range=[0, 1], bins=100, density=True, color="blue", label="Generated", alpha=0.5)
#plt.yscale("log")
plt.legend()
plt.title("Classifier Predictions")
plt.xlabel("Probability")
plt.ylabel("Hist")
plt.savefig("preds_histo.png" ,bbox_inches='tight', pad_inches=0.05)
plt.close()

true_preds_09_ind = np.where(true_preds>0.9)[0]
gen_preds_01_ind = np.where(gen_preds<0.1)[0]

data_true_preds_09 = data_true[true_preds_09_ind]
data_generated_preds_01 = data_generated[gen_preds_01_ind]

data_true_raw_preds_09 = data_true_raw[true_preds_09_ind]
data_generated_raw_preds_01 = data_generated_raw[gen_preds_01_ind]

"""
for i in range(7):
    plt.hist(data_true_preds_09[:, i], bins=100, density=True, color="red", label="0.9", alpha=0.5)
    plt.hist(data_true[:, i], bins=100, density=True, color="blue", label="All", alpha=0.5)
    plt.savefig(f"data_true09_dim{i}.png")
    plt.legend()
    plt.close()

for i in range(7):
    plt.hist(data_generated_preds_01[:, i], bins=100, density=True, color="red", label="0.9", alpha=0.5)
    plt.hist(data_generated[:, i], bins=100, density=True, color="blue", label="All", alpha=0.5)
    plt.savefig(f"data_generated01_dim{i}.png")
    plt.legend()
    plt.close()
"""
c = [8,9,10,12,13,14]
for i, channel in enumerate(range(6)):
    obs_train = data_true_preds_09[:, channel]
    obs_test = data_true[:, channel]
    obs_generated = data_generated[:, channel]
    # Get the name and the range of the observable
    obs_name = obs_names[c[channel]]
    obs_range = obs_ranges[c[channel]]
    # Create the plot
    plot_obs(pp=f"data_true09_dim{i}.png",
             obs_train=obs_train,
             obs_test=obs_test,
             obs_predict=obs_generated,
             name=obs_name,
             range=obs_range)

for i, channel in enumerate(range(6)):
    obs_train = data_true_raw_preds_09[:, channel]
    obs_test = data_true_raw[:, channel]
    obs_generated = data_generated_raw[:, channel]
    # Get the name and the range of the observable
    obs_name = obs_names[c[channel]]
    obs_range = obs_ranges[c[channel]]
    # Create the plot
    plot_obs(pp=f"data_true_raw09_dim{i}.png",
             obs_train=obs_train,
             obs_test=obs_test,
             obs_predict=obs_generated,
             name=obs_name,
             range=obs_range)