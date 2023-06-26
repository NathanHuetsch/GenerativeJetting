import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.distributions as D
from matplotlib.backends.backend_pdf import PdfPages
import os, sys
os.chdir("..")
from Source.Models.autoregGMM import AutoRegGMM
from Source.Models.autoregBinned import AutoRegBinned
from Source.Util.simulateToyData import ToySimulator
from Source.Util.util import load_params, get, get_device
from Source.Experiments import toy


path = sys.argv[1]

#for i in range(3, 4):
i=9
path_path = path + f"run_0{i}/"
params = load_params(path_path + "paramfile.yaml")

params["warm_start"] = True
params['train'] = False

params['redirect_console'] = True
params["n_samples"] = 10000000
params['iterations'] = 30
params['plot_loss'] = False
params["plot_sigma"] = True
params['plot_mu_sigma'] = True
params['plot'] = True
params['sigma_path'] = "run"
params["warm_start_path"] = path_path
params["obs_ranges"] = [[0.1, 0.9]]
params["model_name"] = "model_run3"
experiment = toy.Toy_Experiment(params)
experiment.full_run()



