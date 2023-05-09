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

for i in range(0,10):
    path_path = path + f"DDPM_base_{i}/"
    params = load_params(path_path + "paramfile.yaml")

    params["warm_start"] = True
    params['train'] = False

    params['redirect_console'] = False
    params['iterations'] = 30
    params['plot_loss'] = False
    params["plot_sigma"] = False
    params['plot_mu_sigma'] = False
    params['plot'] = True
    params['sigma_path'] = "run"
    params["warm_start_path"] = path_path
    experiment = toy.Toy_Experiment(params)
    experiment.full_run()



