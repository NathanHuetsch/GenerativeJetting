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


params = load_params(path + "paramfile.yaml")

params["warm_start"] = True
params["warm_start_path"] = path
params['train'] = False

params['redirect_console'] = False
params['iterations'] = 30
params['plot_loss'] = False
params["plot_sigma"] = False
params['plot_mu_sigma'] = False
params['plot'] = False
params['sigma_path'] = '/remote/gpu05/palacios/GenerativeJetting/runs/toy/Ramp/sigmas/base_1'
params['obs_ranges'] = [[0.1,0.9]]


experiment = toy.Toy_Experiment(params)
experiment.full_run()


