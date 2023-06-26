import torch
import numpy as np
import sys
import matplotlib.pyplot as plt
import Source
from Source.Experiments import z1, z2, z3, toy
from Source.Util.util import load_params, get

# Load params
params_path = "/remote/gpu07/huetsch/GenerativeJetting/params/Ramp2D_TBD.yaml"
params = load_params(params_path)
params["redirect_console"] = True
params["log"] = False

for i in range(5):
    params["run_name"] = f"run_{i}"
    experiment = toy.Toy_Experiment(params)
    experiment.full_run()