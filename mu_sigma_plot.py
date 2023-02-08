import os
from Source.Experiments import toy
from Source.Util.util import load_params
import matplotlib.pyplot as plt
import numpy as np
import torch
import sys

path = sys.argv[1]


par = load_params(os.path.join(path, "paramfile.yaml"))

par["warm_start"] = True
par["warm_path"] = path

for j in range(50,10001,50):
    par["model_name"] = f"model_epoch_{j}"
    ex = toy.Toy_Experiment(par)
    ex.model = ex.build_model(par, prior_path = par['warm_path'])

    fig, axes = plt.subplots()

    for i in range(len(ex.model.net.blocks[0])):
        try:
            mu = ex.model.net.blocks[0][i].mu_w.data.cpu()
            sig= ex.model.net.blocks[0][i].logsig2_w.data.cpu()
            axes.scatter(mu,np.e**sig, label=f"{i+1}. layer", s=1)
        except:
            pass


    axes.set_xlabel("$\mu_\Theta$",fontsize=14)
    axes.set_ylabel("$\sigma_\Theta^2$",fontsize=14)
    axes.set_xlim(-2.5,2.5)
    axes.set_ylim(-0.001,0.0102)
    axes.legend(fontsize=12, frameon=False,loc="lower left")
    axes.tick_params(axis='both', which='major', labelsize=14)
    axes.set_title(f"Network with {par['model_parameters']} parameters", fontsize=14)

    fig.tight_layout(pad=3.0)
    fig.savefig(f"/remote/gpu05/palacios/Plots/mu_sigma_simple/mu_sigma_epoch_{j}.png", dpi=300)




