import os
from Source.Experiments import toy
from Source.Util.util import load_params, get_device
import matplotlib.pyplot as plt
import numpy as np
import torch
import sys
import matplotlib as mpl
from Source.Networks.vblinear import VBLinear

path = sys.argv[1]


par = load_params(os.path.join(path, "paramfile.yaml"))

prior_prec = par["prior_prec"]
prior_prec = 1 / prior_prec**.5
bayesian = par["bayesian"]

print(prior_prec)

par["warm_start"] = True
par["warm_path"] = path
par["device"] = get_device()
os.makedirs(os.path.join(path, "mu_sigma"), exist_ok=True)

col = mpl.cm.Set1(np.linspace(0,1,20))

for j in range(par["n_epochs"]):
    par["model_name"] = f"model_epoch_{j}"

    fig, axes = plt.subplots()
    try:
        ex = toy.Toy_Experiment(par)
        ex.model = ex.build_model(par, prior_path = par['warm_path'])
        for i in range(par["n_blocks"]):
            block = ex.model.net.blocks[i]
            print(len(block))
            for layer in range(len(block)):
                if isinstance(block[layer], VBLinear):

                    mu = block[layer].mu_w.data.cpu().flatten()
                    sig = block[layer].logsig2_w.data.cpu().flatten()

                    axes.scatter(mu,(np.e**sig)**.5, s=1, color=col[layer], label=f"layer{layer}")


        xrange = 2.
        axes.set_xlabel("$\mu_\Theta$",fontsize=14)
        axes.set_ylabel("$\sigma_\Theta$",fontsize=14)
        axes.set_xlim(-xrange, xrange)
        axes.set_ylim(- .03 * prior_prec, prior_prec * 1.1)
        axes.plot([-xrange, xrange], [0., 0.], "k--", alpha=.3)
        axes.plot([-xrange, xrange], [prior_prec, prior_prec], "k--", alpha=.3)
        axes.legend(fontsize=12,loc="lower left")
        axes.tick_params(axis='both', which='major', labelsize=14)
        axes.set_title(f"Network with {par['model_parameters']} parameters", fontsize=14)

        fig.tight_layout(pad=3.0)
        fig.savefig(os.path.join(path, f"mu_sigma/mu_sigma_epoch_{j}.png"), dpi=300, bbox_inches="tight")
        plt.close()
    except ValueError: #file does not exist
        plt.close()
