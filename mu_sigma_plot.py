import os
from Source.Experiments import toy
from Source.Util.util import load_params
import matplotlib.pyplot as plt
import numpy as np
import torch
import sys

path = sys.argv[1]
model_name = sys.argv[2]
out = sys.argv[3]


par = load_params(os.path.join(path, "paramfile.yaml"))

par["warm_start"] = True
par["warm_path"] = path
par["model_name"] = model_name

ex = toy.Toy_Experiment(par)
ex.model = ex.build_model(par, prior_path = par['warm_path'])

fig, axes = plt.subplots()

for i in range(len(ex.model.net.blocks[0])):
    try:
        mu = ex.model.net.blocks[0][i].mu_w.data
        sig= ex.model.net.blocks[0][i].logsig2_w.data
        axes.scatter(mu,np.e**sig, label=f"{i}. layer", s=1)
    except:
        pass


axes.set_xlabel("$\mu_\Theta$",fontsize=14)
axes.set_ylabel("$\sigma_\Theta^2$",fontsize=14)
axes.legend(fontsize=14, frameon=False)
axes.tick_params(axis='both', which='major', labelsize=14)
axes.set_title(f"Network with {par['model_parameters']} parameters", fontsize=14)

fig.tight_layout(pad=3.0)
fig.savefig(f"../Test/Plots/{out}.pdf")




