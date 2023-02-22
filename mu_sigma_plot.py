import os
from Source.Experiments import toy
from Source.Util.util import load_params, get_device
import matplotlib.pyplot as plt
import numpy as np
import torch
import sys
import matplotlib as mpl

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

col = mpl.cm.Set1(np.linspace(0,1,9))

for j in range(par["n_epochs"]):
    par["model_name"] = f"model_epoch_{j}"

    fig, axes = plt.subplots()
    try:
        ex = toy.Toy_Experiment(par)
        ex.model = ex.build_model(par, prior_path = par['warm_path'])
        for i in range(par["n_blocks"]):
            if bayesian >= 1:
                mu1 = ex.model.net.transformer.h[i].mlp.c_fc.mu_w.data.cpu().flatten()
                mu2 = ex.model.net.transformer.h[i].mlp.c_proj.mu_w.data.cpu().flatten()
                sig1 = ex.model.net.transformer.h[i].mlp.c_fc.logsig2_w.data.cpu().flatten()
                sig2 = ex.model.net.transformer.h[i].mlp.c_proj.logsig2_w.data.cpu().flatten()
                muMLP = torch.cat([mu1, mu2])
                sigMLP = torch.cat([sig1, sig2])
                
                axes.scatter(muMLP,(np.e**sigMLP)**.5, s=1, color=col[0], label="mlp")
            if bayesian >= 2:
                mu1 = ex.model.net.transformer.h[i].attn.c_attn.mu_w.data.cpu().flatten()
                mu2 = ex.model.net.transformer.h[i].attn.c_proj.mu_w.data.cpu().flatten()
                sig1 = ex.model.net.transformer.h[i].attn.c_attn.logsig2_w.data.cpu().flatten()
                sig2 = ex.model.net.transformer.h[i].attn.c_proj.logsig2_w.data.cpu().flatten()
                muATT = torch.cat([mu1, mu2])
                sigATT = torch.cat([sig1, sig2])
                
                axes.scatter(muATT,(np.e**sigATT)**.5, s=1, color=col[1], label="att")
            if bayesian >= 3:
                muWTE = ex.model.net.transformer.wte.mu_w.data.cpu().flatten()
                muEND = ex.model.net.lm_head.mu_w.data.cpu().flatten()
                sigWTE = ex.model.net.transformer.wte.logsig2_w.data.cpu().flatten()
                sigEND = ex.model.net.lm_head.logsig2_w.data.cpu().flatten()
            
                axes.scatter(muEND,(np.e**sigEND)**.5, s=1, color=col[3], label="gauss")
                axes.scatter(muWTE,(np.e**sigWTE)**.5, s=1, color=col[2], label="emb")

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







