import numpy as np
import torch
import os, sys, time
sys.path.append(os.getcwd()) #can we do better?
from Source.Models.inn import INN
from Source.Models.tbd import TBD
from Source.Models.ddpm import DDPM
from Source.Models.autoregGMM import AutoRegGMM
from Source.Util.util import load_params, get_device
from Source.Util.datasets import Dataset
from Source.Util.preprocessing import preformat, preprocess, undo_preprocessing
from Source.Experiments.zn import Zn_Experiment

device = get_device()

nBNN = 10
n_samples = 1000000

def genHistograms(path, nModels=10):
    if device=="cuda":
        sys.stdout = open(f"paper/jet/stdout_{path}.txt", "w", buffering=1)
        sys.stderr = open(f"paper/jet/stderr_{path}.txt", "w", buffering=1)
        
    print(f"## Generating histograms for {path} ##")
    t0 = time.time()

    # load param dict
    params = load_params("runs/"+path + "/paramfile.yaml")
    params["iterations"] = nBNN
    params["n_samples"] = n_samples
    params["device"] = device
    params["train"] = False

    experiment = Zn_Experiment(params)
    if device=="cpu":
        params["data_path"] = "/media/jspinner/shared/Studium/project1/data/z+njet_full.h5"
    experiment.load_data()
    experiment.preprocess_data_njets()
    experiment.model = eval(params["model"])(params)
    state_dict = torch.load("runs/"+path+"/models/model_run0.pt", map_location=device)
    experiment.model.load_state_dict(state_dict)
    experiment.build_dataloaders()

    experiment.generate_samples()

    np.save(f"paper/jet/{path}_train_1.npy", experiment.data_train_1)
    np.save(f"paper/jet/{path}_train_2.npy", experiment.data_train_2)
    np.save(f"paper/jet/{path}_train_3.npy", experiment.data_train_3)
    np.save(f"paper/jet/{path}_test_1.npy", experiment.data_test_1)
    np.save(f"paper/jet/{path}_test_2.npy", experiment.data_test_2)
    np.save(f"paper/jet/{path}_test_3.npy", experiment.data_test_3)
    np.save(f"paper/jet/{path}_sample_1.npy", experiment.samples_1)
    np.save(f"paper/jet/{path}_sample_2.npy", experiment.samples_2)
    np.save(f"paper/jet/{path}_sample_3.npy", experiment.samples_3)
    
    t1 = time.time()
    print(f"Total time consumption: {t1-t0:.2f}s = {(t1-t0)/60:.2f}min = {(t1-t0)/60**2:.2f}h")

    if device=="cuda":
        sys.stdout.close()
        sys.stderr.close()

genHistograms("AutoRegGMMcond_4342")
