import numpy as np
import torch
import os, sys, time
sys.path.append(os.getcwd()) #can we do better?
from Source.Models.inn import INN
from Source.Models.tbd import TBD
from Source.Models.ddpm import DDPM
from Source.Models.autoregGMM import AutoRegGMM
from Source.Models.autoregBinned import AutoRegBinned
from Source.Util.util import load_params, get_device
from Source.Experiments import toy
from Source.Util.simulateToyData import ToySimulator

import matplotlib.pyplot as plt

paths_ramp = [f"runs/paper_rampv2_{i}/" for i in range(1, 11)]
paths_sphere = [f"runs/paper_spherev2_{i}/" for i in range(1, 11)]
device = get_device()

def genHistograms(paths):
    print(f"## Generating histograms for {paths} ##")
    t0 = time.time()
    
    nbins = 60
    nBNN = 30
    n_samples = 1000000
    nEnsemble = len(paths)
    #range_ramp = [-0.1, 1.1]
    range_ramp = [0.1, 0.9]
    range_gauss_sphere = [0.5, 1.5]

    dup_last = lambda a: np.append(a, a[-1])

    # generate data, then generate histograms (for each BNN in the ensemble)
    histograms = np.zeros((5+nEnsemble, nbins+1, 2)) #first 4 for bins, train, test, BNN ensemble and uncertainty stuff, then individual BNNs
    for ipath in range(nEnsemble):
        print(f"Starting to generate histograms for BNN {ipath}")
        
        # load param dict
        params = load_params(paths[ipath] + "paramfile.yaml")
        params["device"] = device
        if params["toy_type"] == "ramp":
            params["data_path"] = "../data/2dRamp.npy"
        elif params["toy_type"] == "gauss_sphere":
            params["data_path"] = "../data/2dGaussSphere.npy"

        # load model
        model = eval(params["model"])(params, out=False)
        state_dict = torch.load(paths[ipath]+"models/model_run0.pt", map_location=params["device"])
        model.load_state_dict(state_dict)

        # load data
        # (could do this outside of the loop, we keep it in the loop to improve readability)
        data = np.load(params["data_path"])
        data_split = params["data_split"]
        n_data = len(data)
        cut1 = int(n_data - data_split[0])
        cut2 = int(n_data * (data_split[0] + data_split[1]))
        data_train = data[:cut1]
        data_test = data[cut2:]

        # generate events
        data_predict = np.zeros((0, data_train.shape[1]))
        for _ in range(nBNN):
            data_predict= np.append(data_predict, model.sample_n(n_samples), axis=0)
    
        # compute component of interest
        def get_obs(data):
            if params["toy_type"] == "ramp":
                return data[:,1]
            elif params["toy_type"] == "gauss_sphere":
                return ToySimulator.getSpherical(data)[0]
        obs_train = get_obs(data_train)
        obs_test = get_obs(data_test)
        obs_predict = get_obs(data_predict)

        range_toy = range_ramp if params["toy_type"]=="ramp" else range_gauss_sphere
        # generate histograms for training and test data
        if ipath==0: 
            y_t,  bins = np.histogram(obs_test, bins=nbins, range=range_toy)
            y_tr, _ = np.histogram(obs_train, bins=nbins, range=range_toy)
    
        # generate histograms for generated data
        weight_samples = nBNN #to avoid confusion
        obs_predict = obs_predict.reshape(weight_samples,
            len(obs_predict)//weight_samples)
        hists_g = np.array([np.histogram(obs_predict[i,:], bins=nbins, range=range_toy)[0]
                    for i in np.arange(weight_samples)])
        hists = [y_t, np.mean(hists_g, axis=0), y_tr]
        hist_errors = [np.sqrt(y_t), np.std(hists_g, axis=0), np.sqrt(y_tr)]
        integrals = [np.sum((bins[1:] - bins[:-1]) * y) for y in hists]
        scales = [1 / integral if integral != 0. else 1. for integral in integrals]

        if ipath==0:
            histograms[0, :, 0] = bins
            histograms[1, :, 0] = dup_last(y_tr * scales[2])
            histograms[1, :, 1] = dup_last(np.sqrt(y_tr) * scales[2])
            histograms[2, :, 0] = dup_last(y_t * scales[0])
            histograms[2, :, 1] = dup_last(np.sqrt(y_t) * scales[0])
        histograms[5+ipath, :, 0] = dup_last(hists[1] * scales[1])
        histograms[5+ipath, :, 1] = dup_last(hist_errors[1] * scales[1])


    # compute
    histograms[3, :, 0] = dup_last(np.mean(histograms[5:, :-1, 0], axis=0)) #histogram means (for normalization)
    histograms[3, :, 1] = dup_last(1/nEnsemble * np.sum(histograms[5:, :-1, 1]**2, axis=0)**.5) #gaussian error propagation -> effective uncertainty (reduced!)
    histograms[4, :, 0] = dup_last(np.mean(histograms[5:, :-1, 1], axis=0)) #mean of uncertainties (just to have it)
    histograms[4, :, 1] = dup_last(np.std(histograms[5:, :-1, 1], axis=0)) #uncertainty on uncertainty

    if params["model"]=="AutoRegGMM":
        model_type = "GMM"
    elif params["model"]=="AutoRegBinned":
        model_type = "Binned"
    np.save(f"paper/toy/{model_type}_{params['toy_type']}.npy", histograms)

    t1 = time.time()
    print(f"Total time consumption: {t1-t0:.2f}s = {(t1-t0)/60:.2f}min = {(t1-t0)/60**2:.2f}h")

if device=="cuda":
    sys.stdout = open("paper/toy/stdout.txt", "w", buffering=1)
    sys.stderr = open("paper/toy/stderr.txt", "w", buffering=1)

genHistograms(paths_ramp)
genHistograms(paths_sphere)

if device=="cuda":
    sys.stdout.close()
    sys.stderr.close()
