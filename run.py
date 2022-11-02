import numpy as np
import torch
from Source.Models.inn import INN
from Source.Models.tbd import TBD
from Source.Models.ddpm import DDPM
import matplotlib.pyplot as plt
from Source.Util.plots import plot_obs, delta_r
from Source.Util.preprocessing import preprocessing_4c, undo_preprocessing_4c
from Source.Util.preprocessing import preprocessing_6c, undo_preprocessing_6c
from Source.Util.util import get_device, save_params
import os
import time
from datetime import datetime


def run(param):

    # data
    data_dir = param.get("data_dir")
    data_path = os.path.join(data_dir, "Z_2.npy")
    channels = param.get("channels")
    device = get_device()
    param["device"] = device
    data_np = np.load(data_path)

    param["datetime"] = str(datetime.now())

    dim = param.get("dim")
    if dim == 4:
        events, events_mean, events_std, u, s = preprocessing_4c(data_np)
    elif dim == 6:
        events, events_mean, events_std, u, s = preprocessing_6c(data_np)
    else:
        raise ValueError("dim not implemented")
    data = torch.Tensor(events).to(device)

    data_train = data[:data.shape[0] // 2]
    data_test = data[data.shape[0] // 2:]

    model = param["model"]
    warm_start = param["warm_start"]

    # model
    if model == "INN":
        model = INN(param)
        model.define_model_architecture()
        param["model_parameters"] = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        if warm_start:
            warm_start_path = param["warm_start_path"] + "/models/model.pt"
            state_dicts = torch.load(warm_start_path, map_location=model.device)
            model.model.load_state_dict(state_dicts)
    elif model == "SBD":
        model = TBD(param)
        param["model_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if warm_start:
            warm_start_path = param["warm_start_path"] + "/models/model.pt"
            state_dicts = torch.load(warm_start_path, map_location=model.device)
            model.load_state_dict(state_dicts)
    elif model == "DDPM":
        model = DDPM(param)
        param["model_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if warm_start:
            warm_start_path = param["warm_start_path"] + "/models/model.pt"
            state_dicts = torch.load(warm_start_path, map_location=model.device)
            model.load_state_dict(state_dicts)
    else:
        raise ValueError("Unknown model")
    train = param.get("train", True)

    if train:
        os.makedirs("models", exist_ok = True)
        save_every = param.get("save_every", 100)
        save_checkpoints = param.get("save_checkpoints", False)

        t0 = time.time()
        model.run_training(data=data_train,
                           n_epochs=param.get("n_epochs"),
                           batch_size=param.get("batch_size"),
                           lr=param.get("lr"),
                           betas=param.get("betas"),
                           save_every=save_every,
                           save_checkpoints=save_checkpoints)
        t1 = time.time()
        param["traintime"] = t1-t0
        torch.save(model.state_dict(), "models/model.pt")

    model.eval()
    # sample and plot
    n_samples = param.get("n_samples")
    t0 = time.time()
    samples = model.sample_n_parallel(n_samples)
    t1 = time.time()
    param["sampletime"] = t1-t0

    if dim == 4:
        events_undone = undo_preprocessing_4c(events, events_mean, events_std, u, s)
        samples_undone = undo_preprocessing_4c(samples, events_mean, events_std, u, s)
    elif dim == 6:
        events_undone = undo_preprocessing_6c(events, events_mean, events_std, u, s)
        samples_undone = undo_preprocessing_6c(samples, events_mean, events_std, u, s)
    np.save('events_undone.npy', events_undone)
    np.save('samples_undone.npy', samples_undone)

    plot = param.get("plot", True)
    if plot:
        os.makedirs("plots", exist_ok = True)
        os.chdir("plots")
        plt.plot(model.loss_hist)
        plt.savefig("loss.pdf", dpi=150)

        if dim==4:
            idx_phi1 = 0
            idx_eta1 = 1
            idx_phi2 = 2
            idx_eta2 = 3
        elif dim==6:
            idx_phi1 = 1
            idx_eta1 = 2
            idx_phi2 = 4
            idx_eta2 = 5

        # curve plot
        file_name = 'Z_2_delta_R_j1_j2.pdf'
        obs = lambda events: delta_r(events, idx_phi1, idx_eta1, idx_phi2, idx_eta2)
        obs_name = '\Delta R_{j_1 j_2}'
        plot_obs(file_name, obs(events_undone[events_undone.shape[0] // 2:]),
                 obs(events_undone[:events_undone.shape[0] // 2]),
                 obs(samples_undone), obs_name, range=[0, 8])

        if dim == 6:
            file_name = 'Z_2_pT_j1.pdf'
            obs_name = 'p_{T,j_1}[GeV]'
            plot_obs(file_name, events_undone[events_undone.shape[0] // 2:, 0],
                     events_undone[:events_undone.shape[0] // 2, 0],
                     samples_undone[:, 0], obs_name, range=[0, 150])

            file_name = 'Z_2_pT_j2.pdf'
            obs_name = 'p_{T,j_2}[GeV]'
            plot_obs(file_name, events_undone[events_undone.shape[0] // 2:, 3],
                     events_undone[:events_undone.shape[0] // 2, 3],
                     samples_undone[:, 3], obs_name, range=[0, 150])


        # 2d heatmap plot


        file_name = 'deta_dphi.png'
        fig = plt.figure(figsize=(12, 6))
        fig.add_subplot(1, 2, 1)
        deta = samples_undone[:, idx_eta1] - samples_undone[:, idx_eta2]
        dphi = samples_undone[:, idx_phi1] - samples_undone[:, idx_phi2]
        dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
        plt.hist2d(deta, dphi, bins=100, range=[[-5, 5], [-np.pi, np.pi]])
        plt.xlabel('$\Delta \eta$')
        plt.ylabel('$\Delta \phi$')
        plt.title('model')
        fig.add_subplot(1, 2, 2)
        deta = events_undone[:, idx_eta1] - events_undone[:, idx_eta2]
        dphi = events_undone[:, idx_phi1] - events_undone[:, idx_phi2]
        dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
        plt.hist2d(deta, dphi, bins=100, range=[[-5, 5], [-np.pi, np.pi]])
        plt.xlabel('$\Delta \eta$')
        plt.ylabel('$\Delta \phi$')
        plt.title('ground truth')
        plt.savefig(file_name)

        os.chdir("..")

    save_params(param)
