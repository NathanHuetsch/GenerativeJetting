from Source.Util.physics import EpppToPTPhiEta
import numpy as np


def preprocess_log_jet_mass(events_four_vector):
    # convert to (pT, phi, eta, mu)
    events = EpppToPTPhiEta(events_four_vector, reduce_data=False, include_masses=True)

    # apply log transform to pT
    events[:, 0] = np.log(events[:, 0])
    events[:, 4] = np.log(events[:, 4])
    events[:, 8] = np.log(events[:, 8] - 20 + 1e-2)
    events[:, 12] = np.log(events[:, 12] - 20 + 1e-2)

    phi_idx_skip_first_muon = np.arange(3) * 4 + 5
    for i in phi_idx_skip_first_muon:
        # make phi relative
        events[:, i] = events[:, i] - events[:, 1]
        # limit to [-pi, pi]
        events[:, i] = (events[:, i] + np.pi) % (2 * np.pi) - np.pi
        # apply atanh transform
        events[:, i] = np.arctanh(events[:, i] / np.pi)

    # apply log transform to jet mass
    events[:, 11] = np.log(events[:, 11])
    events[:, 15] = np.log(events[:, 15])

    # discard muon masses and first muon delta_phi
    discard_dims = np.array([1, 3, 7])
    keep_dims = np.array([i for i in range(events.shape[1]) if i not in discard_dims])
    events = events[:, keep_dims]

    # apply standardization and whitening
    events_mean = events.mean(0, keepdims=True)
    events_std = events.std(0, keepdims=True)
    events = (events - events_mean) / events_std
    u, s, vt = np.linalg.svd(events.T @ events / events.shape[0])
    events = events @ u
    events = events / np.sqrt(s)[None]

    # return preprocessed events and information needed to undo transformations
    return events, events_mean, events_std, u, s


def undo_preprocessing_log_jet_mass(events, events_mean, events_std, u, s):
    # undo whitening
    events = events * np.sqrt(s)[None]
    events = events @ u.T

    # undo standardization
    events = events * events_std + events_mean

    # undo log transform (jet mass)
    events[:, 8] = np.exp(events[:, 8])
    events[:, 12] = np.exp(events[:, 12])

    # undo atanh transform
    phi_idx = [3, 6, 10]
    for i in phi_idx:
        events[:, i] = np.tanh(events[:, i]) * np.pi

    # undo log transform
    events[:, 0] = np.exp(events[:, 0])
    events[:, 2] = np.exp(events[:, 2])
    events[:, 5] = np.exp(events[:, 5]) + 20 - 1e-2
    events[:, 9] = np.exp(events[:, 9]) + 20 - 1e-2

    return events


def preprocess(events_four_vector, channels=None):

    # convert to (pT, phi, eta, mu)
    events = EpppToPTPhiEta(events_four_vector, reduce_data=False, include_masses=True)

    # apply log transform to pT
    events[:, 0] = np.log(events[:, 0])
    events[:, 4] = np.log(events[:, 4])
    events[:, 8] = np.log(events[:, 8] - 20 + 1e-2)
    events[:, 12] = np.log(events[:, 12] - 20 + 1e-2)

    phi_idx = [1, 5, 9, 13]
    for i in phi_idx:
        # make phi relative
        if i != 1:
            events[:, i] = events[:, i] - events[:, 1]
        # limit to [-pi, pi]
        events[:, i] = (events[:, i] + np.pi) % (2 * np.pi) - np.pi
        # apply atanh transform
        events[:, i] = np.arctanh(events[:, i] / np.pi)

    # discard muon masses and first muon delta_phi
    if channels is not None:
        events = events[:, channels]

    # apply standardization and whitening
    events_mean = events.mean(0, keepdims=True)
    events_std = events.std(0, keepdims=True)
    events = (events - events_mean) / events_std
    u, s, vt = np.linalg.svd(events.T @ events / events.shape[0])
    events = events @ u
    events = events / np.sqrt(s)[None]

    # return preprocessed events and information needed to undo transformations
    return events, events_mean, events_std, u, s


def undo_preprocessing(events, events_mean, events_std, u, s,
                       channels=None,
                       keep_all=False):
    # undo whitening
    events = events * np.sqrt(s)[None]
    events = events @ u.T

    # undo standardization
    events = events * events_std + events_mean

    if channels is not None:
        temp = events.copy()
        events = np.zeros((events.shape[0], 16))
        events[:, channels] = temp
    # undo atanh transform
    phi_idx = [1, 5, 9, 13]
    for i in phi_idx:
        events[:, i] = np.tanh(events[:, i]) * np.pi

    # undo log transform
    events[:, 0] = np.exp(events[:, 0])
    events[:, 4] = np.exp(events[:, 4])
    events[:, 8] = np.exp(events[:, 8]) + 20 - 1e-2
    events[:, 12] = np.exp(events[:, 12]) + 20 - 1e-2

    if channels is None or keep_all:
        return events
    else:
        return events[:, channels]
