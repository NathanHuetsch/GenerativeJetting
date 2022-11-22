from Source.Util.physics import EpppToPTPhiEta
import numpy as np

"""
Methods to perform the physics preprocessing of the 16dim Z+2jets input data
Code based on Theos implementation but somewhat changed 
"""


def preprocess(events_four_vector, channels=None, fraction=None):
    """
    :param events_four_vector: the data as a numpy array. Assumed to be of shape [* , 16]
    :param channels: a list of channels we want to keep
    :param fraction: fraction of data kept
    :return: the preprocessed data and some statistics necessary to reproduce the original data
    """

    # keep only fraction of dataset to run tests
    if fraction is not None:
        n = round(fraction * events_four_vector.shape[0])
        events_four_vector = events_four_vector[:n]

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

    # discard unwanted channels
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
    """
    :param events: the preprocessed data as a numpy array of shape [* , len(channels)]
    :param events_mean: the mean of the original data (as returned by the preprocess() method)
    :param events_std: the std of the original data (as returned by the preprocess() method)
    :param u: the u of the original data (as returned by the preprocess() method)
    :param s: the s of the original data (as returned by the preprocess() method)
    :param channels: the channels of the data
    :param keep_all: if False only return the specified channels. if True return an array of shape
                     [* , 16] where the remaining channels are filled with zeros
    :return: the data in the original format with the preprocessing undone
    """
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
