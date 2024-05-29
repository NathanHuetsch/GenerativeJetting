from Source.Util.physics import EpppToPTPhiEta
import numpy as np
import torch
from Source.Util.util import get


def preformat(data):
    '''
    Bring data in Z_2.npy format (E, px, py, pz) into the (pT, phi, eta, mu) format that is used by the generator
    and make phi angles relative to the first phi angle (= fix coordinate system)
    :data: Events in the (E, px, py, pz) format
    :returns: Events in the (pT, phi, eta, mu) format of shape (n_events, 4*(2+n_jets))
    '''

    events = EpppToPTPhiEta(data, reduce_data=False, include_masses=True)

    events[:, 5::4] = events[:, 5::4] - events[:, 1, None]
    events[:, 1] = 0
    events[:, 1::4] = (events[:, 1::4] + np.pi) % (2*np.pi)- np.pi

    return events

def preprocess(data, params):
    """
    Bring data into the format used during training
    :param data: the data as a numpy array.
    :param params: param dict for options
    :return: the preprocessed data, the mean and std of the data
    """

    preprocess = get(params, "jet_preprocessing", True)
    channels = params["channels"]

    events = data.copy()

    if preprocess:
        # apply log transform to pT
        events[:, 0] = np.log(events[:, 0])
        events[:, 4] = np.log(events[:, 4])
        events[:, 8::4] = np.log(events[:, 8::4] - 20 + 1e-2)

        # apply log transform to mu
        events[:,3::4] = np.log(events[:,3::4])

        # apply artanh transform to phi
        events[:, 1::4] = np.arctanh(events[:, 1::4]/np.pi)

    # discard unwanted channels
    events = events[:, channels]

    # apply standardization
    events_mean = events.mean(0, keepdims=True)
    events_std = events.std(0, keepdims=True)
    events = (events - events_mean) / events_std

    # return preprocessed events and information needed to undo transformations
    return events, events_mean, events_std


def undo_preprocessing(data, events_mean, events_std, params):
    """
    The exact inverse of preprocess
    :param data: the preprocessed data as a numpy array of shape [* , len(channels)]
    :param events_mean: the mean of the original data (as returned by the preprocess() method)
    :param events_std: the std of the original data (as returned by the preprocess() method)
    :param params: param dict for options
    :return: the data in the original format with the preprocessing undone
    """

    preprocess = get(params, "jet_preprocessing", True)
    channels = params["channels"]
    events = data.copy()

    # undo standardization
    events = events * events_std + events_mean

    if channels is not None:
        temp = events.copy()
        events = np.zeros((events.shape[0], 20))
        events[:, channels] = temp

    if preprocess:
        # undo atanh transform
        events[:,1::4] = np.tanh(events[:, 1::4]) * np.pi

        # undo log transform
        events[:, 0] = np.exp(events[:, 0])
        events[:, 4] = np.exp(events[:, 4])
        events[:, 8::4] = np.exp(events[:, 8::4]) + 20 - 1e-2
        events[:, 3::4] = np.exp(events[:, 3::4])

    return events


def encode_condition(x, n=1):
    m = 4 - n
    con = []
    for i in x:
        a = np.zeros(m)
        a[int(i) - n] = 1
        con.append(a)
    return torch.tensor(np.array(con))


def decode_condition(x, n=1):
    con = []
    for i in x:
        a = np.nonzero(i)[0]
        con.append(a+n)
    return np.array(con)
