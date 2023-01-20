from Source.Util.physics import EpppToPTPhiEta
import numpy as np
import  torch

"""
Methods to perform the physics preprocessing of the 16dim Z+2jets input data
Code based on Theos implementation but somewhat changed 
"""
def preformat(data):
    '''
    Bring data in Z_2.npy format (E, px, py, pz) into the (pT, phi, eta, mu) format that is used by the generator
    and make phi angles relative to the first phi angle (= fix coordinate system)
    :data: Events in the (E, px, py, pz) format
    :returns: Events in the (pT, phi, eta, mu) format of shape (n_events, 4*(2+n_jets))
    '''
    if conditional:
        events = EpppToPTPhiEta(data[:, :-1], reduce_data=False, include_masses=True)
    else:
        events = EpppToPTPhiEta(data, reduce_data=False, include_masses=True)
    
    events[:, 5::4] = events[:, 5::4] - events[:,1,None]
    events[:, 1::4] = (events[:, 1::4] + np.pi) % (2*np.pi)- np.pi
    return events

def preprocess(data, params):
    """
    Bring data into the format used during training
    :param data: the data as a numpy array. Assumed to be of shape [* , 8+4*n_jets+1]
    :param params: param dict for options
    :return: the preprocessed data, the hot-encoded condition and some statistics necessary to reproduce the original data
    """
    preprocess = get(params, "preprocess", 3)
    channels = params["channels"]
    conditional = get(params, "conditional", False)
    n_jets = get(params, "n_jets", 2)

    events = data.copy()

    if preprocess>=1:
        # apply log transform to pT
        events[:, 0] = np.log(events[:, 0])
        events[:, 4] = np.log(events[:, 4])
        events[:, 8::4] = np.log(events[:, 8::4] - 20 + 1e-2)

        events[:, 1::4] = np.arctanh(events[:, 1::4]/np.pi)

    # discard unwanted channels
    if channels is not None:
        events = events[:, channels]

    # apply standardization
    if preprocess>=2:
        events_mean = events.mean(0, keepdims=True)
        events_std = events.std(0, keepdims=True)
        events = (events - events_mean) / events_std

    # apply whitening
    if preprocess>=3:
        u, s, vt = np.linalg.svd(events.T @ events / events.shape[0])
        events = events @ u
        events = events / np.sqrt(s)[None]

    if conditional and n_jets != 3:
        condition = encode_condition(data[:, -1], n=n_jets)
        events = np.append(events, condition, axis=1)

    # return preprocessed events and information needed to undo transformations
    return events, events_mean, events_std, u, s


def undo_preprocessing(data, events_mean, events_std, u, s, params):
    """
    The exact inverse of preprocess
    :param data: the preprocessed data as a numpy array of shape [* , len(channels)]
    :param events_mean: the mean of the original data (as returned by the preprocess() method)
    :param events_std: the std of the original data (as returned by the preprocess() method)
    :param u: the u of the original data (as returned by the preprocess() method)
    :param s: the s of the original data (as returned by the preprocess() method)
    :param params: param dict for options
    :return: the data in the original format with the preprocessing undone
    """
    preprocess = get(params, "preprocess", 3)
    channels = params["channels"]
    conditional = get(params, "conditional", False)
    n_jets = get(params, "n_jets", 2)
    
    if conditional and n_jets != 3:
        cut = 4 - n_jets
        events = data[:, :-cut]
    else:
        events = data

    # undo whitening
    if preprocess>=3:
        events = events * np.sqrt(s)[None]
        events = events @ u.T

    # undo standardization
    if preprocess>=2:
        events = events * events_std + events_mean

    if channels is not None:
        temp = events.copy()
        events = np.zeros((events.shape[0], 20))
        events[:, channels] = temp

    if preprocess>=1:
        # undo atanh transform
        events[:,1::4] = np.tanh(events[:, 1::4]) * np.pi

        # undo log transform
        events[:, 0] = np.exp(events[:, 0])
        events[:, 4] = np.exp(events[:, 4])
        events[:, 8::4] = np.exp(events[:, 8::4]) + 20 - 1e-2

    if conditional and n_jets != 3:
        cut = 4 - n_jets
        condition = decode_condition(data[:, -cut:], n=n_jets)
        events = np.append(events, condition, axis=1)

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
