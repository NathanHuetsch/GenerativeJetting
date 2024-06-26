import numpy as np

"""
Methods to do some physics calculations with the jet observables
Code copy-pasted from Theo

Currently only the EpppToPTPhiEta is used in this project
"""

def EpppToPTPhiEta(events, reduce_data, dim = 4, include_masses=False):
    pt = get_pt(events, dim)
    phi = get_phi(events, dim)
    eta = get_eta(events, dim)
    mass = get_mass(events, dim)

    dim_out = dim - int(not include_masses)
    n_objects = events.shape[1] // dim
    events_out = np.empty((events.shape[0], n_objects*dim_out))
    events_out[:,0::dim_out] = pt
    events_out[:,1::dim_out] = phi
    events_out[:,2::dim_out] = eta
    if include_masses:
        if reduce_data:
            print("Warning, include_masses and reduce_data are both set, ignoring reduce_data")
        events_out[:, 3::dim_out] = mass
    elif reduce_data:
        events_out = reduce(events_out)

    return events_out

def PTPhiEtaToEppp(events, masses, dim = 3):
    """reconstruct E, px, py, pz from pt, eta, phi and the particle masses"""
    if events.shape[1] == 7:
        events = reconstruct(events)
    px, py = np.cos(events[:,1::dim]) * events[:,0::dim], np.sin(events[:,1::dim]) * events[:,0::dim]


    #Calculate pz via pz = sqrt((tanh(eta)**2 * pt**2) /(1 - tanh(eta)**2))
    tanh_eta = (np.exp(2*events[:,2::dim]) - 1)/(np.exp(2*events[:,2::dim]) + 1)
    pz = np.sqrt((tanh_eta**2 * events[:,0::dim]**2) /(1 - tanh_eta**2))

    #E ** 2 = M ** 2 + P ** 2
    E = np.sqrt(masses ** 2 + px ** 2 + py ** 2 + pz ** 2)
    events_out = np.empty((events.shape[0], events.shape[1])) #Note: Theo hat 12 instead of events.shape[1]
    events_out[:,0::4] = E
    events_out[:,1::4] = px
    events_out[:,2::4] = py
    events_out[:,3::4] = pz
    return events_out

def PTPhiPZToPTPhiEta(events, dim = 3, eps=1e-15):
    reduce_data = False
    if events.shape[1] == 7:
        events = reconstruct(events)
        reduce_data = True
    Ps = np.sqrt(np.square(events[:,0::dim]) + np.square(events[:,2::dim]))
    eta = 0.5 * (np.log(np.clip(np.abs(Ps + events[:,2::dim]), eps, None)) -
                 np.log(np.clip(np.abs(Ps - events[:,2::dim]), eps, None)))
    events_out = np.empty(events.shape)
    events_out[:, np.mod(np.arange(events_out.shape[1])+2,dim)!=0] = events[:, np.mod(np.arange(events.shape[1])+2,dim)!=0]
    events_out[:, 2::dim] = eta
    return reduce(events_out) if reduce_data else events_out

def PTPhiEtaToPTPhiPZ(events, dim = 3):
    reduce_data = False
    if events.shape[1] == 7:
        events = reconstruct(events)
        reduce_data = True

    tanh_eta = (np.exp(2*events[:,2::dim]) - 1)/(np.exp(2*events[:,2::dim]) + 1)
    pz = np.sqrt((tanh_eta**2 * events[:,0::dim]**2) /(1 - tanh_eta**2))
    events_out = np.empty(events.shape)
    events_out[:, np.mod(np.arange(events_out.shape[1])+2,dim)!=0] = events[:, np.mod(np.arange(events.shape[1])+2,dim)!=0]
    events_out[:, 2::dim] = pz
    return reduce(events_out) if reduce_data else events_out

def EpppToPTPhiPZ(events, reduce_data, dim=4, include_masses=False):
    pt = get_pt(events, dim)
    phi = get_phi(events, dim)
    pz = events[:, 3::dim]
    mass = get_mass(events, dim)

    dim_out = dim - int(not include_masses)
    events_out = np.empty((events.shape[0], 3*dim_out))
    events_out[:,0::dim_out] = pt
    events_out[:,1::dim_out] = phi
    events_out[:,2::dim_out] = pz
    if include_masses:
        if reduce_data:
            print("Warning, include_masses and reduce_data are both set, ignoring reduce_data")
        events_out[:, 3::dim_out] = mass
    elif reduce_data:
        events_out = reduce(events_out)

    return events_out


def PTPhiPZToEppp(events, masses, dim = 3):
    """reconstruct E, px, py, pz from pt, eta, phi and the particle masses"""
    if events.shape[1] == 7:
        events = reconstruct(events)
    px, py = np.cos(events[:,1::dim]) * events[:,0::dim], np.sin(events[:,1::dim]) * events[:,0::dim]
    pz = events[:, 2::dim]
    #E ** 2 = M ** 2 + P ** 2
    E = np.sqrt(masses ** 2 + px ** 2 + py ** 2 + pz ** 2)
    events_out = np.empty((events.shape[0], 12))
    events_out[:,0::4] = E
    events_out[:,1::4] = px
    events_out[:,2::4] = py
    events_out[:,3::4] = pz
    return events_out

def EpppToPTdelPhiETA(events, reduce_data, dim=4, include_masses=False):
    events_out = EpppToPTPhiEta(events, reduce_data, include_masses=include_masses)
    if not include_masses and dim == 4:
        dim = 3
    phi1, phi2, phi3 = events_out[:,1], events_out[:,1+dim], events_out[:,1+2*dim]
    #delphi12, delphi23 = phi1 - phi2, phi2 - phi3
    delphi12 = (phi1 - phi2 + np.pi) % (2*np.pi) - np.pi
    delphi23 = (phi2 - phi3 + np.pi) % (2*np.pi) - np.pi
    events_out[:,1], events_out[:,1+2*dim] = delphi12, delphi23
    return np.concatenate((events_out[:,:1+dim], events_out[:,2+dim:]), axis=-1)


def get_pt(events, dim):
    return np.sqrt(events[:,1::dim] **2 + events[:,2::dim] **2)

def get_phi(events, dim):
    return np.arctan2(events[:,1::dim], events[:,2::dim])

def get_eta(events, dim, eps = 1e-15):
    # 0.5 * log(sqrt(p + pz/p - pz)), taken from https://en.wikipedia.org/wiki/Pseudorapidity
    # and https://github.com/ramonpeter/phasespace/blob/9bdf4083d8cf9f258e371645a89fb8d7788895bd/observables.py;
    Ps = np.sqrt(np.square(events[:,1::dim]) + np.square(events[:,2::dim]) + np.square(events[:,3::dim]))
    eta = 0.5 * (np.log(np.clip(np.abs(Ps + events[:,3::dim]), eps, None)) -
                 np.log(np.clip(np.abs(Ps - events[:,3::dim]), eps, None)))
    return np.array(eta)

def get_mass(events, dim):
    #M ** 2 = E ** 2 - P ** 2
    return np.sqrt(np.clip(events[:,0::dim] ** 2 - events[:,1::dim] ** 2 - events[:,2::dim] ** 2 - events[:,3::dim] ** 2, 1e-6, None))

# def reduce(events):
#     """Reduce a 9-dimensional event vector to 7 dimensional event vector"""
#     return events[:,2:]
#
# def reconstruct(events):
#     """A function to recalculate dropped observables from a 7-vector"""
#     phi1, phi2 = events[:, 2], events[:, 5]
#     pt1, pt2 = events[:, 1], events[:, 4]
#     px1, py1 = np.cos(phi1)*pt1, np.sin(phi1)*pt1
#     px2, py2 = np.cos(phi2)*pt2, np.sin(phi2)*pt2
#     pt = np.sqrt((px1 + px2)**2 + (py1 + py2)**2)
#     phi = np.arccos(np.abs(px1 + px2) / pt)
#     phi += np.pi/2 * ((px1 + px2) < 0) + np.pi * ((py1 + py2) < 0) - np.pi
#     events_out = np.empty((events.shape[0], 9))
#     events_out[:,0] = pt
#     events_out[:,1] = phi
#     events_out[:,2:] = events
#     return events_out

def reduce(events):
    """Reduce a 9-dimensional event vector to 7 dimensional event vector"""
    events = np.concatenate((events[:,:6], events[:,8:]), axis=-1)
    return events

def reconstruct(events):
    """A function to recalculate dropped observables from a 7-vector"""
    phi1, phi2 = events[:, 1], events[:, 4]
    pt1, pt2 = events[:, 0], events[:, 3]
    px1, py1 = np.cos(phi1)*pt1, np.sin(phi1)*pt1
    px2, py2 = np.cos(phi2)*pt2, np.sin(phi2)*pt2
    pt = np.sqrt((px1 + px2)**2 + (py1 + py2)**2)
    phi = np.arctan2(-(py1 + py2), -(px1 + px2))
    events_out = np.empty((events.shape[0], 9))
    events_out[:,:6] = events[:,:6]
    events_out[:,6] = pt
    events_out[:,7] = phi
    events_out[:,8:] = events[:,6:]
    return events_out

def apply_scale(events, format, downscale=True, upscale=None, reduce_data=False):
    if downscale:
        if format == 1:
            scale = np.std(events) #EPPP
            events= events/scale
        elif format == 2:
            ind = np.array([0, 3, 6]) #PT
            if reduce_data:
                ind = np.array([0, 3]) #PT
            scale = np.std(events[:, ind])
            print("Scaling: {}".format(scale))
            events[:, ind] = events[:, ind]/scale
        elif format == 3:
            ind = np.array([0, 3, 4, 7, 8, 11]) #PT M
            scale = np.std(events[:, ind])
            events[:, ind] = events[:, ind]/scale
        elif format == 4:
            ind = np.array([0, 2, 3, 5, 6, 8]) #PT PZ
            if reduce_data:
                ind = np.array([0, 2, 3, 5, 6]) #PT PZ
            scale = np.std(events[:, ind])
            events[:, ind] = events[:, ind]/scale
        elif format == 5:
            ind = np.array([0, 2, 3, 4, 6, 7, 8, 10, 11 ]) #PT PZ M
            scale = np.std(events[:, ind])
            events[:, ind] = events[:, ind]/scale
        elif format == 6:
            ind = np.array([0, 3, 5]) #PT
            scale = np.std(events[:, ind])
            events[:, ind] = events[:, ind]/scale
        elif format in [7, 8, 9]:
            scale = 1
        elif format in [10,11,12,13]:
            dim = round(events.shape[1] / 3)
            if dim > 2:
                # pt_min = np.min(events[:,0::dim]) - 1e-6
                # events[:,0::dim] = np.log(events[:,0::dim] - pt_min)
                # events[:,1::dim] = np.arctanh(events[:,1::dim] / np.pi)
                pt3_ = 6 if format != 13 else 5
                phi_ = [1, 4, 7] if format != 13 else [1, 6]
                pt_min = np.min(events[:,pt3_]) - 5e-5 #log(5e-5) is in bounds of -10
                events[:,0] = np.log(events[:,0] - events[:,3])
                events[:,3] = np.log(events[:,3] - events[:,pt3_])
                events[:,pt3_] = np.log(events[:,pt3_] - pt_min)
                if format != 13:
                    events[:,phi_] = np.arctanh(events[:,phi_] / np.pi)
                #else:
                    #events[:,phi_] = np.arctanh(events[:,phi_] / (2*np.pi))
            else:
                pt_min = np.array(0.)
                events[:,0::dim] = np.arctanh(events[:,0::dim] / np.pi)
            means = np.mean(events, axis=0)
            stds = np.std(events, axis=0)
            events = (events - means) / stds
            scale = np.concatenate([pt_min[None], means, stds], axis=0)
            print("SCALE VECTOR: ", scale)
        return events, scale
    else:
        assert not (upscale is None), "Need scale for upscaling, but got None"
        if format == 1:
            events= events*upscale
        elif format == 2:
            ind = np.array([0, 3, 6]) #PT
            events[:, ind] = events[:, ind]*upscale
        elif format == 3:
            ind = np.array([0, 3, 4, 7, 8, 11]) #PT M
            events[:, ind] = events[:, ind]*upscale
        elif format == 4:
            ind = np.array([0, 2, 3, 5, 6, 8]) #PT PZ
            events[:, ind] = events[:, ind]*upscale
        elif format == 5:
            ind = np.array([0, 2, 3, 4, 6, 7, 8, 10, 11 ]) #PT PZ M
            events[:, ind] = events[:, ind]*upscale
        elif format == 6:
            ind = np.array([0, 3, 5]) #PT M
            events[:, ind] = events[:, ind]*upscale
        elif format in [10,11,12,13]:
            dim = round(events.shape[1] / 3)
            pt_min = upscale[0]
            means = upscale[1:1+events.shape[1]]
            stds = upscale[1+events.shape[1]:1+2*events.shape[1]]
            events = events * stds + means
            if dim > 2:
                # events[:,0::dim] = np.exp(events[:,0::dim]) + pt_min
                pt3_ = 6 if format != 13 else 5
                phi_ = [1, 4, 7] if format != 13 else [1, 6]
                events[:,pt3_] = np.exp(events[:,pt3_]) + pt_min
                events[:,3] = events[:,pt3_] + np.exp(events[:,3])
                events[:,0] = events[:,3] + np.exp(events[:,0])
                if format != 13:
                    events[:,phi_] = np.tanh(events[:,phi_]) * np.pi
                #else:
                    #events[:,phi_] = np.tanh(events[:,phi_]) * 2*np.pi
            else:
                events[:,0::dim] = np.tanh(events[:,0::dim]) * np.pi
        return events, upscale

def get_M_ll(events_in, masses=0.1):
    events = events_in[:,:12].copy()
    events[:,1] = np.random.uniform(0, 2 * np.pi, size=(events.shape[0],))
    events[:,5::4] = events[:, 5::4] + events[:,1,None]

    events_Eppp = PTPhiEtaToEppp(events,masses=masses,dim=4)

    p1p2 = events_Eppp[:, 0] * events_Eppp[:, 4] - (events_Eppp[:, 1] * events_Eppp[:, 5] + events_Eppp[:, 2] *
                                                    events_Eppp[:, 6] + events_Eppp[:, 3] * events_Eppp[:, 7])
    return np.sqrt(2*masses**2+2*p1p2)
