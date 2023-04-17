import numpy as np
import torch, time
from Source.Util.util import get

'''
Methods for discretization in the GPT-style models
'''

''' #old implementation that puts the mean in the middle of the bin
def create_bins_weighted(data, n_bins):
    assert len(data.shape) == 1
    n = len(data)
    bin_edges_indices = np.round(np.linspace(0, n-1, n_bins+1)).astype(int)
    data_sort = np.sort(data)
    bin_edges = data_sort[bin_edges_indices]
    bin_means = (bin_edges[1:]+bin_edges[:-1])/2
    return bin_edges, bin_means
'''
def create_bins_weighted(data, n_bins):
    '''
    Helper function to create weighted (=same amount of events in each bin) bins
    :data: Data to create bins on, format is (n_events)
    :n_bins: Number of bins to be created
    :returns: Mean values of the created bins
    '''
    assert len(data.shape) == 1
    n = len(data)
    bin_edges_indices = np.round(np.linspace(0, n-1, n_bins+1)).astype(int)
    data_sort = np.sort(data)
    bin_edges = data_sort[bin_edges_indices]
    bin_means = np.array([np.mean(data_sort[bin_edges_indices[i]:bin_edges_indices[i+1]]) 
                          for i in range(n_bins)])
    #bin_means = (bin_edges[:-1]+bin_edges[1:])/2 #cheap alternative
    return bin_edges, bin_means

def create_bins_equidistant(data, n_bins):
    '''
    Helper function to create bins with equidistant bin size
    :data: Data to create bins on, format is (n_events)
    :n_bins: Number of bins to be created
    :returns: Mean values of the created bins
    '''
    assert len(data.shape) == 1
    bin_edges = np.linspace(np.min(data), np.max(data), n_bins+1)
    bin_means = np.zeros(n_bins)
    for i in range(n_bins):
        idx = np.where(np.array(bin_edges[i]<data) * np.array(data<bin_edges[i+1]))
        if(np.shape(data[idx])[0]==0): #no events in the bin
            bin_means[i] = (bin_edges[i+1]+bin_edges[i])/2
        else:
            bin_means[i] = np.mean(data[idx])
    #bin_means = (bin_edges[:-1]+bin_edges[1:])/2 #cheap alternative
    return bin_edges, bin_means

def discretize_one(data, bin_edges):
    '''    
    Helper function to discretize one batch of data points
    :data: Data points to be discretized, format is (batch_size)
    :bin_means, bin_edges: Parameters to be used in the discretization
    :discretize: Specify type of discretization (needed in discretize_one)
    :returns: Bin indices in which the given data points are located
    '''
    data_idx = np.zeros_like(data, dtype="int")
    for i in range(len(bin_edges)-1):
        idx = np.where(np.array(bin_edges[i]<data) * np.array(data<bin_edges[i+1])) #indices of data points in the current bin
        data_idx[idx]=i
    return data_idx

def discretize_in_batches(data, bin_edges, batch_size):
    '''
    Helper function to discretize the given data points, calling the discretize_one() method batch-wise
    :data: Data to be discretized, format is (n_events)
    :bin_means, bin_edges: Parameters to be used in the discretization
    :discretize: Specify type of discretization (needed in discretize_one)
    :batch_size: Size of batches in which the data should be discretized (should be chosen such that the storage is not overloaded)
    :returns: Bin indices in which the given data points are located
    '''
    n_sections = data.shape[0] // batch_size + 1
    idx_split = np.array_split(np.arange(data.shape[0]), n_sections)
    data_idx = np.concatenate([discretize_one(data[idx], bin_edges) for idx in idx_split])
    return data_idx

def discretize(data, params, n_jets=2):
    '''
    Main method for the discretization, called in the preprocessing() method
    :data: Data to be discretized, shape is (n_events, n_channels)
    :params: Param dict specifying how the discretization is performed
    :n_jets: Number of jets in the generated events, required because this is the first condition
    :returns: Discretized data (bin indices), together with the bin means used to undo the discretization
    '''
    n_bins = get(params, "n_bins", int(params["n_head"])*int(params["n_per_head"]))
    batchsize_discretize = get(params, "batchsize_discretize", 100000)
    discretize = params["discretize"]
    
    bin_edges = np.zeros((n_bins+1, np.shape(data)[1]))
    bin_means = np.zeros((n_bins, np.shape(data)[1]))
    data_ret = np.zeros((np.shape(data)[0], np.shape(data)[1]+1), dtype="int")
    data_ret[:,0] = n_jets
    for i in range(len(data[0,:])):
        dataFlattened = data[:, i].flatten()
        if(discretize==1): #equidistant bins
            bin_edges[:,i], bin_means[:,i] = create_bins_equidistant(dataFlattened, n_bins)    
        elif(discretize == 2): #non-equidistant bins, same amount of events in each bin
            bin_edges[:,i], bin_means[:,i] = create_bins_weighted(dataFlattened, n_bins)
        else:
            raise ValueError(f"discretize: Discretize={discretize} not implemented. Defaulting to no discretization")
        data_ret[:,i+1] = discretize_in_batches(dataFlattened, bin_edges[:,i], batchsize_discretize)
    return data_ret, bin_edges, bin_means

def linearNoise(n, dx, dy, h):
    '''Create linear noise with the specified properties from flat noise using the transformation method'''
    x = np.random.rand(n)
    y = h*dx/dy * ( (1+dy*(dy+2*h)/h**2 * x)**.5 -1)
    return y

def undo_discretize(samples, params, bin_edges, bin_means):
    '''
    Main method to undo the discretization, called in the undo_preprocessing() method
    :samples: Samples to be undiscretized, shape is (n_events, n_channels)
    :params: Param dict specifying how the discretization is performed
    :bin_means: Bin means needed to undo the discretization, shape is (n_bins, n_channels)
    :returns: Undiscretized version of the samples
    '''
    add_noise = params["add_noise"]
    discretize = params["discretize"]
    n_bins = get(params, "n_bins", int(params["n_head"])*int(params["n_per_head"]))
    
    samples_ret = np.zeros((np.shape(samples)[0], np.shape(samples)[1]-1), dtype="float")
    assert bin_means is not None, "undo_discretize: missing argument bin_means"
    assert bin_edges is not None, "undo_discretize: missing argument bin_edges"
    for i in range(np.shape(samples)[1]-1):
        if(add_noise == 0): #no noise
            samples_ret[:,i] = bin_means[samples[:,i+1],i]
        elif(add_noise == 1): #flat noise
            delta = bin_edges[1:,i]-bin_edges[:-1,i]
            for j in range(n_bins):
                idx = np.where(bin_edges[samples[:,i+1],i]==bin_edges[j,i])[0]
                samples_ret[idx,i] = bin_edges[samples[idx,i+1],i] + delta[j] * np.random.rand(np.shape(idx)[0])
        elif(add_noise == 2): #linear noise
                h, _ = np.histogram(bin_means[samples[:,i+1],i], bins=bin_edges[:,i])
                h = h / (bin_edges[1:,i]-bin_edges[:-1,i]) #n_bins
                dx = bin_means[1:,i]-bin_means[:-1,i] #n_bins-1
                dy = h[1:]-h[:-1] #n_bins-1
                
                idx = np.where(bin_means[samples[:,i+1],i]==bin_means[0,i])[0]
                rand = np.random.rand(np.shape(idx)[0])
                idx_here = idx[np.where(rand<.5)]
                idx_next = idx[np.where(rand>=.5)]
                samples_ret[idx_here,i] = bin_edges[0,i] + np.random.rand(np.shape(idx_here)[0]) * (bin_means[0,i]-bin_edges[0,i])
                for j in range(n_bins-1):
                    idx = np.where(bin_means[samples[:,i+1],i]==bin_means[j+1,i])[0]
                    rand = np.random.rand(np.shape(idx)[0])
                    idx_here = np.append(idx_next, idx[np.where(rand<.5)])
                    idx_next = idx[np.where(rand>=.5)]
                    samples_ret[idx_here,i] = bin_means[j,i] + linearNoise(np.shape(idx_here)[0], dx[j], dy[j], h[j])
                samples_ret[idx_next,i] = bin_means[n_bins-1,i] + np.random.rand(np.shape(idx_next)[0]) * (bin_edges[n_bins,i]-bin_means[n_bins-1,i])
        else:
            if i==0: #only print the warning once
                print(f"undo_discretize: add_noise={add_noise} not implemented yet. Defaulting to add_noise=0.")
            samples_ret[:,i] = bin_means[samples[:, i+1],i]
    return samples_ret
