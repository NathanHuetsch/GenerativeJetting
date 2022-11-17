import numpy as np
import torch

'''
Methods for discretization in the GPT-style models
TBD: Use numpy commands in discretize_wrapper
'''

def create_bins(data, n_bins):
    assert len(data.shape) == 1
    n = len(data)
    bin_edges_indices = np.round(np.linspace(0, n, n_bins+1)).astype(int)
    data_sort = np.sort(data)
    bin_means = np.array([np.mean(data_sort[bin_edges_indices[i]:bin_edges_indices[i+1]]) 
                          for i in range(n_bins)])
    return bin_means
    
def discretize(data, bin_means):
    data_idx = np.argmin(abs(data[:, None] - bin_means[None]), 1)
    return data_idx
    
def discretize_in_batches(data, bin_means, batch_size):
    n_sections = data.shape[0] // batch_size + 1
    idx_split = np.array_split(np.arange(data.shape[0]), n_sections)
    data_idx = np.concatenate([discretize(data[idx], bin_means) for idx in idx_split])
    return data_idx
    
def discretize_wrapper(data, i, nbins, batch_size):
    dataFlattened = data[:, i].flatten()
    bins = create_bins(dataFlattened, nbins)
    tokens = discretize_in_batches(dataFlattened, bins, batch_size)
    if(i%2 == 1):
        tokens += nbins
    tokensTorch = torch.from_numpy(tokens)
    return tokensTorch
