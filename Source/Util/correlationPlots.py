import numpy as np
import matplotlib.pyplot as plt

def correlationPlot(data, filenameOut, var=None, nbins=50):
    '''Correlation plot for one dataset'''
    nvar = len(data[0,:])
    f, axarr = plt.subplots(nvar, nvar, figsize=[30, 30])
    plt.tight_layout()
    
    if(type(var) is np.ndarray):
        assert nvar == len(var), "Variable name count does not match data count"
    
    for i in range(nvar):
        for j in range(nvar):
            plt.grid(True)
            
            if i == j: # diagonal
                xmin = np.amin(data[:,i])
                xmax = np.amax(data[:,i])
                axarr[i][j].hist(data[:,i], bins=nbins, alpha=0.5, color=col[0], range=(xmin, xmax), density=True)
                if(type(var) is np.ndarray):
                    axarr[i][j].set_xlabel(r"${%s}$" % var[i])
                axarr[i][j].grid(True)
            
            elif i>j: # off diagonal
                xs = data[:,i]
                ys = data[:,j]
                
                xmin = max(np.mean(xs) - 2*np.std(xs), min(xs))
                xmax = min(np.mean(xs) + 2*np.std(xs), max(xs))
                ymin = max(np.mean(ys) - 2*np.std(ys), min(ys))
                ymax = min(np.mean(ys) + 2*np.std(ys), max(ys))

                axarr[i][j].hist2d(xs, ys, bins=nbins, range=[[xmin, xmax], [ymin, ymax]], density=True)
                
                if(type(var) is np.ndarray):
                    axarr[i][j].set_xlabel(r"${%s}$" % var[i])
                    axarr[i][j].set_ylabel(r"${%s}$" % var[j])
            elif i<j:
                axarr[i][j].remove()

    f.savefig(filenameOut, format="pdf")
    plt.close()

def correlationPlotDelta(data1, data2, filenameOut, var=None, nbins=50):
    '''Correlation plot for two datasets, with differences of the two datasets in off-diagonal entries'''
    nvar = len(data1[0,:])
    f, axarr = plt.subplots(nvar, nvar, figsize=[30, 30])
    plt.tight_layout()
    
    assert np.shape(data1) == np.shape(data2), "data1 and data2 dont have the same shape"
    if(type(var) is np.ndarray):
        assert nvar == len(var), "Variable name count does not match data count"
    
    for i in range(nvar):
        for j in range(nvar):
            plt.grid(True)
            
            if i == j: # diagonal
                xmin = np.amin([data1[:,i], data2[:,i]])
                xmax = np.amax([data1[:,i], data2[:,i]])
                axarr[i][j].hist(data1[:,i], bins=nbins, alpha=.5, color=col[0], range=(xmin, xmax), density=True)
                axarr[i][j].hist(data2[:,i], bins=nbins, alpha=.5, color=col[1], range=(xmin, xmax), density=True)
                if(type(var) is np.ndarray):
                    axarr[i][j].set_xlabel(r"${%s}$" % var[i])
                axarr[i][j].grid(True)
            
            elif i>j: # off diagonal
                xs = data1[:,i]
                ys = data1[:,j]
                
                xmin = max(np.mean(xs) - 2*np.std(xs), min(xs))
                xmax = min(np.mean(xs) + 2*np.std(xs), max(xs))
                ymin = max(np.mean(ys) - 2*np.std(ys), min(ys))
                ymax = min(np.mean(ys) + 2*np.std(ys), max(ys))
                
                H1, xedges, yedges = np.histogram2d(data1[:,i], data1[:,j], bins=nbins, range=[[xmin, xmax],[ymin, ymax]])
                H1 = H1.T
                H2, _, _ = np.histogram2d(data2[:,i], data2[:,j], bins=nbins, range=[[xmin, xmax],[ymin, ymax]])
                H2 = H2.T
                
                X, Y = np.meshgrid(xedges, yedges)
                dH = np.abs(H1-H2)/(1e-1 + H1+H2)
                
                axarr[i][j].pcolormesh(X, Y, dH, vmin=0., vmax=1., cmap="viridis", rasterized=True)
                
                if(type(var) is np.ndarray):
                    axarr[i][j].set_xlabel(r"${%s}$" % var[i])
                    axarr[i][j].set_ylabel(r"${%s}$" % var[j])
            elif i<j:
                axarr[i][j].remove()

    f.savefig(filenameOut, format="pdf")
    plt.close()

def correlationPlotCompare(data1, data2, filenameOut, var=None, nbins=50):  
    '''Correlation plot for two datasets, with the two datasets in
    different colors in off-diagonal entries'''  
    nvar = len(data1[0,:])
    f, axarr = plt.subplots(nvar, nvar, figsize=[30, 30])
    plt.tight_layout()   
    
    assert np.shape(data1) == np.shape(data2), "data1 and data2 dont have the same shape"
    if(type(var) is np.ndarray):
        assert nvar == len(var), "Variable name count does not match data count"
    
    for i in range(nvar):
        for j in range(nvar):
            plt.grid(True)
            
            if i == j: # diagonal
                xmin = np.amin([data1[:,i], data2[:,i]])
                xmax = np.amax([data1[:,i], data2[:,i]])
                axarr[i][j].hist(data1[:,i], bins=nbins, alpha=.5, color=col[0], range=(xmin, xmax), density=True)
                axarr[i][j].hist(data2[:,i], bins=nbins, alpha=.5, color=col[1], range=(xmin, xmax), density=True)
                if(type(var) is np.ndarray):
                    axarr[i][j].set_xlabel(r"${%s}$" % var[i])
                axarr[i][j].grid(True)
            
            elif i>j: # off diagonal
                xs = data1[:,i]
                ys = data1[:,j]
                
                xmin = max(np.mean(xs) - 2*np.std(xs), min(xs))
                xmax = min(np.mean(xs) + 2*np.std(xs), max(xs))
                ymin = max(np.mean(ys) - 2*np.std(ys), min(ys))
                ymax = min(np.mean(ys) + 2*np.std(ys), max(ys))
                
                H1, xedges, yedges = np.histogram2d(data1[:,i], data1[:,j], bins=nbins, range=[[xmin, xmax],[ymin, ymax]])
                H1 = H1.T
                H2, _, _ = np.histogram2d(data2[:,i], data2[:,j], bins=nbins, range=[[xmin, xmax],[ymin, ymax]])
                H2 = H2.T
                
                X, Y = np.meshgrid(xedges, yedges)
                
                axarr[i][j].pcolormesh(X, Y, H1, alpha=.5, cmap="Reds", rasterized=True)
                axarr[i][j].pcolormesh(X, Y, H2, alpha=.5, cmap="Blues", rasterized=True)
                
                if(type(var) is np.ndarray):
                    axarr[i][j].set_xlabel(r"${%s}$" % var[i])
                    axarr[i][j].set_ylabel(r"${%s}$" % var[j])
            elif i<j:
                axarr[i][j].remove()

    f.savefig(filenameOut, format="pdf")  
    plt.close()
