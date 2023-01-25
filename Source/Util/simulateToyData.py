import numpy as np
from Source.Util.util import get

class ToySimulator:
    def __init__(self, params):
        self.params = params
        self.type = get(self.params,"toy_type","ramp")
        self.n_data = get(self.params, "n_data", 1000000)

        if self.type == "camel":
            self.data = self.get_camelback()
        elif self.type == "ramp":
            self.data = self.get_ramps()
        elif self.type == "gauss_sphere":
            self.data = self.get_gauss_sphere()
        else:
            raise ValueError("Type must be either camel, ramp or gauss_sphere")


    def get_camelback(self):
        '''
        Create samples from a n-dimensional camelback, where the camelbacks lie in the "all plus" and "all minus" quadrants
        :n_dim: Number of dimensions in which the camelback lives
        :n_per_dim: Number of samples generated
        :returns: Array filled with samples, format (n_per_dim, n_dim)
        '''
        n_dim = get(self.params, "n_dim", 2)
        mu = get(self.params, "mu", 1)
        sigma = get(self.params, "sigma", .1)

        samples = np.zeros((self.n_data, n_dim))

        plusorminus = np.random.rand(self.n_data)
        idxPlus = np.where(plusorminus<.5)[0]
        idxMinus = np.where(plusorminus>=.5)[0]

        samples[idxPlus,:] = sigma*np.random.randn(len(idxPlus), n_dim) + mu
        samples[idxMinus,:] = sigma*np.random.randn(len(idxMinus), n_dim) - mu
        return samples

    def get_ramps(self):
        '''
        Create samples in a high-dimensional space, following either flat, linear or quadratic distributions.
        :n_flat: Number of flat dimensions
        :n_lin: Number of linear dimensions
        :n_quad: Number of quadratic dimensions
        :returns: Array filled with samples, format (n_per_dim, n_flat+n_lin+n_quad),
        where the flat dimensions come first and are followed by the linear and quadratic dimensions
        '''
        n_flat = get(self.params, "n_flat", 1)
        n_lin = get(self.params, "n_lin", 1)
        n_quad = get(self.params, "n_quad", 0)

        samples = np.zeros((self.n_data, n_flat+n_lin+n_quad))
        for i in range(n_flat):
            samples[:,i] = np.random.rand(self.n_data)
        for i in range(n_lin):
            samples[:,n_flat+i] = (np.random.rand(self.n_data))**(1/2)
        for i in range(n_quad):
            samples[:,n_flat+n_lin+i] = (np.random.rand(self.n_data))**(1/3)
        return samples

    def get_gauss_sphere(self):
        '''
        Create samples from a n-dimensional hollow hypersphere with a gaussian surface
        :n_dim: Number of dimensions in which the hypersphere lives
        :n_samples: Number of samples generated
        :returns: Array filled with samples, format (n_samples, n_dim)
        '''
        n_dim = get(self.params, "n_dim", 2)
        mu = get(self.params, "mu", 1.)
        sigma = get(self.params, "sigma", .1)
        half = get(self.params,"half", False)

        samples = np.zeros((self.n_data, n_dim))

        R = np.abs(sigma * np.random.randn(self.n_data) + mu)
        phi = np.random.rand(self.n_data, n_dim - 1)
        phi[:, 0] *= np.pi if half else 2 * np.pi
        phi[:, 1:] *= np.pi

        # recursively build the coordinate transformation from (R, phi0, phi1...) to (x0, x1, x2...)
        expr = R
        for i in range(n_dim - 1):
            samples[:, i] = expr * np.cos(phi[:, i])
            expr = expr * np.sin(phi[:, i])
        samples[:, -1] = expr
        return samples

    def get_R(self):
        '''Calculate radius of samples following a hypersphere distribution'''
        return np.sum(self.n_data ** 2, axis=1) ** .5