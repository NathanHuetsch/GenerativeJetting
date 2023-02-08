import numpy as np
import matplotlib.pyplot as plt
#import hdf5plugin
import h5py


class Dataset:

    def __init__(self, file_path, fraction=None, conditional=False):
        self.file_path = file_path
        self.data = h5py.File(self.file_path)['events']['block0_values']
        if fraction is not None:
            n = round(fraction * self.data.shape[0])
            self.data = self.data[:n]
        self.entries = [np.count_nonzero(event) for event in self.data]
        if conditional:
            self.z_1, self.z_2, self.z_3 = self.get_conditional_dataset()
        else:
            self.z_1, self.z_2, self.z_3 = self.get_individual_datasets()

    def get_individual_datasets(self):
        z_1 = []
        z_2 = []
        z_3 = []
        failures = []
        for i, event in enumerate(self.data):
            n = self.entries[i]
            x = event[:n]
            if n == 12:
                z_1.append(x)
            elif n == 16:
                z_2.append(x)
            elif n == 20:
                z_3.append(x)
            else:
                failures.append(n)

        z_1 = np.array(z_1)
        z_2 = np.array(z_2)
        z_3 = np.array(z_3)

        return z_1, z_2, z_3

    def get_conditional_dataset(self):
        z_1 = []
        z_2 = []
        z_3 = []
        for i, event in enumerate(self.data):
            m = self.entries[i] / 4 - 2

            c_1 = 12
            c_2 = 16
            x = event[:c_1]
            x = np.append(x, m)
            z_1.append(x)

            if m != 1:

                y = event[:c_2]
                y = np.append(y, m)
                z_2.append(y)

            if m == 3:
                z = event
                z = np.append(z, m)
                z_3.append(z)

        z_1 = np.array(z_1)
        z_2 = np.array(z_2)
        z_3 = np.array(z_3)

        return z_1, z_2, z_3
