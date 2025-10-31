# load and read files to prepare them for likelihood function

#imports
import numpy as np

def data_points_reader(filename):
    dat = np.genfromtxt(filename, skip_header = 1, names=('z, mu'))

    return dat

def covariance_reader(filename):
    cov = np.genfromtxt(filename)



    cov_matrix = np.empty((31, 31))

        for i in range(31):
            min = 31 * i
            max = 31 * (i + 1)
            cov_matrix[i:31] = cov[min:max]

        return cov_matrix

'''
ask about cov matrix file format: are the first 31 points in the list the first row?
'''