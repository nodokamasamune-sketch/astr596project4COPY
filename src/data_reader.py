# load and read files to prepare them for likelihood function

# imports
import numpy as np
import os

def data_points_reader(filename):
    dat = np.genfromtxt(filename, skip_header = 1, names=('z', 'mu'))

    return dat

def covariance_reader(filename):
    # CHECKED loading jla_covmatrix works + fixed np.reshape() 
    cov = np.genfromtxt(filename)
    cov_matrix = cov.reshape(31, 31)


    return cov_matrix

def get_data():
    # CHECKED loading jla_mub and jla_covmatrix works 

    current_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    data_dir = os.path.join(project_root, 'data')


    mubpath = os.path.join(data_dir, 'jla_mub.txt')
    cov_path = os.path.join(data_dir, 'jla_mug_covmatrix.txt')

    dat = data_points_reader(mubpath) 
    z_data = dat['z']
    mu_data = dat['mu']

    C = covariance_reader(cov_path)
    Cinv = np.linalg.inv(C)

    return z_data, mu_data, Cinv


'''
ask about cov matrix file format: are the first 31 points in the list the first row?
'''