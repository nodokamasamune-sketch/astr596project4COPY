# log likelihood for supernova data

#imports
import numpy as np
from data import data_points_reader, covariance_reader



def log_likelihood(theta, z_data, mu_data, Cinv):

    

    pass

def log_prior(theta):
    pass

def log_posterior(theta, z_data, mu_data, Cinv):
    pass


'''
make into class?

Invert C once at the start using np.linalg.inv(C), then reuse Cinv

Use flat (uniform) priors with physical bounds (think about reasonable ranges from the Scientific Background)

Return -np.inf for parameters outside prior bounds
'''