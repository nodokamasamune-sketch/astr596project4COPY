# log likelihood for supernova data

#imports
import numpy as np
from data import data_points_reader, covariance_reader



def log_likelihood(theta, z_data, mu_data, Cinv):
# log likelihood for gaussian errors with covariance C eqn:
#ln L(theta) = - 1/2 n-sum (1, j = 1) r_i [C-1]_ij r_j
#r_i = mu_i^obs - mu_i^theory(z_i;theta)


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