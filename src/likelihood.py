# log likelihood for supernova data

#imports
import numpy as np
from data import data_points_reader, covariance_reader
from cosmology import luminosity_distance, distance_modulus, intd, eta

dat = data_points_reader('../data/jla_mub.txt')
z_data = dat['z']
mu_data = dat['mu']
c = covariance_reader('../data/jla_mug_covmatrix.txt')
Cinv = np.linalg.inv(C)

def log_likelihood(theta, z_data, mu_data, Cinv):
    # log likelihood for gaussian errors with covariance C eqn:
    #ln L(theta) = - 1/2 n-sum (1, j = 1) r_i [C-1]_ij r_j
    #r_i = mu_i^obs - mu_i^theory(z_i;theta)
    Omega_m, h = theta

    loglikelihood = 0

    for i in range(len(z_data)):
        mu_theoryi = distance_modulus(z_data[i], Omega_m, h=h, Pen=True)
        ri = mu_data[i] - mu_theoryi
        for j in range(len(z_data)):
            mu_theoryj = distance_modulus(z_data[j], Omega_m, h=h, Pen=True)
            rj = mu_data[j] - mu_theoryj

            llh = ri * Cinv[i][j] * rj
            loglikelihood = loglikelihood + llh

    return - 1/2 * loglikelihood


def log_prior(theta):
    # prior probabilities
    Omega_m, h = theta
    pass

def log_posterior(theta, z_data, mu_data, Cinv):
    # way to sample the posterior
    pass


'''
make into class?

Invert C once at the start using np.linalg.inv(C), then reuse Cinv

Use flat (uniform) priors with physical bounds (think about reasonable ranges from the Scientific Background)

Return -np.inf for parameters outside prior bounds
'''