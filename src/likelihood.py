# log likelihood for supernova data

#imports
import numpy as np
from data_reader import data_points_reader, covariance_reader
from cosmology import luminosity_distance, distance_modulus, intd, eta

def data():
    dat = data_points_reader('../data/jla_mub.txt')
    z_data = dat['z']
    mu_data = dat['mu']
    C = covariance_reader('../data/jla_mug_covmatrix.txt')
    Cinv = np.linalg.inv(C)

    return z_data, mu_data, Cinv


class likelihood:
    def __init__(self, z_data, mu_data, Cinv, model_func):
        self.z_data = z_data
        self.mu_data = mu_data
        self.Cinv = Cinv

    def log_prior(self, theta):
        # prior probabilities
        Omega_m, h = theta
        if 0.0 < Omega_m < 1.0 and 0.4 < h < 1.0:
            return 0.0 # log (1.0)
        else:
            return -np.inf
        
    def log_likelihood(self, theta):
        # log likelihood for gaussian errors with covariance C eqn:
        #ln L(theta) = - 1/2 n-sum (1, j = 1) r_i [C-1]_ij r_j
        #r_i = mu_i^obs - mu_i^theory(z_i;theta)

        Omega_m, h = theta
        # theoretical mu
        mu_theory = np.array([self.model_func(z, Omega_m, h=h, Pen=True) for z in self.z_data])
        # residuals
        r = self.mu_data - mu_theory
        
        loglikelihood = - 1/2 * np.dot(r, np.dot(self.Cinv, r))

        return loglikelihood

    def log_posterior(self, theta):
        # way to sample the posterior

        lp = self.log_prior(theta)

        if not np.isfinite(lp):
            return -np.inf
        llh = self.log_likelihood(theta)

        return lp + llh  



'''
make into class?

Invert C once at the start using np.linalg.inv(C), then reuse Cinv

Use flat (uniform) priors with physical bounds (think about reasonable ranges from the Scientific Background)

Return -np.inf for parameters outside prior bounds
'''