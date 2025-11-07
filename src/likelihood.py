# log likelihood for supernova data

#imports
import numpy as np
#from src.data_reader import get_data
from src.cosmology import distance_modulus



def log_likelihood(theta, z_data, mu_data, Cinv):
    # log likelihood for gaussian errors with covariance C eqn:
    #ln L(theta) = - 1/2 n-sum (1, j = 1) r_i [C-1]_ij r_j
    #r_i = mu_i^obs - mu_i^theory(z_i;theta)
    Omega_m, h = theta

    # theoretical mu
    mu_theory = np.array([distance_modulus(z, Omega_m, h=h, Pen=True) for z in z_data])
    
    # residuals
    r = mu_data - mu_theory

    loglikelihood = - 0.5 * np.dot(r, np.dot(Cinv, r))

    return loglikelihood


def log_prior(theta):
    # prior probabilities
    # use assertions
    # sample from uniform distribution and take log + vectorize
    Omega_m, h = theta
    if 0.0 < Omega_m < 1.0 and 0.4 < h < 1.0:
        return 0.0 # log (1.0)
    else:
        return -np.inf

def log_posterior(theta, z_data, mu_data, Cinv):
    # way to sample the posterior

    lp = log_prior(theta)

    if not np.isfinite(lp):
        return -np.inf
    llh = log_likelihood(theta, z_data, mu_data, Cinv)

    return lp + llh

'''
make into class? - no

Invert C once at the start using np.linalg.inv(C), then reuse Cinv - yes

Use flat (uniform) priors with physical bounds (think about reasonable ranges from the Scientific Background) omega=0.3, h= [0.4, 1.0]

Return -np.inf for parameters outside prior bounds
'''