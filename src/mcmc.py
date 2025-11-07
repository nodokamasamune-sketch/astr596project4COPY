#metropolis-hastings MCMC

# general package imports
import numpy as np
from scipy.stats import multivariate_normal
from data_reader import get_data
np.random.seed(42)

z_data, mu_data, Cinv = get_data()


def proposal(theta, proposal_cov):
    theta_new = multivariate_normal.rvs(mean=theta, cov=proposal_cov)
    log_Q_ratio = 0.0  # Symmetric, so Q(θ|θ')/Q(θ'|θ) = 1
    return theta_new, log_Q_ratio



def metropolis_hastings(log_prob_fn, theta_init, sigmas, n_steps, args=()):

    sigma_omega, sigma_h = sigmas
    proposal_cov = np.array([[sigma_omega**2, 0], [0, sigma_h**2]])

    theta = np.array(theta_init, dtype=float)
    chain = np.zeros((n_steps, len(theta)), dtype=float)
    n_accepted = 0

    log_pi_current = log_prob_fn(theta, *args)

    for i in range(n_steps):
        theta_proposed, log_Q_ratio = proposal(theta, proposal_cov)
        log_pi_proposed = log_prob_fn(theta_proposed, *args)

        log_alpha = log_pi_proposed - log_pi_current + log_Q_ratio

        if np.isfinite(log_alpha) and np.log(np.random.uniform()) < log_alpha:
            theta = theta_proposed
            log_pi_current = log_pi_proposed
            n_accepted += 1
        
        chain[i] = theta
    acceptance_rate = n_accepted / n_steps

    return chain, acceptance_rate



