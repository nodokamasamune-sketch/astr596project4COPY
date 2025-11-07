# hamiltonina monte carlo
import numpy as np
from likelihood import log_posterior

def grad


def hamiltonian_monte_carlo(log_prob_fn, grad_log_prob_fn, theta_init, epsilon, L, n_steps, args=()):
    theta = theta_init
