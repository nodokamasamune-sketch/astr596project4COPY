# HMC energy conservation: ΔH distribution centered near 0, acceptance 60-80% when tuned

# find optimal epsilon and L step size

# epsilon : delta H < 1 for most samples, and acceptance rate 60 - 90 
# if acceptance too high --> increase epsilon
# if acceptance too low --> decerase epsilon

# L :
# L too short --> proposals close to current state; high autocorelation
# L too long --> trajectory curves back & wasted computation time
# L tuned: trajectory travels approximately one autocorrelation length in parameters space
# Roughly L * epsilon ~ diameter of posterior typical set
# Rule of thumb: start with L ~ 10 - 50; adjust based on ACF
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from src.hmc import grad_log_posterior, leapfrog, hamiltonian, hamiltonian_monte_carlo

from src.likelihood import log_posterior
#def log_posterior(theta, z_data, mu_data, Cinv):

from src.data_reader import get_data

# Get directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build a path to ../outputs/tables relative to the script
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "outputs", "tables")

# Make sure the directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

z_data, mu_data, Cinv = get_data()

# tuned guess:
ep = 0.01
L = 10


n_steps = 5000
theta_init = np.array([0.3, 0.7])

chain, deltas, accept_rate = hamiltonian_monte_carlo(log_posterior, theta_init, n_steps, ep, L, args=(z_data, mu_data, Cinv))

print(accept_rate)
print(np.mean(deltas))
print(np.median(deltas))
