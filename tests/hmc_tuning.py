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

from src.hmc import grad_log_posterior, leapfrog, hamiltonian, hamiltonian_monte_carlo

from src.likelihood import log_posterior
#def log_posterior(theta, z_data, mu_data, Cinv):

from src.data_reader import get_data

z_data, mu_data, Cinv = get_data()

# initial guess:
ep = 0.1
L = 30

def energy_cons(log_post_fn,theta_init, epsilon, L, n_steps=100, args=()):
    deltas = []
    for step in range(int(n_steps)):
        p_init = np.random.normal(0, 1, size=theta_init.shape)
        H_init = hamiltonian(theta_init, p_init, log_post_fn, *args)
        theta_prop, p_prop = leapfrog(theta_init, p_init, epsilon, L, log_post_fn, grad_log_posterior, *args)
        H1 = hamiltonian(theta_prop, p_prop, log_post_fn, *args)
        deltas.append(H1 - H_init)
    return np.array(deltas)


deltas = energy_cons(log_posterior, np.array([0.3, 0.7]), ep, L, n_steps=100, args=(z_data, mu_data, Cinv))


steps = np.arange(deltas.size)
plt.plot(steps, deltas)
plt.show()