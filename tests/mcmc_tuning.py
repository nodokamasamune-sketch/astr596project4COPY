# generate sigma values to obtain an acceptance range [0.2, 0.5]
import numpy as np

from src.mcmc import metropolis_hastings
# def metropolis_hastings(log_prob_fn, theta_init, sigmas, n_steps, args=()):

from src.likelihood import log_posterior
#def log_posterior(theta, z_data, mu_data, Cinv):

from src.data_reader import get_data

z_data, mu_data, Cinv = get_data()


sigmas = np.array([0.05, 0.02])

n = 1000


theta_init = np.array([0.3, 0.7])    

acc_rates = []
for i in range(10):
    chain, acc_rate = metropolis_hastings(log_posterior, theta_init, sigmas, n, args=(z_data, mu_data, Cinv))
    acc_rates.append(acc_rate)


while min(acc_rates) < 0.2 or max(acc_rates) > 0.5:
    if min(acc_rates) < 0.2:
        g = np.random.uniform(0.6, 0.9)
        sigmas = sigmas*(g**2)

    if max(acc_rates) > 0.5:
        g = np.random.uniform(1.1, 1.5)
        sigmas = sigmas*(g**2)
    
    acc_rates = []
    for i in range(10):
        chain, acc_rate = metropolis_hastings(log_posterior, theta_init, sigmas, n, args=(z_data, mu_data, Cinv))
        acc_rates.append(acc_rate)

        
print(acc_rates, sigmas)

