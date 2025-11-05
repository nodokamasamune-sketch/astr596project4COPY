import matplotlib.pyplot as plt
import numpy as np

from src.mcmc import metropolis_hastings
from src.likelihood import log_posterior
from src.data_reader import get_data



z_data, mu_data, Cinv = get_data()
theta_init = np.array([0.5, 0.5])
n_steps = 50000
chain, acc_rate = metropolis_hastings(log_posterior, theta_init, sigma_omega=0.008, sigma_h=0.02, n_steps=50000, args=(z_data, mu_data, Cinv))



steps = np.arange(len(chain))

plt.figure(figsize=(6,5))
scatter = plt.scatter(
    chain[:, 0], chain[:, 1],
    c=steps, cmap='plasma', s=10, alpha=0.7
)

plt.xlabel(r'$\Omega_m$')
plt.ylabel(r'$h$')
plt.title('MCMC posterior samples (colored by iteration)')
plt.colorbar(scatter, label='Iteration')
plt.tight_layout()
plt.savefig('posterior_convergence.png')
plt.close()



plt.figure(figsize=(15, 5))
plt.plot(chain[:, 0])
plt.axhline(np.average(chain[-int(n_steps * 0.8):, 0]), color='r', ls='--')
#plt.xlim(0, n_steps)
plt.xlabel('iteration')
plt.ylabel('omega_m')
plt.savefig('omega_m_chain.png')
plt.close();

plt.figure(figsize=(15, 5))
plt.plot(chain[:, 1])
plt.axhline(np.average(chain[-int(n_steps * 0.8):, 1]), color='r', ls='--')
#plt.xlim(0, n_steps)
plt.xlabel('iteration')
plt.ylabel('h')
plt.savefig('h_chain.png')
plt.close();