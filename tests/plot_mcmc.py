import matplotlib.pyplot as plt
import numpy as np
import os
from src.mcmc import metropolis_hastings
from src.likelihood import log_posterior
from src.data_reader import get_data



z_data, mu_data, Cinv = get_data()
theta_init = np.array([0.5, 0.5])
n_steps = 50000
sigmas = np.array([0.039, 0.015])
chain, acc_rate = metropolis_hastings(log_posterior, theta_init, sigmas=sigmas, n_steps=50000, args=(z_data, mu_data, Cinv))



steps = np.arange(len(chain))



# change to log
# do a contour plot

plt.figure(figsize=(6,5))
scatter = plt.scatter(
    chain[:, 0], chain[:, 1],
    c=np.log10(steps), cmap='plasma', s=10, alpha=0.7
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

from src.diagnostics import gelman_rubin
def plot_rhat_convergence(chains, plot_dir):
    """
    Plot split-Gelman-Rubin (R̂) vs iteration for convergence monitoring.

    Parameters
    ----------
    chains : ndarray
        Array of shape (n_chains, n_steps, n_params)
    plot_dir : str
        Directory to save the output figure
    """
    n_chains, n_steps, n_params = chains.shape
    step_grid = np.linspace(100, n_steps, 20, dtype=int)  # evaluate R̂ at 20 evenly spaced points

    rhat_vals = np.zeros((len(step_grid), n_params))

    # Compute R̂ progressively as chains grow
    for i, n in enumerate(step_grid):
        truncated = chains[:, :n, :]
        Rhat = gelman_rubin(truncated)
        rhat_vals[i, :] = Rhat

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(step_grid, rhat_vals[:, 0], label=r"$\Omega_m$", color='tab:blue')
    plt.plot(step_grid, rhat_vals[:, 1], label=r"$h$", color='tab:orange')
    plt.axhline(1.0, color='black', linestyle='--', linewidth=1, label="Ideal R̂ = 1.0")
    plt.axhline(1.1, color='gray', linestyle=':', linewidth=1, label="R̂ = 1.1 (convergence threshold)")

    plt.xlabel("Iteration")
    plt.ylabel("Split R̂")
    plt.title("Gelman–Rubin Convergence Diagnostic (Split R̂ vs Iteration)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Save plot
    save_path = os.path.join(plot_dir, "gelman_rubin_convergence.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved Gelman–Rubin convergence plot to: {save_path}")


