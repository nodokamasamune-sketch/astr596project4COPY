# tests/hmc_tuning_plot.py
import os
import numpy as np
import matplotlib.pyplot as plt

from src.hmc import grad_log_posterior, leapfrog, hamiltonian, hamiltonian_monte_carlo
from src.likelihood import log_posterior
from src.data_reader import get_data

# -------------------
# Directory setup
# -------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "..", "outputs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# -------------------
# HMC parameters
# -------------------
z_data, mu_data, Cinv = get_data()

epsilon = 0.01   # leapfrog step size
L = 10           # number of leapfrog steps
n_steps = 1000   # number of HMC iterations
theta_init = np.array([0.3, 0.7])

# -------------------
# Run HMC sampler
# -------------------
chain, deltas, accept_rate = hamiltonian_monte_carlo(
    log_post_fn=log_posterior,
    theta_init=theta_init,
    epsilon=epsilon,
    L=L,
    n_steps=n_steps,
    args=(z_data, mu_data, Cinv)
)

print(f"Acceptance rate: {accept_rate:.3f}")
deltas = np.asarray(deltas)

# -------------------
# Plot ΔH vs iteration
# -------------------
plt.figure(figsize=(8, 4.5))
plt.plot(np.arange(len(deltas)), deltas, lw=0.8, color="tab:blue", alpha=0.7)
plt.axhline(0, color="k", lw=1, linestyle="--", alpha=0.6)
plt.axhline(1, color="tab:red", lw=0.8, linestyle=":", alpha=0.5)
plt.axhline(-1, color="tab:red", lw=0.8, linestyle=":", alpha=0.5)
plt.xlabel("Iteration", fontsize=12)
plt.ylabel(r"$\Delta H$ (Hamiltonian difference)", fontsize=12)
plt.title("ΔH vs Iteration (Energy Conservation Diagnostic)")
plt.grid(alpha=0.3)
plt.tight_layout()

delta_iter_path = os.path.join(FIG_DIR, "deltaH_vs_iteration.png")
plt.savefig(delta_iter_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {delta_iter_path}")

# -------------------
# Plot ΔH histogram
# -------------------
plt.figure(figsize=(7, 5))
plt.hist(deltas, bins=40, color="tab:orange", alpha=0.75, edgecolor="k")
plt.axvline(0, color="k", lw=1, linestyle="--", alpha=0.6)
plt.axvline(1, color="tab:red", lw=0.8, linestyle=":", alpha=0.6)
plt.axvline(-1, color="tab:red", lw=0.8, linestyle=":", alpha=0.6)
plt.xlabel(r"$\Delta H$ (Hamiltonian difference)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.title("Histogram of ΔH (Energy Conservation Check)")
plt.grid(alpha=0.3)
plt.tight_layout()

delta_hist_path = os.path.join(FIG_DIR, "deltaH_histogram.png")
plt.savefig(delta_hist_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {delta_hist_path}")
