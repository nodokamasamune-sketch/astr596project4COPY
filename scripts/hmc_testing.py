import os
import numpy as np
import corner
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


from scripts.run_hmc import run_hmc
#from scripts.plotting import _ensure_dir, plot_trace_chains, plot_autocorrelation_compare, plot_corner_comparison, plot_data_vs_model



# Default plot styles (match your project style)
MIN_FONT = 12


def plot_trace_single(chains, sampler_name="MCMC"):
    """
    Generate trace plots for a single sampler (MCMC or HMC).

    Parameters
    ----------
    chains : ndarray, shape (m, n, 2)
        Array of posterior samples with shape:
        m = number of chains, n = samples per chain, 2 = parameters (Ωₘ, h).
    sampler_name : str
        Name of the sampler ("MCMC" or "HMC"), used for labeling and filenames.

    Outputs
    -------
    Saves two figures (Omega_m and h traces) in ../outputs/figures.
    """
    # Safety checks
    m, n, p = chains.shape
    if p != 2:
        raise ValueError("Expected chains with 2 parameters: Ωₘ and h")

    # Set up directories
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    FIG_DIR = os.path.join(BASE_DIR, "..", "outputs", "figures")
    os.makedirs(FIG_DIR, exist_ok=True)

    # Parameter info
    param_labels = [r"$\Omega_m$", r"$h$"]
    safe_labels = ["Omega_m", "h"]

    # Generate color palette (unique color per chain)
    cmap = plt.cm.get_cmap("tab10", m)

    # Create one figure per parameter
    for j, (label, safe_label) in enumerate(zip(param_labels, safe_labels)):
        fig, ax = plt.subplots(figsize=(8, 4))
        for i in range(m):
            color = cmap(i)
            ax.plot(
                np.arange(n),
                chains[i, :, j],
                lw=0.8,
                alpha=0.8,
                label=f"Chain {i+1}",
                color=color
            )

        ax.set_xlabel("Iteration", fontsize=MIN_FONT)
        ax.set_ylabel(label, fontsize=MIN_FONT)
        ax.set_title(f"{sampler_name} Trace - {safe_label}", fontsize=MIN_FONT + 1)
        ax.legend(fontsize=MIN_FONT - 2)
        ax.grid(alpha=0.3)
        plt.tight_layout()

        # Save figure with ASCII-only filename
        fname = f"{sampler_name.lower()}_trace_{safe_label}.png"
        path = os.path.join(FIG_DIR, fname)
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved: {path}")



chains_hmc, lags, rhos, ds, accrates = run_hmc(n_steps=2000, m=4, epsilon=0.01, L=9)
plot_trace_single(chains_hmc, sampler_name="hmc")