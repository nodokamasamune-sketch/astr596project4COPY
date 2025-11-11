'''
USED ChatGPT TO WRITE DOC STRINGS AND CLEAN UP PLOTTING FUNCTIONS, GENERATE COLORS, AND LINESTYLES.
'''

import os
import numpy as np
import matplotlib.pyplot as plt
import corner

# Global style and directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "..", "outputs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

MCMC_COLOR = "#1f77b4"
HMC_COLOR = "#d62728"
MIN_FONT = 12
plt.rcParams.update({
    "font.size": MIN_FONT,
    "axes.labelsize": MIN_FONT,
    "legend.fontsize": MIN_FONT,
    "xtick.labelsize": MIN_FONT - 1,
    "ytick.labelsize": MIN_FONT - 1,
    "figure.dpi": 300
})


def _adaptive_scatter_style(n_points):
    """Return (size, alpha) for scatter points based on sample size."""
    if n_points < 1000:
        return 10, 0.7
    elif n_points < 5000:
        return 6, 0.5
    elif n_points < 20000:
        return 3, 0.3
    else:
        return 1, 0.2

# trace plots : 4 plots
def plot_trace_all(chains, method_name="MCMC"):
    """
    Line-based trace plots for all chains and both parameters.
    Uses adaptive alpha and linewidth for readability with long chains.
    """
    param_labels = [r"$\Omega_m$", r"$h$"]
    param_names = ["Omega_m", "h"]

    n_chains, n_samples, _ = chains.shape

    # adapt transparency and line width to chain length
    if n_samples < 1000:
        alpha, lw = 0.9, 1.2
    elif n_samples < 5000:
        alpha, lw = 0.7, 1.0
    elif n_samples < 20000:
        alpha, lw = 0.5, 0.8
    else:
        alpha, lw = 0.3, 0.6

    cmap = plt.cm.get_cmap("tab10", n_chains)

    for j, (p_label, p_name) in enumerate(zip(param_labels, param_names)):
        fig, ax = plt.subplots(figsize=(8, 4))

        for i in range(n_chains):
            ax.plot(
                np.arange(n_samples),
                chains[i, :, j],
                color=cmap(i),
                lw=lw,
                alpha=alpha,
                label=f"{method_name} Chain {i+1}"
            )

        ax.set_xlabel("Iteration", fontsize=MIN_FONT)
        ax.set_ylabel(p_label, fontsize=MIN_FONT)
        ax.set_title(f"{method_name} Trace – {p_name}", fontsize=MIN_FONT + 1)
        ax.legend(fontsize=MIN_FONT - 2)
        ax.grid(alpha=0.3)
        plt.tight_layout()

        fname = f"{method_name.lower()}_trace_{p_name}.png"
        path = os.path.join(FIG_DIR, fname)
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved trace plot: {path}")



# autocorrelation plotting
def plot_autocorrelation_compare(acf_mcmc, acf_hmc):
    """
    Compare autocorrelation using adaptive scatter points for MCMC vs HMC.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    param_names = ["Omega_m", "h"]

    for ax, pname, idx in zip(axes, param_names, [0, 1]):
        # MCMC autocorr
        for lags, rhos in acf_mcmc[idx]:
            s, alpha = _adaptive_scatter_style(len(lags))
            ax.scatter(lags, rhos, color=MCMC_COLOR, alpha=alpha, s=s,
                       label="MCMC" if "MCMC" not in [l.get_label() for l in ax.collections] else "")
        # HMC autocorr
        for lags, rhos in acf_hmc[idx]:
            s, alpha = _adaptive_scatter_style(len(lags))
            ax.scatter(lags, rhos, color=HMC_COLOR, alpha=alpha + 0.1, s=s,
                       marker="x", label="HMC" if "HMC" not in [l.get_label() for l in ax.collections] else "")
        ax.set_xlabel("Lag")
        ax.set_ylabel(r"Autocorrelation $\rho(k)$")
        ax.set_title(f"Autocorrelation – {pname}")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle("Autocorrelation Comparison (MCMC vs HMC)")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    fname = "autocorrelation_comparison.png"
    path = os.path.join(FIG_DIR, fname)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved autocorrelation comparison: {path}")


# corner plot
def plot_corner_compare(samples_mcmc, samples_hmc):
    """
    Overlay corner plots for first chain of MCMC and HMC.
    """
    fig = corner.corner(samples_mcmc,
                        color=MCMC_COLOR,
                        labels=[r"$\Omega_m$", r"$h$"],
                        label_kwargs={"fontsize": MIN_FONT},
                        show_titles=True,
                        title_fmt=".3f",
                        title_kwargs={"fontsize": MIN_FONT - 1},
                        bins=30,
                        smooth=1.0,
                        alpha=0.5)
    corner.corner(samples_hmc,
                  color=HMC_COLOR,
                  fig=fig,
                  bins=30,
                  smooth=1.0,
                  alpha=0.5)
    fig.suptitle("Corner Plot Comparison (MCMC vs HMC)", fontsize=MIN_FONT + 2)

    fname = "corner_comparison.png"
    path = os.path.join(FIG_DIR, fname)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved corner plot comparison: {path}")


# data vs model
def plot_data_vs_model(z_data, mu_data, Cinv, samples_mcmc, samples_hmc,
                       n_samples=80, z_grid=None):
    """
    Adaptive scatter-based posterior predictive plot for data vs model.
    """
    from src.cosmology import distance_modulus

    if z_grid is None:
        z_grid = np.linspace(0, 1.3, 200)

    # observational uncertainties
    try:
        cov = np.linalg.inv(Cinv)
        mu_err = np.sqrt(np.diag(cov))
    except Exception:
        mu_err = 1.0 / np.sqrt(np.diag(Cinv))

    plt.figure(figsize=(8, 6))
    plt.errorbar(z_data, mu_data, yerr=mu_err, fmt="o", color="k", ms=3.5, label="JLA data", alpha=0.8)

    # random posterior draws
    idx_m = np.random.choice(samples_mcmc.shape[0], size=min(n_samples, samples_mcmc.shape[0]), replace=False)
    idx_h = np.random.choice(samples_hmc.shape[0], size=min(n_samples, samples_hmc.shape[0]), replace=False)

    # Adaptive scatter for posterior predictive curves
    s_curve, alpha_curve = _adaptive_scatter_style(len(z_grid))

    for theta in samples_mcmc[idx_m]:
        mu_curve = np.array([distance_modulus(z, *theta) for z in z_grid])
        plt.scatter(z_grid, mu_curve, color=MCMC_COLOR, alpha=alpha_curve, s=s_curve)
    for theta in samples_hmc[idx_h]:
        mu_curve = np.array([distance_modulus(z, *theta) for z in z_grid])
        plt.scatter(z_grid, mu_curve, color=HMC_COLOR, alpha=alpha_curve + 0.05, s=s_curve, marker="x")

    # Median posterior prediction
    med_m = np.median(samples_mcmc, axis=0)
    med_h = np.median(samples_hmc, axis=0)
    mu_med_m = np.array([distance_modulus(z, *med_m) for z in z_grid])
    mu_med_h = np.array([distance_modulus(z, *med_h) for z in z_grid])
    plt.scatter(z_grid, mu_med_m, color=MCMC_COLOR, s=12, label="MCMC median")
    plt.scatter(z_grid, mu_med_h, color=HMC_COLOR, s=12, marker="x", label="HMC median")

    plt.xlabel("Redshift $z$")
    plt.ylabel("Distance modulus μ (mag)")
    plt.title("Data vs Model — Posterior Predictive Samples (adaptive scatter)")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    fname = "data_vs_model_scatter.png"
    path = os.path.join(FIG_DIR, fname)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved adaptive scatter data vs model plot: {path}")
