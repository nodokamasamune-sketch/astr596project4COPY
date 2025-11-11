import numpy as np
import matplotlib.pyplot as plt
import corner
from scripts.run_mcmc import run_mcmc

import numpy as np
import matplotlib.pyplot as plt
import corner
import os

def reweight_with_planck_prior(samples, mean_planck=0.674, sigma_planck=0.005, plot=True):
    """
    Reweight posterior samples by Gaussian Planck prior on h,
    save resulting plots in ../outputs/figures directory.

    Parameters
    ----------
    samples : ndarray, shape (n_samples, 2)
        Posterior samples [Omega_m, h].
    mean_planck : float
        Mean of Planck prior on h.
    sigma_planck : float
        Std of Planck prior on h.
    plot : bool
        If True, generate and save diagnostic plots.

    Returns
    -------
    dict
        Weighted summaries before and after applying the prior.
    """
    omega_m = samples[:, 0]
    h = samples[:, 1]

    # Compute Planck prior weights
    weights = np.exp(-0.5 * ((h - mean_planck) / sigma_planck) ** 2)
    weights /= np.sum(weights)

    def summarize(x, w=None):
        """Compute mean, std, and 16/50/84th percentiles."""
        if w is None:
            mean = np.mean(x)
            std = np.std(x)
            ci = np.percentile(x, [16, 50, 84])
        else:
            mean = np.average(x, weights=w)
            std = np.sqrt(np.average((x - mean) ** 2, weights=w))
            sorted_idx = np.argsort(x)
            cdf = np.cumsum(w[sorted_idx])
            ci = np.interp([0.16, 0.50, 0.84], cdf, x[sorted_idx])
        return dict(mean=mean, std=std, ci=ci)

    # Summaries
    summary_before = {"Omega_m": summarize(omega_m), "h": summarize(h)}
    summary_after = {"Omega_m": summarize(omega_m, weights), "h": summarize(h, weights)}

    # Print summary
    print("\n=== Parameter Summary ===")
    for param in ["Omega_m", "h"]:
        b = summary_before[param]
        a = summary_after[param]
        print(f"\n{param}:")
        print(f"  Before prior: mean={b['mean']:.4f} ± {b['std']:.4f}, 68% CI={b['ci']}")
        print(f"  After prior:  mean={a['mean']:.4f} ± {a['std']:.4f}, 68% CI={a['ci']}")

    if plot:
        # --- Setup save directory ---
        base_dir = os.path.dirname(os.path.abspath(__file__))
        fig_dir = os.path.join(base_dir, "..", "outputs", "figures")
        os.makedirs(fig_dir, exist_ok=True)

        # --- 1) Marginal histograms before/after reweighting ---
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].hist(omega_m, bins=40, density=True, histtype='step', color='tab:blue', label='Before prior')
        axes[0].hist(omega_m, bins=40, weights=weights, density=True, histtype='stepfilled',
                     alpha=0.4, color='tab:orange', label='After Planck prior')
        axes[0].set_xlabel(r'$\Omega_m$')
        axes[0].set_ylabel('Density')
        axes[0].legend()

        axes[1].hist(h, bins=40, density=True, histtype='step', color='tab:blue', label='Before prior')
        axes[1].hist(h, bins=40, weights=weights, density=True, histtype='stepfilled',
                     alpha=0.4, color='tab:orange', label='After Planck prior')
        axes[1].axvline(mean_planck, color='k', linestyle='--', label='Planck prior mean')
        axes[1].set_xlabel('h')
        axes[1].legend()
        plt.suptitle("Effect of Planck Prior on Posterior")
        plt.tight_layout()

        hist_path = os.path.join(fig_dir, "planck_prior_histograms.png")
        plt.savefig(hist_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved histogram plot: {hist_path}")

        # --- 2) Corner plot overlay (joint distribution) ---
        try:
            fig = corner.corner(samples, color='tab:blue', bins=40, smooth=1.0, alpha=0.4,
                                labels=[r"$\Omega_m$", r"$h$"])
            corner.corner(samples, weights=weights, color='tab:orange', bins=40, smooth=1.0,
                          alpha=0.4, fig=fig)
            fig.suptitle("Joint Posterior: Before (blue) vs After (orange) Planck Prior")
            corner_path = os.path.join(fig_dir, "planck_prior_corner.png")
            fig.savefig(corner_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"✅ Saved corner plot: {corner_path}")
        except Exception as e:
            print(f"⚠️ Could not create corner plot: {e}")

    return {"before": summary_before, "after": summary_after}

chains_mcmc, acf_mcmc, pooled_mcmc = run_mcmc(n_steps=5000, m=4, burn_frac=0.2)
result = reweight_with_planck_prior(pooled_mcmc)
