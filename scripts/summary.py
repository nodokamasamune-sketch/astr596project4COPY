import numpy as np
import pandas as pd
import os

def summarize_posterior(pooled_chain, sampler_name="MCMC"):
    """
    Compute mean, std, credible intervals, and correlation between parameters.
    Saves summary table as CSV in ../outputs/tables.
    
    Parameters
    ----------
    pooled_chain : ndarray of shape (N_samples, 2)
        Flattened post-burn-in posterior samples.
    sampler_name : str
        "MCMC" or "HMC"
    
    Returns
    -------
    summary_table : pd.DataFrame
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(base_dir, "..", "outputs", "tables")
    os.makedirs(outdir, exist_ok=True)

    param_names = ["Omega_m", "h"]

    means = np.mean(pooled_chain, axis=0)
    stds = np.std(pooled_chain, axis=0)
    p16, p50, p84 = np.percentile(pooled_chain, [16, 50, 84], axis=0)
    corr = np.corrcoef(pooled_chain.T)[0, 1]

    summary_data = {
        "Parameter": param_names,
        "Mean": np.round(means, 4),
        "StdDev": np.round(stds, 4),
        "16th": np.round(p16, 4),
        "50th": np.round(p50, 4),
        "84th": np.round(p84, 4)
    }

    summary_table = pd.DataFrame(summary_data)
    summary_table["Correlation(Omega_m,h)"] = ["", f"{corr:.3f}"]

    fname = f"{sampler_name.lower()}_posterior_summary.csv"
    fpath = os.path.join(outdir, fname)
    summary_table.to_csv(fpath, index=False)

    print(f"\n✅ Saved posterior summary for {sampler_name} to: {fpath}")
    print(summary_table.to_string(index=False))
    return summary_table
