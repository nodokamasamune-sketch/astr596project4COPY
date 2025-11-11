import numpy as np
import os
from src.mcmc import metropolis_hastings
from src.likelihood import log_posterior
from src.data_reader import get_data
from src.diagnostics import autocorrelation, effective_sample_size, gelman_rubin
from tests.mcmc_tuning import sigma_tuning

def _make_output_dir(subdir="data"):
    base = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(base, "..", "outputs", subdir)
    os.makedirs(outdir, exist_ok=True)
    return outdir

def run_mcmc(n_steps=10000, m=4, burn_frac=0.2):
    """
    Run multiple MCMC chains, discard burn-in, compute Rhat and ESS,
    and return outputs compatible with plotting utilities.

    Returns:
        postburn_chains : ndarray (m, n_steps_after_burn, 2)
        acf_data : list of tuples [(lags_Ωm, rhos_Ωm), (lags_h, rhos_h)] per chain
        pooled_chain : ndarray ((m * n_steps_after_burn), 2)
    """
    z_data, mu_data, Cinv = get_data()
    _, sigmas = sigma_tuning(z_data, mu_data, Cinv)

    chains = np.zeros((m, n_steps, 2))
    acc_rates = []

    print(f"Running {m} MCMC chains for {n_steps} steps each...")

    # Run all chains
    for k in range(m):
        omega_m0 = np.random.uniform(0.0, 1.0)
        h0 = np.random.uniform(0.4, 1.0)
        theta_init = np.array([omega_m0, h0])

        chain, acc_rate = metropolis_hastings(
            log_posterior,
            theta_init,
            sigmas,
            n_steps,
            args=(z_data, mu_data, Cinv)
        )

        chains[k, :, :] = chain
        acc_rates.append(acc_rate)
        print(f"  Chain {k+1}: acceptance rate = {acc_rate:.3f}")

    # Burn-in removal
    burn_idx = int(burn_frac * n_steps)
    postburn_chains = chains[:, burn_idx:, :]

    # Convergence check
    Rhat = gelman_rubin(postburn_chains)
    print(f"\nGelman–Rubin R̂: Ωm={Rhat[0]:.3f}, h={Rhat[1]:.3f}")
    if np.any(Rhat > 1.1):
        print("⚠️ Warning: chains may not have converged (R̂ > 1.1).")

    # Effective sample size
    ess_omegam, _ = effective_sample_size(postburn_chains.reshape(-1, 2)[:, 0])
    ess_h, _ = effective_sample_size(postburn_chains.reshape(-1, 2)[:, 1])
    print(f"ESS: Ωm={ess_omegam:.1f}, h={ess_h:.1f}")

    # Compute autocorrelation data
    acf_data = [[], []]  # [Omega_m autocorrs, h autocorrs]
    for k in range(m):
        lags_omegam, rho_omegam = autocorrelation(postburn_chains[k, :, 0])
        lags_h, rho_h = autocorrelation(postburn_chains[k, :, 1])
        acf_data[0].append((lags_omegam, rho_omegam))
        acf_data[1].append((lags_h, rho_h))

    # Pool chains
    pooled_chain = postburn_chains.reshape(-1, 2)

    # Save results
    outdir = _make_output_dir("data")
    np.save(os.path.join(outdir, "mcmc_chains.npy"), chains)
    np.save(os.path.join(outdir, "mcmc_postburn.npy"), postburn_chains)
    np.save(os.path.join(outdir, "mcmc_pooled.npy"), pooled_chain)
    print(f"✅ Saved MCMC outputs to {outdir}")

    return postburn_chains, acf_data, pooled_chain
