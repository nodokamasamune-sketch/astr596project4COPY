import numpy as np
import os
from src.hmc import hamiltonian_monte_carlo
from src.likelihood import log_posterior
from src.data_reader import get_data
from src.diagnostics import autocorrelation, effective_sample_size, gelman_rubin

def _make_output_dir(subdir="data"):
    base = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(base, "..", "outputs", subdir)
    os.makedirs(outdir, exist_ok=True)
    return outdir

def run_hmc(n_steps=10000, m=4, burn_frac=0.2, epsilon=0.005, L=9):
    """
    Run multiple HMC chains, discard burn-in, compute Rhat and ESS,
    and return outputs compatible with plotting utilities.

    Returns:
        postburn_chains : ndarray (m, n_steps_after_burn, 2)
        acf_data : list of tuples [(lags_Ωm, rhos_Ωm), (lags_h, rhos_h)] per chain
        pooled_chain : ndarray ((m * n_steps_after_burn), 2)
    """
    z_data, mu_data, Cinv = get_data()
    chains = np.zeros((m, n_steps, 2))
    delta_Hs, acc_rates = [], []

    print(f"Running {m} HMC chains for {n_steps} steps each (ε={epsilon}, L={L})...")

    for k in range(m):
        omega_m0 = np.random.uniform(0.0, 1.0)
        h0 = np.random.uniform(0.4, 1.0)
        theta_init = np.array([omega_m0, h0])

        chain, deltas, acc_rate = hamiltonian_monte_carlo(
            log_post_fn=log_posterior,
            theta_init=theta_init,
            epsilon=epsilon,
            L=L,
            n_steps=n_steps,
            args=(z_data, mu_data, Cinv)
        )
        chains[k, :, :] = chain
        delta_Hs.append(deltas)
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

    # Autocorrelation
    acf_data = [[], []]
    for k in range(m):
        lags_omegam, rho_omegam = autocorrelation(postburn_chains[k, :, 0])
        lags_h, rho_h = autocorrelation(postburn_chains[k, :, 1])
        acf_data[0].append((lags_omegam, rho_omegam))
        acf_data[1].append((lags_h, rho_h))

    pooled_chain = postburn_chains.reshape(-1, 2)

    # Save results
    outdir = _make_output_dir("data")
    np.save(os.path.join(outdir, "hmc_chains.npy"), chains)
    np.save(os.path.join(outdir, "hmc_postburn.npy"), postburn_chains)
    np.save(os.path.join(outdir, "hmc_pooled.npy"), pooled_chain)
    np.save(os.path.join(outdir, "hmc_deltaH.npy"), delta_Hs)
    print(f"✅ Saved HMC outputs to {outdir}")

    return postburn_chains, acf_data, pooled_chain
