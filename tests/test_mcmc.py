import numpy as np
import pandas as pd
import os
from src.mcmc import metropolis_hastings
from src.likelihood import log_posterior
from src.data_reader import get_data
from src.diagnostics import autocorrelation, effective_sample_size, gelman_rubin
from tests.mcmc_tuning import sigma_tuning

# Get directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build a path to ../outputs/tables relative to the script
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "outputs", "tables")

# Make sure the directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

 
z_data, mu_data, Cinv = get_data()

theta_init = np.array([0.3, 0.7])

acc_rates, sigmas = sigma_tuning(z_data, mu_data, Cinv)
#sigmas = np.array([0.03305506, 0.01322202])


print(sigmas)

n_steps = 10000

chains = np.zeros((4, n_steps, 2))

summary_data = []

for k in range(4):
    # Randomly initialize Omega_m and h in [0, 1]
    omega_m0 = np.random.uniform(0, 1)
    h0 = np.random.uniform(0.4, 1)
    theta_init = np.array([omega_m0, h0])

    # Run Metropolis-Hastings sampler
    chain, acc_rate = metropolis_hastings(
        log_posterior,
        theta_init,
        sigmas,
        n_steps,
        args=(z_data, mu_data, Cinv)
    )

    # Diagnostics for each parameter
    ess_omegam, tau_omegam = effective_sample_size(chain[:, 0])
    ess_h, tau_h = effective_sample_size(chain[:, 1])

    summary_data.append({
        "Chain": k + 1,
        "Init_Ωm": round(omega_m0, 3),
        "Init_h": round(h0, 3),
        "ESS_Ωm": round(ess_omegam, 1),
        "τ_Ωm": round(tau_omegam, 2),
        "ESS_h": round(ess_h, 1),
        "τ_h": round(tau_h, 2),
        "Acceptance": round(acc_rate, 3)
    })

    chains[k, :, :] = chain

# Build per-chain summary table
summary_table = pd.DataFrame(summary_data)

# Compute R-hat across all chains
Rhat = gelman_rubin(chains)
rhat_omegam = round(float(Rhat[0]), 3)
rhat_h = round(float(Rhat[1]), 3)

# Add summary row for R-hat only
rhat_row = {
    "Chain": "R̂",
    "Init_Ωm": "",
    "Init_h": "",
    "ESS_Ωm": "",
    "τ_Ωm": "",
    "ESS_h": "",
    "τ_h": "",
    "Acceptance": "",
}


summary_table = pd.concat([summary_table, pd.DataFrame([rhat_row])], ignore_index=True)
summary_table.loc[summary_table["Chain"] == "R̂", ["ESS_Ωm", "ESS_h"]] = [rhat_omegam, rhat_h]

# Print results
print("\n=== MCMC Diagnostics Summary ===")
print(summary_table.to_string(index=False))

output_path = os.path.join(OUTPUT_DIR, "mcmc_diagnostics_summary.csv")
summary_table.to_csv(output_path, index=False)
print(f"\nSaved diagnostics table to: {output_path}")
