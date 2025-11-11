import os
import numpy as np
import corner
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from scripts.summary import summarize_posterior
from scripts.run_mcmc import run_mcmc
from scripts.run_hmc import run_hmc
from src.data_reader import get_data
from scripts.plotting import plot_trace_all, plot_autocorrelation_compare, plot_corner_compare, plot_data_vs_model


# After running both
chains_mcmc, acf_mcmc, pooled_mcmc = run_mcmc(n_steps=5000, m=4, burn_frac=0.2)
chains_hmc, acf_hmc, pooled_hmc = run_hmc(n_steps=5000, m=4, burn_frac=0.2, epsilon=0.01, L=3)

# Generate summaries
summary_mcmc = summarize_posterior(pooled_mcmc, "MCMC")
summary_hmc = summarize_posterior(pooled_hmc, "HMC")

plot_trace_all(chains_mcmc, "MCMC")
plot_trace_all(chains_hmc, "HMC")
plot_autocorrelation_compare(acf_mcmc, acf_hmc)
plot_corner_compare(pooled_mcmc, pooled_hmc)

z_data, mu_data, Cinv = get_data()
plot_data_vs_model(z_data, mu_data, Cinv, pooled_mcmc, pooled_hmc)
