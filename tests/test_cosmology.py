import os
import numpy as np
import timeit
import pandas as pd
from src.cosmology import distance_modulus

# Get directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build a path to ../outputs/tables relative to the script
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "outputs", "tables")

# Make sure the directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

Omega_m, h = 0.3, 0.7
target_mu = 42.26
z_grid = np.arange(0.1, 1.31, 0.1)
n = 100 # timing purposes

# Results storage
summary = {
    "Boundary (D_L(0)=0)": None,
    "μ_int(0.5)": None,
    "μ_pen(0.5)": None,
    "Δμ target (mag)": None,
    "Max Δμ(%) over z∈[0,1.3]": None,
    "Pen faster (×)": None
}

dl0 = distance_modulus(0.0, Omega_m, h, Pen=False)
summary["Boundary (D_L(0)=0)"] = 'Zero' if np.isclose(dl0, 0, atol=1e-6) else 'Not zero'

mu_int = distance_modulus(0.5, Omega_m, h, Pen=False)
mu_pen = distance_modulus(0.5, Omega_m, h, Pen=True)
summary["μ_int(0.5)"] = f"{mu_int:.3f}"
summary["μ_pen(0.5)"] = f"{mu_pen:.3f}"
summary["Δμ target (mag)"] = f"{abs(mu_int - target_mu):.3f}"

mu_int_grid = np.array([distance_modulus(z, Omega_m, h, Pen=False) for z in z_grid])
mu_pen_grid = np.array([distance_modulus(z, Omega_m, h, Pen=True) for z in z_grid])
rel_diff = np.abs(mu_pen_grid - mu_int_grid) / mu_int_grid * 100
summary["Max Δμ(%) over z∈[0,1.3]"] = f"{rel_diff.max():.3f}"

int_time = timeit.timeit(lambda: distance_modulus(0.5, Omega_m, h, Pen=False), number=n)
pen_time = timeit.timeit(lambda: distance_modulus(0.5, Omega_m, h, Pen=True), number=n)
speedup = int_time / pen_time if pen_time > 0 else np.inf
summary["Pen faster (×)"] = f"{speedup:.2f}"

validation_table = pd.DataFrame([summary])

print("\n=== Cosmology Validation Summary ===")
print(validation_table.to_string(index=False))

output_path = os.path.join(OUTPUT_DIR, "cosmology_diagnostics_summary.csv")
validation_table.to_csv(output_path, index=False)
print(f"\nSaved diagnostics table to: {output_path}")