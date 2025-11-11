"""
USED ChatGPT TO WRITE DOC STRING 

Compute the theoretical distance modulus μ(z) for a flat ΛCDM universe.

This module provides functions to calculate the distance modulus as a function
of redshift z and matter density parameter Ω_m, assuming spatial flatness 
(Ω_Λ = 1 - Ω_m). The distance modulus is derived from the luminosity distance,
which depends on the cosmological expansion rate.

Functions:
    distance_modulus(z, Omega_m, h=0.7, Pen=False)
        Computes μ(z) = 25 + 5 log10(D_L / Mpc) using either direct integration 
        or Pen’s analytic approximation.
    luminosity_distance(z, I, h=0.7)
        Converts a dimensionless comoving distance integral I(z) into a 
        luminosity distance D_L(z) in Mpc.
    eta(a, Omega_m)
        Auxiliary function used in Pen’s (1999) analytic approximation for 
        comoving distance.
    intd(z, Omega_m, Pen=False)
        Computes the dimensionless comoving distance integral from 0 to z.

Assumptions:
    - Flat universe (Ω_k = 0)
    - H_0 = 100 * h km/s/Mpc
    - c = 3 × 10⁵ km/s
    - Optional use of Pen (1999) analytic approximation for speed.

Returns:
    μ(z): distance modulus in magnitudes.
"""

#compute theoretical distance modulus assuming a flat universe
import numpy as np
from scipy.integrate import quad


def distance_modulus(z, Omega_m, h=0.7, Pen=False):
    """
    Compute theoretical distance modulus μ(z) for a flat universe.
    """
    if z == 0:
        return 0.0  # D_L(0)=0 → μ(0)=0

    I = intd(z, Omega_m, Pen=Pen)
    DL_star = luminosity_distance(z, I, h)

    if DL_star <= 0:
        raise ValueError(f"Luminosity distance nonpositive: D_L = {DL_star}")

    mu = 25 + 5 * np.log10(DL_star)
    return mu


def luminosity_distance(z, I, h=0.7):
    #for flat univere (Omega_lambda = 1 - Omega_m), the luminosity distance eqn: D_L(z) = [c (1 + z)] / H_0 * integral(dz' / E(z')) from 0 to z
    #E(z) = sqrt(Omega_m * (1 + z)^3 + (1 - Omega_m))
    
    H_0 = 100 * h #km/s/Mpc
    c = 3e5 #km/s
    D_L = (c / H_0) * (1 + z) * I
    return D_L


def eta(a, Omega_m):
    s = ((1 - Omega_m) / Omega_m) ** (1/3)
    eta = 2 * np.sqrt(s**3 + 1) * ((1/a**4 - 0.154 * s/a**3) + (0.4304 * s**2/a**2) + (0.19097*s**3/a) + (0.066941*s**4))**(-1/8)
    return eta


def intd(z, Omega_m, Pen=False):
    if Pen==False:
        inte = lambda z, Omega_m: 1/np.sqrt(Omega_m * (1 + z)**3 + (1 - Omega_m))
        I, error = quad(inte, 0, z, args=(Omega_m,))

    else:
        I = (eta(1, Omega_m) - eta(1/(1 + z), Omega_m))
    
    return I

