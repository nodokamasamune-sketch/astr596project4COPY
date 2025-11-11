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

'''
ask about Mpc --> unit conversion?
how much of a difference in mu should we see between integration and approximation ? 0.4%
put definitions + eqns outside
ask how to use calculator
'''