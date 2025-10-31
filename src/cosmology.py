#compute theoretical distance modulus assuming a flat universe
import numpy as np
from scipy.integrate import quad

def luminosity_distance(z, Omega_m, h, Pen=False):
    #for flat univere (Omega_lambda = 1 - Omega_m), the luminosity distance eqn: D_L(z) = [c (1 + z)] / H_0 * integral(dz' / E(z')) from 0 to z
    #E(z) = sqrt(Omega_m * (1 + z)^3 + (1 - Omega_m))
    H_0 = 100 #km/s/Mpc
    c = 3e5 #km/s

    if Pen==False: #numerical integration using scipy.integrate.quad
        def lum_d_eqn(z):
            return 1 / np.sqrt (Omega_m * (1 + z)**3 + (1 - Omega_m))
        int, error = quad(lum_d_eqn, 0, z)
        D_L = c * (1 + z) / H_0 * int

    if Pen==True: #Pen approximation
        def n(a, Omega_m):
            s = ((1 - Omega_m) / Omega_m) ** (1/3)
            #n = 2 * np.sqrt(s**3 + 1) * (1/a**4 - 0.154 * s/a**3 + 0.4304 * s**2/a**2 + 0.19097*s**3/a + 0.066941*s**4)
            n = 1
            return n
        
        D_L = c / H_0 * (1 + z) * ( n(1, Omega_m) - n((1/(1 + z)), Omega_m) )

    return D_L

def distance_modulus(z, Omega_m, h = 0.7, Pen=False):
    #distance modulus eqn: mu = 25 - 5 log_10 (h) + 5 log_10 (D*_L / Mpc)
    DL_star = luminosity_distance(z, Omega_m, Pen)
    mu = 25 - 5 * np.log10(h) + 5 * np.log10(DL_star) 
    
    return mu

'''
ask about Mpc --> unit conversion?
ask about h value --. 1 or 0.7?
'''