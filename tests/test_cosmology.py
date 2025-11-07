import numpy as np
import timeit

from src.cosmology import distance_modulus

z = 0.5
Omega_m = 0.3
h = 0.7
n = 100

def dist_mod(Pen):
    mu = distance_modulus(z, Omega_m, h, Pen=Pen)
    return mu

# calculate mu for both int and approx, compare to 42.26 mag and check that mu_approx is within 0.4% of mu_int
mu_int = dist_mod(Pen=False)
pen_int = dist_mod(Pen=True)

target_mu = 42.26 # mag

if (target_mu - 0.001*target_mu) < mu_int < (target_mu+0.001*target_mu):
    print(f'Integrated distance modulus: {mu_int:.5} mag. Integration is correct.')
else:
    print(f'Integrated distance modulus: {mu_int:.5} mag. Integration is incorrect.')    
if (target_mu - 0.001*target_mu) < pen_int < (target_mu+0.001*target_mu):
    print(f'Pen approximated distance modulus: {pen_int:.5} mag. Approximation is correct.')
else:
    print(f'Pen approximated distance modulus: {pen_int:.5} mag. Approximation is incorrect.')

if (mu_int - 0.004*mu_int) < pen_int < (mu_int + 0.004*mu_int):
    print('Pen approximates integrated distance modulus to within 0.4 percent accuracy.')
else:
    print('Pen does not approximate integrated distance modulus correctly.')

# compare times
int_time = timeit.timeit(lambda: dist_mod(Pen=False), number=n)

pen_time = int = timeit.timeit(lambda: dist_mod(Pen=True), number=n)

print(f"Integration time for {n} interation(s): {int_time:.5} s")
print(f"Pen approximation time for {n} interation(s): {pen_time:.5} s")

if int_time > pen_time:
    print(f"Pen approximation is {int_time/pen_time:.5} faster than integrating.")
else:
    print(f"Integrating is {pen_time/int_time:.5} faster than Pen approximation.")