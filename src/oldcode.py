dat = data_points_reader('../data/jla_mub.txt')
z_data = dat['z']
mu_data = dat['mu']
C = covariance_reader('../data/jla_mug_covmatrix.txt')
Cinv = np.linalg.inv(C)

def log_likelihood(theta, z_data, mu_data, Cinv):
    # log likelihood for gaussian errors with covariance C eqn:
    #ln L(theta) = - 1/2 n-sum (1, j = 1) r_i [C-1]_ij r_j
    #r_i = mu_i^obs - mu_i^theory(z_i;theta)
    Omega_m, h = theta

    # theoretical mu
    mu_theory = np.array([[distance_modulus(z, Omega_m, h=h, Pen=True)] for z in z_data])
    
    # residuals
    r = mu_data - mu_theory

    loglikelihood = - 1/2 * np.dot(r, np.dot(Cinv, r))

    return loglikelihood


def log_prior(theta):
    # prior probabilities
    Omega_m, h = theta
    if 0.0 < Omega_m < 1.0 and 0.4 < h < 1.0:
        return 0.0 # log (1.0)
    else:
        return -np.inf

def log_posterior(theta, z_data, mu_data, Cinv):
    # way to sample the posterior

    lp = log_prior(theta)

    if not np.isfinite(lp):
        return -np.inf
    llh = log_likelihood(theta, z_data, mu_data, Cinv)

    return lp + llh




def proposal(theta, sigma):
    step = np.random.normal(0, sigma, size=len(theta))
    theta_new = theta + step

    log_Q_ratio = 0.0 # symmetric

    return theta_new, log_Q_ratio


while acc_rate < 0.2 or acc_rate > 0.5:
    chain, acc_rate = metropolis_hastings(log_posterior, theta_init, sigmas, n, args=(z_data, mu_data, Cinv))

    if acc_rate < 0.2:
        g = np.random.uniform(0.6, 0.9)
        sigmas = sigmas*(g**2)

    if acc_rate > 0.5:
        g = np.random.uniform(1.1, 1.5)
        sigmas = sigmas*(g**2)


def hmc_proposal(state, epsilon):
    M = np.array([[1, 0], [0, 1]])
    #p = np.random.normal(0, M)
    #state = (theta, p)

    theta = state[0]
    p = state[1]    

    half_p = p + (epsilon/2) * grad_log_posterior(theta, log_post_fn, h=1e-6)

    full_theta = theta + epsilon * half_p

    full_p = half_p + (epsilon/2) * grad_log_posterior(theta, log_post_fn, h=1e-6)

    return (full_theta, - full_p)


def hamiltonian_monte_carlo(log_prob_fn, grad_log_prob_fn, theta_init, epsilon, L, n_steps, args=()):
    theta = theta_init
    # M = identity matrix
    M = np.array([[1, 0], [0, 1]])
    p = np.random.normal(0, M)
    state = (theta, p)
    state_prop = leapfrog(theta)

    del_H = H(state_prop) - H()



    def hamiltonian(theta, p, log_prob_fn):
    U = - log_prob_fn(theta)
    K = 0.5 * np.dot(p, p)

    return U + K

def hmc(log_prob_fn, grad_log_prob_fn, theta_init, epsilon, L, n_steps):
    theta = theta_init
    chain = []
    accepts = 0

    for step in range(n_steps):
        p = np.random.normal(0, 1, size=theta.shape)

        H_current = hamiltonian(theta, p, log_prob_fn)

        theta_prop, p_prop = leapfrog(theta, p, epsilon, L, log_prob_fn, grad_log_posterior)

        H_prop = hamiltonian(theta_prop, p_prop, log_prob_fn)

        delta_H = H_prop - H_current

        if np.random.rand() < np.exp(-delta_H):
            theta = theta_prop
            accepts += 1
        
        chain.append(theta.copy())

    accept_rate = accepts / n_steps
    
    return np.array(chain), accept_rate
        
def autocorrelation(x):
    N = x.size
    centered = x - x.mean()
    maxlag = N // 2

    nfft = 1 << (2 * N -1).bit_length()
    fx = np.fft.rfft(centered, n=nfft)
    acov = np.fft.irfft(fx * np.conjugate(fx), n=nfft)[:N]
    acov /= N
    var0 = acov[0]

    lags = np.arange(maxlag)
    rho = acov[:maxlag] / var0

    return lags, rho


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

def distance_modulus(z, Omega_m, h = 0.7, Pen=False):
    #distance modulus eqn: mu = 25 - 5 log_10 (h) + 5 log_10 (D*_L / Mpc)
    I = intd(z, Omega_m, Pen=Pen)
    DL_star = luminosity_distance(z, I, h)
    mu = 25 + 5 * np.log10(DL_star) 
    
    return mu