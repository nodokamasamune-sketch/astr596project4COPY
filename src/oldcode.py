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