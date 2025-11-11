"""
USED ChatGPT to write doc strings
"""

# hamiltonina monte carlo
import numpy as np


# grad --> finite differences of log_posterior:
def grad_log_posterior(theta, log_post_fn, h=1e-6, args=()):
    d = len(theta)
    grad = np.zeros(d)

    current_logprob = log_post_fn(theta, *args)
    if not np.isfinite(current_logprob):
        return np.zeros(d)
    
    for i in range(d):
        theta_plus = theta.copy()
        theta_minus = theta.copy()
        theta_plus[i] += h
        theta_minus[i] -= h
        
        f_plus = log_post_fn(theta_plus, *args)
        f_minus = log_post_fn(theta_minus, *args)

        if not np.isfinite(f_plus) or not np.isfinite(f_minus):
            grad[i] = 0.0
        else:
            grad[i] = (f_plus - f_minus) / (2.0 * h)

    return grad


'''
complete hmc algorithm
phase 1: generate proposal via hamiltonian dynamics
 theta = theta_init
 p = np.random.norm(0, M), where M = Identity matrix
 state = (theta, p)
 L steps, \epsilon stepsize --> new_state
 negate momentum: p_new --> -p_new
'''


def leapfrog(theta, p, log_post_fn, grad_fn, epsilon, L, args=()):
    grad = grad_fn(theta, log_post_fn, args=args)
    p = p + 0.5 * epsilon * grad
    
    for i in range(L):
        theta = theta + epsilon * p
        
        if i != L - 1:
            grad = grad_fn(theta, log_post_fn, args=args)
            p = p + epsilon * grad

        
    grad = grad_fn(theta, log_post_fn, args=args)
    p = p + 0.5 * epsilon * grad

    return theta, -p



'''
phase 2: metropolis acceptance- reject or accept proposal
 compute hamiltonian change:
    delta_H = H(new_state) - H(state)
 accept with probability: /alpha = min(1, exp(-delta_H))
    if accepted: state = new_state
    if rejected: state = state (no change)
 look for acc_rate ~ [0.6 - 0.9]
'''


def hamiltonian(theta, p, log_post_fn, args=()):
    """
    Compute Hamiltonian H = U + K
    where U = -log_posterior(theta), K = 0.5 * p^T p
    """
    U = -log_post_fn(theta, *args)
    K = 0.5 * np.dot(p, p)
    return U + K


def hamiltonian_monte_carlo(log_post_fn, theta_init, n_steps, epsilon=0.01, L=10, args=()):
    """
    Hamiltonian Monte Carlo sampler (2 parameters, NumPy version).
    
    Parameters
    ----------
    log_post_fn : callable
        Function returning log posterior (e.g., log_posterior from likelihood.py).
    theta_init : array-like, shape (2,)
        Starting parameter values.
    epsilon : float
        Step size for leapfrog integration.
    L : int
        Number of leapfrog steps per iteration.
    n_steps : int
        Number of HMC iterations.

    Returns
    -------
    chain : ndarray, shape (n_steps, 2)
        Sequence of sampled parameter values.
    accept_rate : float
        Fraction of accepted proposals.
    """
    theta = np.array(theta_init, dtype=float)
    chain = []
    delta_Hs = []
    accepts = 0

    for step in range(int(n_steps)):
        # 1️⃣ Sample fresh momentum p ~ N(0, I)
        p = np.random.normal(0, 1, size=theta.shape)

        # 2️⃣ Compute current Hamiltonian
        H_current = hamiltonian(theta, p, log_post_fn, args=args)

        # 3️⃣ Generate proposal via leapfrog integration
        theta_prop, p_prop = leapfrog(theta, p, log_post_fn, grad_log_posterior, epsilon, L, args=args)

        # 4️⃣ Compute proposed Hamiltonian
        H_proposed = hamiltonian(theta_prop, p_prop, log_post_fn, args=args)

        # 5️⃣ Metropolis acceptance
        delta_H = H_proposed - H_current
        delta_Hs.append(delta_H)
        if np.log(np.random.rand()) < -delta_H:
            theta = theta_prop
            accepts += 1

        # 6️⃣ Store current sample
        chain.append(theta.copy())

    accept_rate = accepts / n_steps
    return np.array(chain), np.array(delta_Hs), accept_rate


