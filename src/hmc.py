# hamiltonina monte carlo
import numpy as np
from likelihood import log_posterior

# do finite and jax.grad

# grad --> finite differences of log_posterior:
def grad_log_posterior(theta, log_post_fn, h=1e-6):
    d = len(theta)
    grad = np.zeros(d)

    for i in range(d):
        theta_plus = theta.copy()
        theta_minus = theta.copy()
        theta_plus[i] += h
        theta_minus[i] -= h
        
        grad[i] = (log_post_fn(theta_plus) - log_post_fn(theta_minus)) / (2*h)

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


'''
phase 2: metropolis acceptance- reject or accept proposal
 compute hamiltonian change:
    delta_H = H(new_state) - H(state)
 accept with probability: /alpha = min(1, exp(-delta_H))
    if accepted: state = new_state
    if rejected: state = state (no change)
 look for acc_rate ~ [0.6 - 0.9]
'''


def hamiltonian_monte_carlo(log_prob_fn, grad_log_prob_fn, theta_init, epsilon, L, n_steps, args=()):
    theta = theta_init
    # M = identity matrix
    M = np.array([[1, 0], [0, 1]])
    p = np.random.normal(0, M)
    state = (theta, p)
    state_prop = hmc_proposal(theta)

    del_H = H(state_prop) - H()

'''
H(theta, p) = U(theta) + K(p)
U(theta) = -ln p(theta|D)
K(p) = 1/2 p.T * M.I * p
dtheta/dt = M.I * p
dp/dt = grad(ln p(theta|D))

complete hmc algorithm
phase 1: generate proposal via hamiltonian dynamics
 theta = theta_init
 p = np.random.norm(0, M), where M = Identity matrix
 state = (theta, p)
 L steps, \epsilon stepsize --> new_state
 negate momentum: p_new --> -p_new


phase 2: metropolis acceptance- reject or accept proposal
 compute hamiltonian change:
    delta_H = H(new_state) - H(state)
 accept with probability: /alpha = min(1, exp(-delta_H))
    if accepted: state = new_state
    if rejected: state = state (no change)
 look for acc_rate ~ [0.6 - 0.9]

 validation values:
 1. acc_rate ~ [0.6, 0.9]
 2. tau ~ [3, 10]
 3. ESS ~ N/10
'''