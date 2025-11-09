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


def leapfrog(theta, p, epsilon, L, log_post_fn, grad_fn):
    grad = grad_fn(theta, log_post_fn)
    p = p + 0.5 * epsilon * grad
    
    for i in range(L):
        theta = theta + epsilon * p
        
        if i != L - 1:
            grad = grad_fn(theta, log_post_fn)
            p = p + epsilon * grad

        
    grad = grad_fn(theta, log_post_fn)
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