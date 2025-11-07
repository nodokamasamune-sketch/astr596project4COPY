# diagnostics
'''
validation tests:
1. autocorrelation- measures correlation between samples separated by lag kappa
2. Effective Sample Size (ESS)
3. Gelman Rubin statistic
'''
import numpy as np

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






def effective_sample_size(x):
    N = x.size
    lags, rho = autocorrelation(x)

    last_pos = 0
    negligible = 0.05
    for i in range(1, len(rho)):
        if abs(rho[i]) < negligible:
            last_pos = i
            break

    tau = 1.0 + 2.0 * np.sum(rho[1:last_pos+1])

    ess = N / tau

    return ess, tau



def gelman_rubin(chains):
    '''
    parameters:
    chains : array-like
        ndarray with shape (m, n, p) with m chains, n samples each, p parameters
    '''
    data = np.asarray(chains)
    p = data.shape[2]
    M, N, _ = data.shape
    Rhat = np.zeros(p)

    for i in range(p):
        chain_param = data[:, :, i]
        means = np.mean(chain_param, axis=1)
        vars_inner = np.var(chain_param, axis=1, ddof=1)

        W = np.mean(vars_inner)
        B = N * np.var(means, ddof=1)

        var_hat = ( (N -1) / N ) * W +(1.0 / N) * B

        Rhat[i] = np.sqrt(var_hat / W)

    return Rhat

 