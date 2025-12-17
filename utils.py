import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
from scipy.optimize import bisect
from scipy.optimize import root_scalar


def curvature_function(lambdah, s, Vt, beta):
    """
    Compute curvature function using precomputed SVD.
    A = U diag(s) Vt
    beta = U.T @ b
    """
    if lambdah == 0:
        return 0
    
    # filter factors
    denom = s**2 + lambdah**2 
    f = s**2 / denom

    # solution y(λ)
    y = Vt.T @ (f *beta/s)
    epsilon = np.linalg.norm(y)**2

    # residual r(λ)
    rho = np.sum(((lambdah**2 / denom)**2) * (beta**2))
    # derivative y'(λ)

    fprime = (-2*lambdah*s) / denom**2
    yprime = (Vt.T * fprime) @ beta
    epsilon_prime = 2 * np.dot(y, yprime)

    curve = ((2 * epsilon * rho) *
             ((lambdah**2) * epsilon_prime * rho +
              2 * lambdah * epsilon * rho +
              (lambdah**4) * epsilon * epsilon_prime)) / \
            (epsilon_prime * ((lambdah**2 * epsilon**2 + rho**2)**(3/2)))

    return -curve

def lcurve(s, Vt, beta, tol=1e-5, max_iter=100):
    """
    Finds corner of L shaped curve formed by the loglog plot of ||Ax-b|| and ||x|| for the tikhonov regularization parameters using precomputed SVD of A.
    A = U diag(s) Vt
    """


    phi = (1 + np.sqrt(5)) / 2  
    invphi = 1 / phi
    invphi2 = invphi**2
    log_lams = np.linspace(-8, 2, 100)
    curves = []
    for loglam in log_lams:
        lam = 10**loglam
        cval = curvature_function(lam, s, Vt, beta)
        curves.append(cval)

    curves = np.array(curves)
    if np.all(curves <= 0):
        return 0
    best_idx = np.argmax(curves)
    best_loglam = log_lams[best_idx]

    window = 1.0  
    log_low = max(log_lams[0], best_loglam - window)
    log_high = min(log_lams[-1], best_loglam + window)
    log_2 = log_high - invphi2 * (log_high - log_low)
    log_1 = log_low + invphi2 * (log_high - log_low)

    curve1 = curvature_function(10**log_1, s, Vt, beta)
    curve2 = curvature_function(10**log_2, s, Vt, beta)

    iter_count = 0
    while np.abs(log_high - log_low) > tol and iter_count < max_iter:
        iter_count += 1

        if curve1 > curve2:
            log_high = log_2
            log_2 = log_1
            curve2 = curve1
            log_1 = log_low + invphi2 * (log_high - log_low)
            curve1 = curvature_function(10**log_1, s, Vt, beta)   
        else:
            log_low = log_1
            log_1 = log_2
            curve1 = curve2
            log_2 = log_high - invphi2 * (log_high - log_low)
            curve2 = curvature_function(10**log_2, s, Vt, beta)

    optimal_lambda = 10**((log_low + log_high) / 2)
    return optimal_lambda


def NCP(r, m, num_angles):
    ''' 
    Stopping criteria: Normalized Cumulative Periodogram

    INPUTS
    r:      Residual vector for i'th iteration.
    m:      Number of pixels in the sinogram.
    num_angles:  The number of view angles.
    '''
    
    nt = int(num_angles)
    nnp = int(m / nt)
    q = int(np.floor(nnp/2))
    c_white = np.linspace(1,q,q)/q
    C = np.zeros((q,nt))
    
    R = r.reshape(nnp,nt)
    for j in range(0,nt):
        RKH = np.fft.fft(R[:,j])
        pk = abs(RKH[0:q+1])
        c = np.cumsum(pk[1:])/np.sum(pk[1:])
        C[:,j] = c

    Nk = np.linalg.norm(np.mean(C,1)-c_white)
    return Nk
