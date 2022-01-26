# Sampling method for the MCMC alpha=1 algorithm of Polson et al. (2011)

from .utils import *
import numpy as np

def sample(X, sigma, rng, nu=1, T=100):
    
    # Extracting useful information
    n, k = X.shape
    
    # Computing the inverse Sigma matrix
    sigma2_inv = np.diag(1/sigma**2)
    
    # Initializing the arrays that will store the samples
    betas = np.zeros((T,k))
    lambdas = np.zeros((T,n))
    omegas = np.zeros((T,k))
    
    # Initializing the variables
    betas[0] = rng.normal(0,1,size=k)
    lambdas[0] = rng.wald(1,1,size=n)
    omegas[0] = rng.wald(1,1,size=k)
    
    # Lists to keep track of indices to remove for stability
    lambdas_idx = list(range(n))
    omegas_idx = list(range(k))

    for t in range(1,T):
        
        if t % 100 == 0:
            print("Step {}".format(t))
        
        # Removing omegas that are infinite to avoid errors
        num_omega = numerical(1/omegas[t-1])
        if num_omega[0]:
            print("At time", t)
            idx, tags = num_omega[1], num_omega[2]
            print("Found a numerical issue for omega, in position {}, with value {}".format(i,tag))
            print("Sparsity condition reached for beta[{}]".format(i))
            betas[:,idx] = 0
            for i in idx:
                if i in omegas_idx:
                    omegas_idx.remove(i)
            print('Remaining coeffs in omega : {}'.format(len(omegas_idx)))
        
        # Removing lambdas that are infinite to avoid errors
        num_lambda = numerical(1/lambdas[t-1])
        if num_lambda[0]:
            print("At time", t)
            idx, tag = num_lambda[1], num_lambda[2]
            print("Found a numerical issue for lambda, in position {}, with value {}".format(i,tag))
            print("Support vector found at index {}".format(i))
            lambdas[:,idx] = np.nan
            for i in idx:
                if i in lambdas_idx:
                    lambdas_idx.remove(i)
            print('Remaining coeffs in lambda : {}'.format(len(lambdas_idx)))

        # Computing B and b
        term1 = nu**(-2)* sigma2_inv[omegas_idx,:][:,omegas_idx] @ np.diag(1/omegas[t-1][omegas_idx])
        term2 = X[lambdas_idx,:][:,omegas_idx].T @ np.diag(1/lambdas[t-1][lambdas_idx]) @ X[lambdas_idx,:][:,omegas_idx]
        B_inv = term1 + term2
        B = np.linalg.inv(B_inv)
        b = B @ X[lambdas_idx,:][:,omegas_idx].T @ (np.ones(len(lambdas_idx)) + 1/lambdas[t-1][lambdas_idx])

        # Sampling beta using previous state
        betas[t][omegas_idx] = rng.multivariate_normal(b, cov=B)

        # Sampling lambda using beta
        for i in lambdas_idx:
            lambda_inv = rng.wald(1/np.abs(1-X[i][omegas_idx].T @ betas[t][omegas_idx]), 1)
            if lambda_inv == 0.0:
                lambdas[t,i] = np.inf
            else:
                lambdas[t,i] = 1/lambda_inv

        # Sampling omega using beta
        for j in omegas_idx:
            omega_inv = rng.wald(nu*sigma[j]/np.abs(betas[t,j]), 1)
            if omega_inv == 0.0:
                omegas[t,j] = np.inf
            else:
                omegas[t,j] = 1/omega_inv
    
    print("End of sampling: {} lambda coeffs removed".format(n-len(lambdas_idx)))
    return betas