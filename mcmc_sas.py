# Sampling method for the MCMC-spike_and_slab algorithm of Polson et al. (2011)

from .utils import *
import numpy as np

def sample(X, sigma, nu=100, pi=None, T=100):
    
    n, k = X.shape
    
    sigma2 = np.diag(sigma**2)
    sigma2_inv = np.diag(1/sigma**2)
    
    if pi is None:
        pi = 1/2*np.ones(k)

    betas = np.zeros((T,k))
    lambdas = np.zeros((T,n))
    gammas = np.ones((T,k))

    betas[0] = rng.normal(0,1,size=k)
    lambdas[0] = rng.normal(0,1,size=n)
    gammas[0] = np.ones(k).astype(int)

    lambdas_idx = list(range(n))
    
    # We need to initialize B and b

    for t in range(1,T):
        
        if t % 100 == 0:
            print("Step {}".format(t))
        
        # Do we need to keep that part ? I would say yes just to get rid of the support vectors
        # num_lambda = numerical(1/lambdas[t-1])
        # if num_lambda[0]:
        #     print("At time", t)
        #     idx, tag = num_lambda[1], num_lambda[2]
        #     print("Found a numerical issue for lambda, in position {}, with value {}".format(i,tag))
        #     print("Support vector found at index {}".format(i))
        #     lambdas[:,idx] = np.nan
        #     for i in idx:
        #         if i in lambdas_idx:
        #             lambdas_idx.remove(i)
        #     print('Remaining coeffs in lambda : {}'.format(len(lambdas_idx)))
            
        # Sampling lambda using beta
        for i in lambdas_idx:
            lambda_inv = rng.wald(1/np.abs(1-X[i][gamma[t-1]].T @ betas[t][gamma[t-1]]), 1)
            if lambda_inv == 0.0:
                lambdas[t,i] = np.inf
            else:
                lambdas[t,i] = 1/lambda_inv
                                                  
        # Sampling gamma using previous state
        # How to do that ? Compute for each term, normalize and then choose between 0 and 1 for each gamma_i ?
        # Where do the values of gamma_i and gamma_(i-1) intervene in the expression ?

        # Computing B and b
        term1 = nu**(-2)* sigma2_inv[omegas_idx,:][:,omegas_idx] @ np.diag(1/omegas[t-1][omegas_idx])
        term2 = X[lambdas_idx,:][:,omegas_idx].T @ np.diag(1/lambdas[t-1][lambdas_idx]) @ X[lambdas_idx,:][:,omegas_idx]
        B = term1 + term2
        b = B @ X[lambdas_idx,:][:,omegas_idx].T @ (np.ones(len(lambdas_idx)) + 1/lambdas[t-1][lambdas_idx])

        # Sampling beta using previous state
        betas[t][omegas_idx] = rng.multivariate_normal(b, cov=B)
    
    return betas