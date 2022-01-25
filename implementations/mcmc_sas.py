# Sampling method for the MCMC spike-and-slab algorithm of Polson et al. (2011)

from .utils import *
import numpy as np

def sample(X, sigma, rng, nu=100, pi=None, T=100):
    
    n, k = X.shape
    
    values = [0,1]
    
    sigma2 = np.diag(sigma**2)
    sigma2_inv = np.diag(1/sigma**2)
    
    if pi is None:
        pi = 1/2*np.ones(k)

    betas = np.zeros((T,k))
    lambdas = np.zeros((T,n))
    gammas = np.zeros((T,k))

    betas[0] = rng.normal(0,1,size=k)
    lambdas[0] = rng.normal(0,1,size=n)
    gammas[0] = rng.binomial(n=1, p=pi)

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
        gamma_idx = gammas[t-1].astype(bool)
        for i in lambdas_idx:
            lambda_inv = rng.wald(1/np.abs(1-X[i][gamma_idx].T @ betas[t][gamma_idx]), 1)
            if lambda_inv == 0.0:
                lambdas[t,i] = np.inf
            else:
                lambdas[t,i] = 1/lambda_inv
                                                  
        # Sampling gamma using newest available values
        current_gamma = np.copy(gammas[t-1])
        for j in range(k):
            p = np.zeros(2)
            for value in values:
                current_gamma[j] = value
                gamma_idx = current_gamma.astype(bool) 
                
                term1 = np.array([pi[i]**current_gamma[i] * (1-pi[i])**(1-current_gamma[i]) for i in range(k)]).prod()
                term2 = np.sqrt( (1/nu**2) * (1/sigma[gamma_idx]**2).prod() / np.linalg.det(B_inv[gamma_idx]))
                term3 = np.exp(- (1/2) * ( b[gammai_dx].T @ X[gamma_idx,:][:,gamma_idx].T @ np.diag(1/lambdas[t-1][lambdas_idx]) @ X[gamma_idx,:][:,gamma_idx] @ b[gamma_idx] - 2 * b[gamma_idx].T @ X[gamma_idx,:][:,gamma_idx].T @ (np.ones(len(lambdas_idx)) + 1/lambdas[t-1][lambdas_idx]) - (1/(2*nu**2)) * b[gamma_idx].T @ sigma2_inv[gamma_idx,:][:,gamma_idx] @ b[gamma_idx] ))
                
                p[value] = term1 * term2 * term3
                
            p = p / p.sum()
            
            gammas[t,j] = rng.choice(values, p=p)

        # Computing B and b
        term1 = nu**(-2)* sigma2_inv[omegas_idx,:][:,omegas_idx] @ np.diag(1/omegas[t-1][omegas_idx])
        term2 = X[lambdas_idx,:][:,omegas_idx].T @ np.diag(1/lambdas[t-1][lambdas_idx]) @ X[lambdas_idx,:][:,omegas_idx]
        B_inv = term1 + term2
        B = np.linalg.inv(B_inv)
        b = B @ X[lambdas_idx,:][:,omegas_idx].T @ (np.ones(len(lambdas_idx)) + 1/lambdas[t-1][lambdas_idx])

        # Sampling beta using previous state
        betas[t][omegas_idx] = rng.multivariate_normal(b, cov=B)
    
    return betas