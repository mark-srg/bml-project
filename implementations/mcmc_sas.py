# Sampling method for the MCMC spike-and-slab algorithm of Polson et al. (2011)

from .utils import *
import numpy as np

def sample(X, sigma, rng, nu=100, pi=None, T=100):
    
    # Extracting useful information
    n, k = X.shape
    # By default, if pi wasn't provided we set pi_j = 1/2
    if pi is None:
        pi = 1/2*np.ones(k) 
    
    # Computing the inverse Sigma matrix
    sigma2_inv = np.diag(1/sigma**2)
    
    # Initializing the arrays that will store the samples
    betas = np.zeros((T,k))
    lambdas = np.zeros((T,n))
    gammas = np.zeros((T,k))
    
    # Initializing the variables
    betas[0] = rng.normal(0,1,size=k)
    lambdas[0] = rng.wald(1,1, size=n)  # doesn't matter, we don't use it
    gammas[0] = rng.binomial(n=1, p=pi)
    gamma_idx = gammas[0].astype(bool)
    
    # List to keep track of indices to remove for stability
    lambdas_idx = list(range(n))    
    
    # Computation of B_gamma and b_gamma as we need them for the first iteration
    B_gamma_inv = (1/nu**2) * sigma2_inv[gamma_idx,:][:,gamma_idx] + X[:,gamma_idx].T @ np.diag(1/lambdas[0]) @ X[:,gamma_idx]
    B_gamma = np.linalg.inv(B_gamma_inv)
    b_gamma = B_gamma @ X[:,gamma_idx].T @ (np.ones(n) + 1/lambdas[0])

    for t in range(1,T):
        
        if t % 500 == 0:
            print("Step {}".format(t))
            
        # Sampling lambda using beta
        for i in lambdas_idx:
            lambda_inv = rng.wald(1/np.abs(1-X[i][gamma_idx].T @ betas[t-1][gamma_idx]), 1)
            if lambda_inv == 0.0:
                lambdas[t,i] = np.inf
            else:
                lambdas[t,i] = 1/lambda_inv
        
        # Removing infinite lambdas
        num_lambda = numerical(1/lambdas[t])
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
                
        # Compute XΛ⁻¹X and X.T(1+λ⁻¹) to avoid recomputing them
        reusable_result1 = X[lambdas_idx,:].T @ np.diag(1/lambdas[t][lambdas_idx]) @ X[lambdas_idx,:]
        reusable_result2 = X[lambdas_idx,:].T @ (np.ones(len(lambdas_idx)) + 1/lambdas[t][lambdas_idx])
                                                  
        # Sampling gamma using newest available values
        current_gamma = np.copy(gammas[t-1])
        for j in range(k):
            p = np.zeros(2)
            
            # Computing gamma for the two possible values 0 and 1 
            current_gamma_0 = np.copy(gammas[t-1])
            current_gamma_0[j] = 0
            gamma_idx_0 = current_gamma_0.astype(bool)
            current_gamma_1 = np.copy(gammas[t-1])
            current_gamma_1[j] = 1
            gamma_idx_1 = current_gamma_1.astype(bool)
            
            # Creating a list to pick one of them later
            current_gammas = [current_gamma_0, current_gamma_1]
            gamma_idxs = [gamma_idx_0, gamma_idx_1]
            
            # Computing B_gamma and b_gamma for each value of gamma_j
            B_gamma_inv_0 = (1/nu**2) * sigma2_inv[gamma_idx_0,:][:,gamma_idx_0] + reusable_result1[gamma_idx_0,:][:,gamma_idx_0]
            B_gamma_0 = np.linalg.inv(B_gamma_inv_0)
            b_gamma_0 = B_gamma_0 @ reusable_result2[gamma_idx_0]
            B_gamma_inv_1 = (1/nu**2) * sigma2_inv[gamma_idx_1,:][:,gamma_idx_1] + reusable_result1[gamma_idx_1,:][:,gamma_idx_1]
            B_gamma_1 = np.linalg.inv(B_gamma_inv_1)
            b_gamma_1 = B_gamma_1 @ reusable_result2[gamma_idx_1]
            
            # Creating a list to pick one of them later
            B_gammas = [B_gamma_0, B_gamma_1]
            b_gammas = [b_gamma_0, b_gamma_1]
            
            # Computing the probability for the value 0
            term1_0 = np.array([(pi[i]**current_gamma_0[i]) * ((1-pi[i])**(1-current_gamma_0[i])) for i in range(k)]).prod()
            term2_0 = np.sqrt((1/nu**2) * (1/sigma[gamma_idx_0]**2).prod() / np.linalg.det(B_gamma_inv_0))
            term3_0 = np.exp(- (1/2) * ( b_gamma_0.T @ reusable_result1[gamma_idx_0,:][:,gamma_idx_0] @ b_gamma_0 - 2 * b_gamma_0.T @ reusable_result2[gamma_idx_0] - (1/(nu**2)) * b_gamma_0.T @ sigma2_inv[gamma_idx_0,:][:,gamma_idx_0] @ b_gamma_0 ))
            p[0] = term1_0 * term2_0 * term3_0
            
            print(term1_0, term2_0, term3_0)
            
            # Computing the probability for the value 1
            term1_1 = np.array([(pi[i]**current_gamma_1[i]) * ((1-pi[i])**(1-current_gamma_1[i])) for i in range(k)]).prod()
            term2_1 = np.sqrt((1/nu**2) * (1/sigma[gamma_idx_1]**2).prod() / np.linalg.det(B_gamma_inv_1))
            term3_1 = np.exp(- (1/2) * ( b_gamma_1.T @ reusable_result1[gamma_idx_1,:][:,gamma_idx_1] @ b_gamma_1 - 2 * b_gamma_1.T @ reusable_result2[gamma_idx_1] - (1/(nu**2)) * b_gamma_1.T @ sigma2_inv[gamma_idx_1,:][:,gamma_idx_1] @ b_gamma_1 ))
            p[1] = term1_1 * term2_1 * term3_1
            
            print(term1_1, term2_1, term3_1)
            print(p)
            
            # Normalizing the probability
            p = p / p.sum()
            
            # Choosing a value for gamma_j based on the probabilities that we computed
            chosen_value = rng.choice(a=np.arange(2).astype(int), p=p)
            current_gamma = current_gammas[chosen_value]
            gammas[t] = np.copy(current_gamma)
            B_gamma = np.copy(B_gammas[chosen_value])
            b_gamma = np.copy(b_gammas[chosen_value])
            gamma_idx = np.copy(gamma_idxs[chosen_value])

        # Sampling beta using newest available values
        betas[t][gamma_idx] = rng.multivariate_normal(b_gamma, cov=B_gamma)
    
    return betas