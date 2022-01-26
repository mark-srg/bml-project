## Bayesian Machine Learning Project
### Based on: Data Augmentation for Support Vector Machines, Polson et al. (2011)
#### Pauline TRUONG & Marc-André SERGIEL

### Installation

In order to be able to use our project, one needs the following Python packages:
- `numpy`
- `matplotlib`
- `pandas`
- `arviz`
- `numpy`
- `sklearn`

### Organization of the repository

In the main folder there are 3 notebooks:
- `mcmc_tests.ipynb`: used to test our implementations of MCMC alpha=1 and spike-and-slab
- `mcmc_study_l1.ipynb`: used to run experiments with the implemented algorithm MCMC alpha=1, compute credible intervals and analyze the chains
- `mcmc_study_sas.ipynb`: used to run experiments with the implemented algorithm MCMC spike-and-slab, compute credible intervals and analyze the chains
- `sklearn_svm.ipynb`: an implementation of scikit-learn' SVM that is used as a baseline to compare our implemented algorithms

The notebooks can be tested independently.  
The `data/` folder contains the e-mail spam dataset.  
The `implementations/` folder contains Python files that implement the sampling methods for both algorithms and some useful methods.
