## Bayesian Machine Learning Project
### Based on the paper Data Augmentation for Support Vector Machines of Polson et al. (2011)
#### Pauline TRUONG & Marc-André SERGIEL

#### Installation

In order to be able to use our project, one needs the following Python packages:
- `numpy`
- `matplotlib`
- `pandas`
- `arviz`
- `numpy`
- `sklearn`

#### Organization of the repository

In the main folder there are 3 notebooks:
- `mcmc_tests.ipynb`: used to test our implementations of MCMC alpha=1 and spike-and-slab
- `mcmc_study.ipynb`: used to run experiments with the implemented algorithms, compute credible intervals and analyze the chains
- `sklearn_svm.ipynb`: an implementation of scikit-learn' SVM that is used as a baseline to compare our implemented algorithms

The notebooks can be tested independently.  
The `data/` folder contains the e-mail spam dataset.  
The `implementations/` folder contains Python files that implement the sampling methods for both algorithms and some useful methods.
