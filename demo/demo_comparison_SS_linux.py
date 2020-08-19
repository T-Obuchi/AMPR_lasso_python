import math
import scipy
import numpy as np
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import glmnet_python
from glmnet import glmnet
from glmnetCoef import glmnetCoef
import os
import sys
pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pardir)

from routine.pathwiseSS_AMPR import pathwiseSS_AMPR

# Parameters for sample generation
alpha = 5                       # Ratio of dataset size to model dimensionaltiy
N = 1000                        # Model dimensionality(number of covariates)
M = math.ceil(alpha * N)        # Dataset size(number of responses)
rho0 = 0.2                      # Ratio of non - zero components in synthetic data
K0 = int(math.ceil(rho0 * N))   # Number of non - zero components
sigmaN2 = 0.01                  # Component - wise noise strength
sigmaB2 = 1. / rho0             # Component - wise signal strength

# Sample generation
seed = 1
np.random.seed(seed)
beta0 = np.zeros(N)
beta0[0: K0] = math.sqrt(sigmaB2) * np.random.randn(K0)
X = np.random.randn(M, N) / math.sqrt(N)
Y = np.dot(X, beta0) + math.sqrt(sigmaN2) * np.random.randn(M)

# Other parameters
lambdaV = np.arange(3, 0, -0.04)    # l1 coefficients
w = 0.5     # 1: no penalty randomization, 0.5: recommended in stability selection
p_w = 0.5   # 0: no penalty randomization, 0.5: recommended in stability selection
tau = 0.5   # 1: standard bootstrap, 0.5: recommended in stability selection

# AMPR
startTime = time.time()
_, _, pathfit_Pi, _, pathfit_lambda,  = pathwiseSS_AMPR(Y, X, lambdaV, w, p_w, tau)
endTime = time.time()
t1 = endTime - startTime

# Numerical sampling for stability selection using glmnet
NEXP = 1000

thre = 10 ** (-8)
Llam = len(lambdaV)
COUNT = np.zeros((N, Llam))

lambda1 = lambdaV/(M*tau)

startTime = time.time()
for nexp in range(NEXP):
    # Initialization
    np.random.seed(nexp)
    betaV = np.zeros((N, Llam))

    # Penalty coeff.randomization
    r1 = np.random.rand(N)
    w_on = (r1 < p_w)
    w_off = np.logical_not(w_on)

    # Data resampling
    r2 = np.random.rand(int(M * tau))
    Ibs = np.ceil(r2 * M)
    Ybs = Y[Ibs.astype(int)-1]
    Xbs = X[Ibs.astype(int)-1, :]
    M_tmp = len(Ybs)

    # Reweighting columns by coeff.randomization
    Xmod = np.zeros((M_tmp, N))
    Xmod[:, w_on] = w * Xbs[:, w_on]
    Xmod[:, w_off] = Xbs[:, w_off]

    # Glmnet
    fit = glmnet(x=Xmod, y=Ybs, family='gaussian', alpha=1.0, maxit=10 ** 8, intr=False, standardize=False, thresh=1e-10, lambdau=lambda1)
    glmnet_ret = glmnetCoef(fit)

    # Recovering original weight
    betaV[w_on] = w * glmnet_ret[1:][w_on]
    betaV[w_off] = glmnet_ret[1:][w_off]

    # Counting non - zero components
    COUNT = COUNT + (abs(betaV) > thre).astype(int)

endTime = time.time()
t2 = endTime - startTime

Pi_exp = COUNT / NEXP  # Stability path

# Plot of stability path
print([t1, t2])  # elapsed time figure hold on

plt.figure(1)
for i in range(1, K0, 100):
    plt.plot(pathfit_lambda, pathfit_Pi[i,:], color='red', marker='o', linestyle='None', label='AMPR' if i == 1 else "")
    plt.plot(lambdaV, Pi_exp[i, :], color='blue', marker='*', linestyle='None', label='Numerical' if i == 1 else "")
for i in range(K0 + 1, N, 200):
    plt.plot(pathfit_lambda, pathfit_Pi[i,:], color='red', marker='o', linestyle='None')
    plt.plot(lambdaV, Pi_exp[i,:], color='blue', marker='*', linestyle='None')
plt.xlabel('$ \lambda $')
plt.ylabel('$ \Pi $')
plt.gca().set_xscale('log')
plt.title('Some stability paths')
plt.legend(fontsize=8, loc='best')
plt.show()


