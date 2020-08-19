import math
import scipy
import numpy as np
import pandas as pd
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import glmnet_python
from glmnet import glmnet
from glmnetPredict import glmnetPredict
from glmnetCoef import glmnetCoef
from glmnetPlot import glmnetPlot
from glmnetPrint import glmnetPrint
import os
import sys
pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pardir)

from routine.AMPR_lasso import AMPR_lasso



# Parameters for sample generation
alpha = 0.5             # Ratio of dataset size to model dimensionaltiy
N = 1000                # Model dimensionality (number of covariates)
M = math.ceil(alpha*N)  # ceil -> math.ceil Dataset size (number of responses)
rho0 = 0.2              # Ratio of non-zero components in synthetic data
K0 = math.ceil(rho0*N)  # Number of non-zero components
sigmaN2 = 0.01          # Component-wise noise strength
sigmaB2 = 1./rho0       # Component-wise signal strength

# Sample generation
seed = 1
np.random.seed(seed)
beta0 = np.zeros(N)
beta0[0:K0] = math.sqrt(sigmaB2) * np.random.randn(K0)     			# True signal
X = np.random.randn(M, N)/math.sqrt(N)                   			# Covariates
Y = np.dot(X, beta0) + math.sqrt(sigmaN2) * np.random.randn(M)     	        # Responses

# Other parameters
lambda1 = 1    	# l1 regularization coefficient
w = 0.5         # 1: no penalty randomization, 0.5: recommended in stability selection
p_w = 0.5       # 0: no penalty randomization, 0.5: recommended in stability selection
tau = 0.5       # 1: standard bootstrap,       0.5: recommended in stability selection

# AMPR
startTime = time.time()
fit_AMPR_beta, _, fit_AMPR_W, fit_AMPR_Pi, _, _, _, _, _ = AMPR_lasso(Y,X,lambda1,w,p_w,tau);
endTime = time.time()
t1 = endTime - startTime

# Numerical sampling using glmnet
NEXP = 1000
betaV = np.zeros((N, NEXP))
lambda1 = lambda1/(M*tau)

startTime = time.time()
for nexp in np.arange(0, NEXP):
    # Initialization
    np.random.seed(nexp)

    # Penalty coeff. randomization
    r1 = np.random.rand(N)
    w_on = r1 < p_w   
    w_off = np.logical_not(w_on)  

    # Data resampling
    r2 = np.random.rand(int(M*tau))
    Ibs = np.ceil(r2*M)
    Ybs = Y[Ibs.astype(int)-1]  # -1 is caused from matlab idx to python idx?
    Xbs = X[Ibs.astype(int)-1, :]
    M_tmp = len(Ybs)

    # Reweighting columns by coeff. randomization
    Xmod = np.zeros((M_tmp, N))

    Xmod[:, w_on] = w * Xbs[:, w_on]
    Xmod[:, w_off] = Xbs[:, w_off]

    fit = glmnet(x=Xmod, y=Ybs, family='gaussian', alpha=1.0, maxit=10 ** 8, intr=False, standardize=False, thresh=1e-10, lambdau=np.array([0.02, lambda1]))
    glmnet_ret = glmnetCoef(fit)

    betaV[w_on, nexp] = w * glmnet_ret[:, 1][1:NEXP+1][w_on]
    betaV[w_off, nexp] = glmnet_ret[:, 1][1:NEXP+1][w_off]

endTime = time.time()
t2 = endTime - startTime

print([t1, t2])  # elapsed time

# Mean value of beta
plt.figure(1)
_, NEXP = betaV.shape
plt.scatter(fit_AMPR_beta, np.mean(betaV[:, 0:NEXP], 1), color='blue', marker='o')
plt.plot([min(fit_AMPR_beta), max(fit_AMPR_beta)], [min(fit_AMPR_beta), max(fit_AMPR_beta)], color='black', marker='_')
plt.xlabel('Semi-analitic')
plt.ylabel('Numerical')
plt.title(r"$ \overline{\beta_i} $ ($ \lambda$=" + str(lambda1) + ")") 
plt.show()

# Intra-sample variance
plt.figure(2)
W_exp = np.sum(betaV[:,0:NEXP] ** 2, axis=1) / NEXP - (np.sum(betaV[:, 0:NEXP], axis=1)/NEXP) ** 2
plt.scatter(fit_AMPR_W, W_exp, color='blue', marker='o')
plt.plot(np.array([min(fit_AMPR_W), max(fit_AMPR_W)]), np.array([min(fit_AMPR_W), max(fit_AMPR_W)]), color='k', marker='_')
plt.xlabel('Semi-analitic')
plt.ylabel('Numerical')
plt.title(r"$ W_i $ ($ \lambda$=" + str(lambda1) + ")") 
plt.show()


# Positive probability
plt.figure(3)
thre = 10 ** (-8)
MASK = (abs(betaV) > thre).astype(int)
P_pos = np.mean(MASK, 1)
v_max = max(fit_AMPR_Pi)
v_min = min(fit_AMPR_Pi)
plt.scatter(fit_AMPR_Pi, P_pos, color='b', marker='o')
plt.plot(np.array([v_min, v_max]), np.array([v_min, v_max]), color='k', marker='_')
plt.xlabel('Semi-analitic')
plt.ylabel('Numerical')
plt.title(r'$\Pi_i$ ($\lambda$=' + str(lambda1) + ')')
plt.show()
