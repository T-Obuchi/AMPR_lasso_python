import math
import numpy as np
import scipy.linalg as spa
import time
import matplotlib.pyplot as plt
import os, sys

pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pardir)

from routine.AMPR_lasso_track import AMPR_lasso_track
from routine.SE_AMPR import SE_AMPR


# Parameters 
alpha = 0.5             # Ratio of dataset size to model dimensionaltiy
N = 20000               # Model dimensionality (number of covariates)
M = math.ceil(alpha*N)  # Dataset size (number of measurements)
rho0 = 0.2              # Ratio of non-zero components in synthetic data
K0 = math.ceil(rho0*N)  # Number of non-zero components
sigmaN2 = 0.01          # Component-wise noise strength
sigmaB2 = 1./rho0       # Component-wise signal strength

# Sample generation
seed = 1
np.random.seed(seed)
beta0 = np.zeros(N)
beta0[0:K0] = math.sqrt(sigmaB2) * np.random.randn(K0)          # True signal
X = np.random.randn(M, N)/math.sqrt(N)                          # Covariates
Y = np.dot(X, beta0) + math.sqrt(sigmaN2) * np.random.randn(M)  # Response

# Other parameters
lambda1 = 1     # l1 regularization coefficient
w = 1           # 1: no penalty randomization, 1/2: recommended in stability selection
p_w = 0         # 0: no penalty randomization, 1/2: recommended in stability selection
tau = 1         # 1: standard bootstrap,       1/2: recommended in stability selection

# AMPR 
chi_in = np.zeros(N)
W_in = np.zeros(N)
beta_in = np.zeros(N)


startTime = time.time()
fit_AMPR_beta, fit_AMPR_chi, fit_AMPR_W, fit_AMPR_Pi, fit_AMPR_A, fit_AMPR_B, fit_AMPR_C \
     = AMPR_lasso_track(Y, X, lambda1, w, p_w, tau, beta_in, chi_in, W_in)
endTime = time.time()
t1 = endTime - startTime

# SE
chi_til_in = np.mean(chi_in)
W_til_in = np.mean(W_in)
MSE_in = spa.norm(beta0 - beta_in) ** 2 / N

startTime = time.time()
fit_SE_chi, fit_SE_W, fit_SE_MSE \
    = SE_AMPR(alpha, sigmaN2, rho0, sigmaB2, lambda1, w, p_w, tau, chi_til_in, W_til_in, MSE_in)
endTime = time.time()
t2 = endTime - startTime

# AMPR result
_, MAXIT = fit_AMPR_beta.shape
chiV = np.zeros(MAXIT)
WV = np.zeros(MAXIT)
MSEV = np.zeros(MAXIT)

for i in range(0, MAXIT):
    chiV[i] = np.mean(fit_AMPR_chi[:, i])
    WV[i] = np.mean(fit_AMPR_W[:, i])
    MSEV[i] = spa.norm(beta0-fit_AMPR_beta[:, i]) ** 2 / N


STEPS = np.arange(0,MAXIT)

# SE result
(MAXIT_SE, ) = fit_SE_chi.shape
STEPS_SE = np.arange(0, MAXIT_SE)

# Comparison
plt.scatter(STEPS, chiV,  color='blue', marker='*', label=r'$\tilde{\chi} (AMPR)$')
plt.scatter(STEPS, WV,  color='green', marker='o',  label=r'$\tilde{W} (AMPR)$')
plt.scatter(STEPS, MSEV,  color='red', marker='+',  label='$MSE (AMPR)$')

plt.plot(STEPS_SE, fit_SE_chi,  color='blue',  label=r'$ \tilde{\chi} (SE)$')
plt.plot(STEPS_SE, fit_SE_W,  color='green',  label=r'$\tilde{W} (SE)$')
plt.plot(STEPS_SE, fit_SE_MSE,  color='red',  label='$MSE (SE)$')

plt.xlabel('Iteration step tt')
plt.legend(loc='upper right', fontsize=8)
plt.title('$\lambda$=' + str(lambda1) + ',w=' + str(w) + ',p_w=' + str(p_w) + r',$ \tau$=' + str(tau))
plt.show()



