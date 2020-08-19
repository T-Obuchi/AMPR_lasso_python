import math
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import os, sys

pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pardir)

from routine.pathwiseSS_AMPR import pathwiseSS_AMPR

# Parameters for sample generation
alpha = 5               # Ratio of dataset size to model dimensionaltiy
N = 1000                # Model dimensionality (number of covariates)
M = math.ceil(alpha*N)  # Dataset size (number of responses)
rho0 = 0.2              # Ratio of non-zero components in synthetic data
K0 = math.ceil(rho0*N)  # Number of non-zero components

sigmaN2 = 0.01          # Component-wise noise strength
sigmaB2 = 1./rho0       # Component-wise signal strength

# Sample generation
seed = 1
np.random.seed(seed)
beta0 = np.zeros(N)
beta0[0:K0] = math.sqrt(sigmaB2) * np.random.randn(K0)          # Non-zero components of true signal
X = np.random.randn(M, N)/math.sqrt(N)                          # Covariates
Y = np.dot(X, beta0) + math.sqrt(sigmaN2) * np.random.randn(M)  # Responses

# Other parameters
lambdaV = np.arange(3.00, 0.04 - 0.04, -0.04)  # l1 coefficients
w = 0.5         # 1: no penalty randomization, 0.5: recommended in stability selection
p_w = 0.5       # 0: no penalty randomization, 0.5: recommended in stability selection
tau = 0.5       # 1: standard bootstrap,       0.5: recommended in stability selection

# AMPR
startTime = time.time()
pathfit_beta, pathfit_W, pathfit_Pi, pathfit_count, pathfit_lambda = pathwiseSS_AMPR(Y, X, lambdaV, w, p_w, tau)
endTime = time.time()
t1 = endTime - startTime

# Plot of stability path
plt.figure(1)
for i in np.arange(0, K0-1, 100):
    plt.plot(pathfit_lambda, pathfit_Pi[i, :], color='blue', marker='*')

for i in np.arange(K0, N-1, 200):
    plt.plot(pathfit_lambda, pathfit_Pi[i, :], color='red', marker='*')

plt.xlabel("$ \lambda $")
plt.ylabel("$ \Pi $")
plt.title("Some stability paths")
plt.gca().set_xscale('log')
plt.show()

# Plot of confidence interval
TP = pathfit_Pi[0:K0, :]    # Stability path for non-zero components
FP = pathfit_Pi[K0:N, :]    # Stability path for zero components

TP_med = np.median(TP, axis=0)
TP_med_pos = np.array(pd.DataFrame(TP).quantile(0.84)) - TP_med
TP_med_neg = TP_med - np.array(pd.DataFrame(TP).quantile(0.16))
FP_med = np.median(FP, axis=0)
FP_med_neg = FP_med - np.array(pd.DataFrame(FP).quantile(0.16))
FP_med_pos = np.array(pd.DataFrame(FP).quantile(0.84)) - FP_med

plt.figure(2)
plt.errorbar(lambdaV, FP_med, [FP_med_neg, FP_med_pos], color='red', marker='o', label='FP')
plt.errorbar(lambdaV, TP_med, [TP_med_neg, TP_med_pos], color='blue', marker='o', label='TP')
plt.gca().set_xscale('log')
plt.legend(fontsize=18, loc='upper left')
plt.ylim([0, 1.2])
plt.xlabel("$ \lambda $", fontsize=18)
plt.ylabel("Probability", fontsize=18)
plt.show()


