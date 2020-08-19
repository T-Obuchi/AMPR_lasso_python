import math
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import sys

pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from routine.AMPR_lasso import AMPR_lasso




# Parameters
alpha = 0.5             # Ratio of dataset size to model dimensionaltiy
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
beta0[0:K0] = math.sqrt(sigmaB2) * np.random.randn(K0)          # True signal
X = np.random.randn(M, N)/math.sqrt(N)                          # Covariates
Y = np.dot(X, beta0) + math.sqrt(sigmaN2) * np.random.randn(M)  # Responses

# Other parameters
lambda1 = 1     # l1 regularization coefficient
w = 0.5         # 1: no penalty randomization, 0.5: recommended in stability selection
p_w = 0.5       # 0: no penalty randomization, 0.5: recommended in stability selection
tau = 0.5       # 1: standard bootstrap,       0.5: recommended in stability selection

# AMPR
startTime = time.time()
beta, _, W, Pi, _, _, _, _, _ = AMPR_lasso(Y, X, lambda1, w, p_w, tau)
endTime = time.time()
t1 = endTime - startTime

fig = plt.figure()
ax = fig.add_subplot(3, 1, 1)
ax.scatter(np.arange(0, len(beta)), beta, c='blue', marker='*', s=10)
ax.set_ylim(-2.0, 1.0)
ax.set_title(r"($ \lambda $ = " + str(lambda1) + ")")
ax.set_ylabel(r"$ \overline{\beta} $")
ax = fig.add_subplot(3, 1, 2)
ax.scatter(np.arange(0, len(W)), W, c='green', marker='o', s=10)
ax.set_ylim(0.0, 4.0)
ax.set_ylabel(r'$W$')
ax = fig.add_subplot(3, 1, 3)
ax.scatter(np.arange(0, len(Pi)), Pi, c='red', marker='+', s=10)
ax.set_ylim(0.0, 0.7)
ax.set_xlabel('INDEX')
ax.set_ylabel(r"$\Pi$")
plt.show()

