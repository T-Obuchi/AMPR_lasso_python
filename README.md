# AMPR_lasso_python
Approximate Message Passing with Resampling (AMPR) for Lasso

This is free software, you can redistribute it and/or modify it under the terms of the GNU General Public License, version 3 or above. See LICENSE.txt for details.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

# DESCRIPTION
Compute the bootstrap average of estimator in Lasso by using a semi-analytic approximate formula called AMPR.
Randomization to penalty coefficients is introduced according to stability selection framework.
Main quantities to be computed are the mean and variance of the Lasso estimator over the bootstrap samples,
and the positive probability of the estimator (probability such that the estimator takes non-zero values).

# USAGE
```python
    beta, chi, W, Pi, A, B, C, count, flag = AMPR_lasso(Y,X,lambda)
    (Some additional inputs can be given. Examples:  
    AMPR_lasso(Y,X,lambda,w,p_w,tau),
    AMPR_lasso(Y,X,lambda,w,p_w,tau,beta_in,W_in,chi_in),
    AMPR_lasso(Y,X,lambda,None,None,None,beta_in,W_in,chi_in,None,None),
    AMPR_lasso(Y,X,lambda,w,p_w,tau,None,None,None,None,None).
    Use None to apply the default value.)
```
Inputs:
- *Y*:         Response vector (M dimensional vector).

- *X*:         Matrix of covariates (M*N dimensional matrix).

- *lambda*:    l1 regularizaiton coefficient.   

- *w*:         Reweighting parameter to the regularization coefficients used in stability selection.
               Default value is *w=1* corresponding to the case of the non-randomized penalty.
               A recommended value for stability selection is *w=0.5*.

- *p_w*:       Fraction of randomization of the regularization coefficients used in stability selection.
               Default value is *p_w=0* corresponding to the case of the non-randomized penalty.
               A recommended value for stability selection is *p_w=0.5*.

- *tau*:       Ratio of the size of bootstrap sample to the size of the original dataset.
               Default value is *tau=1* corresponding to the Bootstrap method's convention.
               A recommended value for stability selection is *tau=0.5*.

- *beta_in*:   Initial estimate of mean value of covariates' coefficients (N dimensional vector).
               Not necessarily needed (but better to be appropriately given for faster convergence).

- *chi_in*:    Initial estimate of rescaled intra-sample variance of covariates' coefficients (N dimensional vector).
               Not necessarily needed (but better to be appropriately given for faster convergence).

- *W_in*:      Initial estimate of variance of covariates' coefficients (N dimensional vector).
               Not necessarily needed (but better to be appropriately given for faster convergence).

- *gamma_min*: Minimum damping factor. (Not necessarily needed, default value is 1)

- *gamma_max*: Maximum damping factor. (Not necessarily needed, default value is 1)

Outputs:
- *beta*:  Mean value of covariates' coefficients (N dimensional vector).

- *chi*:   Rescaled intra-sample variance of covariates' coefficients (N dimensional vector).

- *W*:     Variance of covariates' coefficients (N dimensional vector).

- *Pi*:    Positive probabilities of covariates' coefficients (N dimensional vector).
               (Probabilities such that covariates' coefficients take non-zero values.)

- *A, B, C*:
               Parameters (N dimensional vectors) characterizing
               the probability distributions of covariates' coefficients (see [1] for details).

- *count*: Iteration steps until convergence.

- *flag*:  flag for checking convergence. (0: converged, 1: not converged.).

As other related codes, *SE_AMPR.py* solves state evolution (SE) equations associated with AMPR
and returns dynamical behavior of macroscopic quantities up to MAXIT(=30 in default) steps.
For comparison with the SE result, *AMPR_lasso_track.py* returns the AMPR messages of the first MAXIT(=30 in default) steps.
In SE_AMPR.py, the true parameter *beta_0* is assumed to be Bernoulli-Gaussian whose non-zero component density is *rho_0* 
and the Gaussian's mean and variance are assumed to be zero and *sigmaB2*. 
These parameters *rho_0 and *sigmaB2* are required as arguments. 

Another utility code is *pathwiseSS_AMPR.py*. This computes the so-called stability path,
positive probability of covariate coefficient against the regularization parameter *lambda*,
by using AMPR. The stability path provides useful information for variable selection.

For the theoretical background of AMPR and stability selection, see REFERENCE [1] and [2], respectively.

# DEMONSTRATION
In the "demo" folder, demonstration codes for AMPR_lasso, SE_AMPR, and pathwiseSS_AMPR are available:
demo_AMPR.py, demo_SE.py, and demo_pathwiseSS.py, respectively.
For comparison with direct numerical sampling, demo_comparison_AMPR_mac.py and demo_comparison_SS_mac.py for Mac,
and demo_comparison_AMPR_linux.py and demo_comparison_SS_linux.py for Linux are available.

**Requirement**:
To solve Lasso for each bootstrap sample, python-glmnet (https://github.com/civisanalytics/python-glmnet) is required for demo_comparison_AMPR_mac.py and demo_comparison_SS_mac.py.
Meanwhile, glmnet_python (https://web.stanford.edu/~hastie/glmnet_python/) is required for demo_comparison_AMPR_linux.py and demo_comparison_SS_linux.py.

# REFERENCE
[1] Tomoyuki Obuchi and Yoshiyuki Kabashima: Semi-analytic resampling in Lasso, arXiv:1802.10254.

[2] Nicolai Meinshausen and Peter Buhlmann: Stability selection,
Journal of the Royal Statistical Society: Series B (Statistical Methodology), 72(4):417--473, 2010.
