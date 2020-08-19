from scipy.stats import poisson
import math
import numpy as np
from .AMPR_lasso import AMPR_lasso


def pathwiseSS_AMPR(*args):
    # Parameters

    if len(args) < 2:
        print('two input arguments are needed at least')
        return

    Y = args[0]
    X = args[1]

    M, N = X.shape

    if len(args) >= 3:
        lambda1 = args[2]
    if len(args) >= 4:
        w = args[3]
    if len(args) >= 5:
        p_w = args[4]
    if len(args) >= 6:
        tau = args[5]
    if len(args) >= 7:
        gamma_min = args[6]
    if len(args) >= 8:
        gamma_max = args[7]

    if len(args) < 3 or (lambda1 is None):
        Llam = 100
        lambda_max = round(max(abs(np.dot(X.T,Y))),1)
        lambda_min = 10 ** (-2)
        rate = np.exp(math.log(lambda_min/lambda_max)/(Llam-1))
        lambda1 = lambda_max * (rate ** np.arange(0, Llam))
    else:
        Llam = len(lambda1)

    if len(args) < 4 or (w is None) or w > 1 or w < 0:
        w = 1/2
    if len(args) < 5 or (p_w is None) or p_w > 1 or p_w < 0:
        p_w = 1/2
    if len(args) < 6 or (tau is None) or tau > 1 or tau < 0:
        tau = 1/2
    if len(args) < 7 or (gamma_min is None):
        gamma_min = 1
    if len(args) < 8 or (gamma_max is None):
        gamma_max = 1
    if gamma_max < gamma_min:
        gamma_max = gamma_min

    lambda1 = np.sort(lambda1)[:: -1]  # Sort in descending order

    # Initial condition
    beta = np.zeros(N)
    chi = np.zeros(N)
    W = np.zeros(N)

    # Pathwise evaluation of parameters using AMPR
    pathfit_beta = np.zeros((N, Llam))
    pathfit_W = np.zeros((N, Llam))
    pathfit_Pi = np.zeros((N, Llam))
    pathfit_count = np.zeros((N, Llam))

    for ilam in range(0, Llam):
        lambda_tmp = lambda1[ilam]
        fit_beta, fit_chi, fit_W, fit_P_pos, fit_A, fit_B, fit_C, fit_count, fit_wflag  \
            = AMPR_lasso(Y, X, lambda_tmp, w, p_w, tau, beta, chi, W, gamma_min, gamma_max)

        pathfit_beta[:, ilam] = fit_beta
        pathfit_W[:, ilam] = fit_W
        pathfit_Pi[:, ilam] = fit_P_pos
        pathfit_count[:, ilam] = fit_count
        beta= fit_beta
        chi = fit_chi
        W = fit_W

        if fit_wflag == 1:
            print(['AMPR did not converge and terminate at lambda=', str(lambda_tmp),
                   '. Smaller lambda is not examined'])
            break

    # Output
    pathfit_beta = pathfit_beta[:, 0:ilam+1]
    pathfit_W = pathfit_W[:, 0:ilam+1]
    pathfit_Pi = pathfit_Pi[:, 0:ilam+1]
    pathfit_count = pathfit_count[:, 0:ilam+1]
    pathfit_lambda = lambda1[0:ilam+1]

    return pathfit_beta, pathfit_W, pathfit_Pi, pathfit_count, pathfit_lambda

