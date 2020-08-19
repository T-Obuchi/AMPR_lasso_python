import math
import numpy as np
import scipy.special as sp
from scipy.stats import poisson


def mysoft_th(A, B, s_lambda):
    Llam = len(s_lambda)
    LB = len(B)

    out = np.zeros((LB, Llam))

    for ilam in range(0, Llam):
        out[:, ilam] = (B - np.sign(B)*s_lambda[ilam]) * ((abs(B) > s_lambda[ilam]).astype(np.int)) / A

    return out


def myNorm(x):
    y = np.max(np.abs(x))
    return y * np.sqrt(((x / y)**2).sum())


def AMPR_lasso(*args):
    # Parameters

    if len(args) < 3:
        print('three input arguments are needed at least')
        return

    Y = args[0]
    X = args[1]
    lambda1 = args[2]

    X2 = X ** 2
    M, N = X.shape

    if len(args) >= 4:
        w = args[3]
    if len(args) >= 5:
        p_w = args[4]
    if len(args) >= 6:
        tau = args[5]
    if len(args) >= 7:
        beta_in = args[6]
    if len(args) >= 8:
        chi_in = args[7]
    if len(args) >= 9:
        W_in = args[8]
    if len(args) >= 10:
        gamma_min = args[9]
    if len(args) >= 11:
        gamma_max = args[10]

    if len(args) < 4 or (w is None) or w > 1 or w < 0:
        w = 1
    if len(args) < 4 or (p_w is None) or p_w > 1 or p_w < 0:
        p_w = 0
    if len(args) < 4 or (tau is None) or tau > 1 or tau < 0:
        tau = 1
    if len(args) < 7 or (beta_in is None):
        beta_in = np.zeros(N)
    if len(args) < 8 or (chi_in is None):
        chi_in = np.zeros(N)
    if len(args) < 9 or (W_in is None):
        W_in = np.zeros(N)
    if len(args) < 10 or (gamma_min is None):
        gamma_min = 1
    if len(args) < 11 or (gamma_max is None):
        gamma_max = 1
    if gamma_max < gamma_min:
        gamma_max = gamma_min

    # Integration Measures
    CMAX = 100
    c = np.arange(0, CMAX+1)
    Pc = poisson.pmf(c, tau)
    Pc = Pc / sum(Pc)  # Poisson　measure

    MAX = 10
    dz = 0.01
    z = np.arange(-MAX, MAX + dz, dz)
    Dz = dz * np.exp(-z ** 2 / 2) / math.sqrt(2 * math.pi)  # Gaussian　measure
    S_lam = lambda1 * np.array([1/w, 1])  # Set of lambda
    P_lam = np.array([p_w, 1-p_w])  # Measure on set of lambda

    # Initial condition
    f0 = np.zeros(M)
    f1 = np.zeros(M)
    f2 = np.zeros(M)
    chi_mu = np.dot(X2, chi_in)
    W_mu = np.dot(X2, W_in)

    for mu in range(0, M):
        f1[mu] = np.dot(Pc, c/(1+c*chi_mu[mu]))
        f2[mu] = np.dot(Pc, (c/(1+c*chi_mu[mu]))**2)

    beta = beta_in.copy()
    W = W_in.copy()
    chi = chi_in.copy()
    a = f1 * (Y - np.dot(X, beta))

    # AMPR　main loop
    ERR = 100
    MAXIT = 10000
    gamma = gamma_min  # Damping factor
    count = 1

    while ERR > 10**(-6):
        beta_pre = beta.copy()
        W_pre = W.copy()

        # Moments to Conjugates
        chi_mu = np.dot(X2, chi)
        W_mu = np.dot(X2, W)

        for mu in range(0,  M):
            f1[mu] = np.dot(Pc,  c/(1+c*chi_mu[mu]))
            f2[mu] = np.dot(Pc, (c/(1+c*chi_mu[mu])) ** 2)

        a = f1 * (Y - np.dot(X, beta)+chi_mu * a)
        A = np.dot(X2.T, f1)
        B = np.dot(X.T, a) + A * beta
        C = np.dot(X2.T, (W_mu * f2 + (f2-f1**2) * ((a / f1)**2)))

        # Conjugates to Moments　
        for i in range(0, N):
            b_tmp = mysoft_th(A[i], B[i] + math.sqrt(C[i]) * z, S_lam)

            beta[i] = (1 - gamma) * beta[i] + gamma * np.dot(P_lam, np.dot(Dz, b_tmp))
            W[i] = (1 - gamma) * W[i] + gamma * (np.dot(P_lam, np.dot(Dz, b_tmp ** 2)) - beta[i] ** 2)
            chi[i] = (1 - gamma) * chi[i] + gamma * (1 / (2 * A[i])) \
                     * (np.dot(P_lam, sp.erfc((S_lam - B[i]) / math.sqrt(2 * C[i]))
                        + np.dot(P_lam, sp.erfc((S_lam + B[i]) / math.sqrt(2 * C[i])))))

        ERR = myNorm(beta - beta_pre) / myNorm(beta) + myNorm(W - W_pre) / myNorm(W)

        # Forced termination
        if count >= MAXIT:
            print(['AMPR did not converge in MAXIT=', str(MAXIT), '. The result might be inaccurate.'])
            wflag = 1
            break
        else:
            wflag = 0
            count = count + 1
            # Damping factor tuning
            gamma = gamma_max * (1 - 1 / (count ** 0.1)) + gamma_min / (count ** 0.1)

    # Positive probabilities
    P_pos = np.zeros(N)

    for i in range(0, N):
        P_pos[i] = 0.5 * np.dot(P_lam, sp.erfc((S_lam - B[i]) / math.sqrt(2 * C[i]))
                                + sp.erfc((S_lam + B[i]) / math.sqrt(2 * C[i])))

    return beta, chi, W, P_pos, A, B, C, count, wflag

