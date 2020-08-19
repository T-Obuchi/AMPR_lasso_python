import math
import numpy as np
import scipy.special as sp
from scipy.stats import poisson


def soft_threshold(A, B, s_lambda):
    Llam = len(s_lambda)
    LB = len(B)

    out = np.zeros((2, LB, LB))

    for ilam in range(0, Llam):
        out[ilam] = (B - np.sign(B)*s_lambda[ilam]) * ((abs(B) > s_lambda[ilam]).astype(np.int)) / A

    return out


def SE_AMPR(*args):
    # Parameters
    if len(args) < 5:
        print('five input arguments are needed at least')
        return

    alpha = args[0]
    sigmaN2 = args[1]
    rho0 = args[2]
    sigmaB2 = args[3]
    lambda1 = args[4]

    if len(args) >= 6:
        w = args[5]
    if len(args) >= 7:
        p_w = args[6]
    if len(args) >= 8:
        tau = args[7]
    if len(args) >= 9:
        chi_in = args[8]
    if len(args) >= 10:
        W_in = args[9]
    if len(args) >= 11:
        MSE_in = args[10]

    if len(args) < 6 or (w is None) or w > 1 or w < 0:
        w = 1
    if len(args) < 6 or (p_w is None) or p_w > 1 or p_w < 0:
        p_w = 0
    if len(args) < 6 or (tau is None) or tau > 1 or tau < 0:
        tau = 1
    if len(args) < 9 or (chi_in is None):
        chi_in = 0
    if len(args) < 9 or (W_in is None):
        W_in = 0
    if len(args) < 9 or (MSE_in is None):
        MSE_in = rho0 * sigmaB2


    # Integration measures
    # Poisson
    CMAX = 100
    c = np.arange(0, CMAX+1)
    Pc = poisson.pmf(c, tau)
    Pc = Pc / sum(Pc)  # Poisson measure

    # Gaussian
    MAX = 10
    dz = 0.01
    z = np.arange(-MAX, MAX + dz, dz)
    u = z.reshape(z.size, 1)
    z_2d = z + np.zeros(u.size).reshape(u.size, 1)
    u_2d = u + np.zeros(z.size).reshape(1, z.size)
    Dz = dz * np.exp((-z ** 2) / 2) / math.sqrt(2 * math.pi)
    Du = dz * np.exp((-u ** 2) / 2) / math.sqrt(2 * math.pi)

    # Lambda randomization
    S_lam = lambda1 * np.array([1/w, 1])  # Set of lambda
    P_lam = np.array([p_w, 1-p_w])  # Measure on set of lambda

    # Running parameters and initial condition
    MAXIT = 30
    chi = chi_in
    W = W_in
    MSE = MSE_in

    # Save data
    chiV = np.zeros(MAXIT)
    WV = np.zeros(MAXIT)
    MSEV = np.zeros(MAXIT)
    chiV[0] = chi
    WV[0] = W
    MSEV[0] = MSE

    # main loop
    for t in range(1, MAXIT):

        # Main to conjugate
        chi_til = chi
        W_til = W
        f_in = c / (1 + c * chi_til)
        f1 = np.dot(f_in, Pc)
        f2 = np.dot(f_in ** 2, Pc)

        # Second order parameters
        A = alpha * f1
        C = alpha * f2 * W_til + alpha * (f2 - f1 ** 2) * (MSEV[t - 1] + sigmaN2)

        # Main parameters' update
        chi = 0
        W = 0
        MSE = 0

        # zero component's contribution
        v0 = (A ** 2) * (MSEV[t - 1] + sigmaN2) / alpha
        chi = chi + ((1 - rho0) / A) * np.dot(P_lam, sp.erfc(S_lam / math.sqrt(2 * (v0 + C))))
        h = math.sqrt(v0) * u_2d + math.sqrt(C) * z_2d
        beta = soft_threshold(A, h, S_lam)
        beta_ave = np.dot(P_lam[0] * beta[0] + P_lam[1] * beta[1], Dz)
        beta2_ave = np.dot(P_lam[0] * (beta[0] ** 2) + P_lam[1] * (beta[1] ** 2), Dz)
        W = W + (1 - rho0) * np.dot(Du.T,  beta2_ave - beta_ave ** 2)
        MSE = MSE + (1 - rho0) * np.dot(Du.T, beta_ave ** 2)

        # nonzero component's contribution
        v1 = (A ** 2) * (sigmaB2 + (MSEV[t - 1] + sigmaN2) / alpha)
        chi = chi + (rho0 / A) * np.dot(P_lam, sp.erfc(S_lam / math.sqrt(2 * (v1 + C))))
        h_sx = math.sqrt(v1) * u_2d + math.sqrt(C) * z_2d  # signal - crosstalk merged
        beta = soft_threshold(A, h_sx, S_lam)
        beta_ave_sx = np.dot(P_lam[0] * beta[0] + P_lam[1] * beta[1], Dz)
        beta2_ave_sx = np.dot(P_lam[0] * (beta[0] ** 2) + P_lam[1] * (beta[1] ** 2), Dz)
        W = W + rho0 * np.dot(Du.T, beta2_ave_sx - beta_ave_sx ** 2)

        h_xb = A * math.sqrt(sigmaB2) * u_2d + math.sqrt(v0 + C) * z_2d  # crosstalk - bootstrap merged
        beta = soft_threshold(A, h_xb, S_lam)
        beta_ave_xb = np.dot(P_lam[0] * beta[0] + P_lam[1] * beta[1], Dz)
        fst = sigmaB2
        scd = -2 * math.sqrt(sigmaB2) * np.dot(np.reshape(Du * u, len(Du * u)), beta_ave_xb)
        trd = np.dot(np.reshape(Du, len(Du)), beta_ave_sx ** 2)
        MSE = MSE + rho0 * (fst + scd + trd)

        # Save
        chiV[t] = chi
        WV[t] = W
        MSEV[t] = MSE

    fit_chi = chiV
    fit_W = WV
    fit_MSE = MSEV

    return fit_chi, fit_W, fit_MSE
