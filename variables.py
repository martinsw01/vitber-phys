import numpy as np
from functools import cache

from sector_angle import calc_sector_angle

R = 10
H = 4 * R / (3 * np.pi)
g = 9.81
sigma0 = 1000
sigma = 500


def calc_beta(m_L):
    return calc_sector_angle(sigma, sigma0, R, m_L)


beta0 = calc_beta(0)


def y_M0(beta=beta0):
    return R * np.cos(beta / 2)


# Called many times with the same argument
@cache
def y_C0(beta=beta0):
    return y_M0(beta) - H


def y_MB0(beta=beta0):
    return 4 * R * np.sin(beta / 2) ** 3 / (3 * (beta - np.sin(beta)))


def y_B0(beta=beta0):
    return y_M0(beta) - y_MB0(beta)


def y_D0(beta=beta0):
    return y_M0(beta) - R


A0 = 1 / 2 * R ** 2 * np.pi * sigma / sigma0
m = A0 * sigma0
I_M = 1 / 2 * m * R ** 2
I_C = I_M - m * H ** 2
omega_0 = (m * g * H / I_C) ** 0.5

F_G = -m * g


def calc_gamma(theta, y_C, beta=beta0):
    A = np.cos(beta / 2)
    B = (4 / (3 * np.pi))
    if abs(A - B * (1 - np.cos(theta)) + (y_C - y_C0(beta)) / R) > 1:
        return 0
    return 2 * np.arccos(A - B * (1 - np.cos(theta)) + (y_C - y_C0(beta)) / R)


def A(theta, y_C, beta=beta0):
    gamma = calc_gamma(theta, y_C, beta)
    return 0.5 * R ** 2 * (gamma - np.sin(gamma))


def F_B(A):
    return A * sigma0 * g


def tau_B(theta, A):
    return -F_B(A) * H * np.sin(theta)


def F_f(omega, gamma, k_f):
    return -k_f * gamma * R * omega


def tau_f(omega, y_C, gamma, k_f):
    return F_f(omega, gamma, k_f) * (y_C - R * (np.cos(gamma/2) - 1))


def F_w(t, F_w0, omega_w):
    return F_w0 * np.cos(omega_w * t)


def tau_w(t, y_C, F_w0, omega_w):
    return y_C * F_w(t, F_w0, omega_w)


def F_Lx(theta, m_L):
    return m_L * g * np.cos(theta) * np.sin(theta)


def F_Ly(theta, m_L):
    return -m_L * g * np.cos(theta) ** 2


def tau_L(m_L, s_L):
    return m_L * g * s_L
