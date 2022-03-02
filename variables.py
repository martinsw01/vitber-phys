import numpy as np

from sector_angle import calc_sector_angle

R = 1
H = 4 * R / (3 * np.pi)
g = 9.81
sigma0 = 1000
sigma = 500
beta = calc_sector_angle(sigma, sigma0)

y_M0 = R * np.cos(beta / 2)
y_C0 = y_M0 - H
y_MB0 = 4 * R * np.sin(beta / 2) ** 3 / (3 * (beta - np.sin(beta)))
y_B0 = y_M0 - y_MB0
y_D0 = y_M0 - R

A0 = 1 / 2 * R ** 2 * np.pi * sigma / sigma0
m = A0 * sigma0
I_M = 1 / 2 * m * R ** 2
I_C = I_M - m * H ** 2

F_G = -m * g


def calc_gamma(theta, y_C):
    A = np.cos(beta / 2)
    B = (4 / (3 * np.pi))
    return 2 * np.arccos(A - B * (1 - np.cos(theta)) + (y_C - y_C0) / R)


def A(theta, y_C):
    gamma = calc_gamma(theta, y_C)
    return 0.5 * R**2 * (gamma - np.sin(gamma))


def F_B(A):
    return A * sigma0 * g


def tau_B(theta, A):
    return -F_B(A) * H * np.sin(theta)
