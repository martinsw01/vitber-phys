import numpy as np

from sector_angle import calc_sector_angle

R = 1
H = 4 * R / (3 * np.pi)
g = 9.81
sigma0 = 1000
sigma = 500
beta = calc_sector_angle(sigma, sigma0)

yM0 = R * np.cos(beta / 2)
yC0 = yM0 - H
yMB0 = 4 * R * np.sin(beta / 2) ** 3 / (3 * (beta - np.sin(beta)))
yB0 = yM0 - yMB0
yD0 = yM0 - R

A0 = 1 / 2 * R ** 2 * np.pi * sigma / sigma0
m = A0 * sigma0
IM = 1 / 2 * m * R ** 2
IC = IM - m * H ** 2


def FG():
    return -m * g


def FB(A):
    return A * sigma0 * g


def tauB(theta, A):
    return -FB(A) * H * np.sin(theta)
