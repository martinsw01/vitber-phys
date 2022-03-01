import numpy as np
from scipy.optimize import newton as newtons_method

from variables import sigma0, sigma


def sector_angle_eq(beta):
    return beta - np.sin(beta) - np.pi * sigma / sigma0


def sector_angle_eq_derivative(beta):
    return 1 - np.cos(beta)


def calc_sector_angle():
    return newtons_method(func=sector_angle_eq, x0=2, fprime=sector_angle_eq_derivative)


def main():
    beta = calc_sector_angle()
    print(f"beta={beta}")


if __name__ == '__main__':
    main()
