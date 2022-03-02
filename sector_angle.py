import numpy as np
from scipy.optimize import newton as newtons_method


def sector_angle_eq(sigma, sigma0):
    return lambda beta: beta - np.sin(beta) - np.pi * sigma / sigma0


def sector_angle_eq_derivative(beta):
    return 1 - np.cos(beta)


def calc_sector_angle(sigma, sigma0):
    return newtons_method(func=sector_angle_eq(sigma, sigma0), x0=2, fprime=sector_angle_eq_derivative)

def main():
    beta = calc_sector_angle()
    print(f"beta={beta}")


if __name__ == '__main__':
    main()
