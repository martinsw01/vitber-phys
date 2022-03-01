import numpy as np
from scipy.optimize import newton as newtons_method

from variables import sigma0, sigma


# Sector angle equation; f(beta) = 0
def f(beta):
    return beta - np.sin(beta) - np.pi * sigma / sigma0


# Derivative of sector angle equation
def fprime(beta):
    return 1 - np.cos(beta)


def calc_sector_angle():
    return newtons_method(func=f, x0=2, fprime=fprime)


def main():
    beta = newtons_method(func=f, x0=2, fprime=fprime)
    print(f"beta={beta}")


if __name__ == '__main__':
    main()
