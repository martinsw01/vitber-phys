import numpy as np


def analytic_solution(F_B, h, I, theta0):
    w = np.sqrt(F_B * h / I)

    def theta(t):
        return theta0 * np.cos(w * t)

    return theta
