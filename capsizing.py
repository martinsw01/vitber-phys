import numpy as np

from buoyancy import f
from ode_solver import solve_ode, runge_kutta_4 as rk4
from variables import calc_gamma, y_C0
from viz import plot_states


def has_capsized(w):
    _, y_C, _, _, theta, _ = w
    return abs(theta) > (np.pi - calc_gamma(theta, y_C)) / 2


def solve_and_set_capsized(t0, tend, w0, h):
    t, w = solve_ode(f, x0=t0, xend=tend, y0=w0, h=h, method=rk4)
    theta = w.T[4]

    capsize_index = np.argmax(has_capsized(w.T))
    if capsize_index:  # i.e if not 0; argmax returns 0 if all results are False
        w[capsize_index:] = [0, 0, 0, 0, np.sign(theta[capsize_index]) * np.pi/2, 0]

    return t, w


def capsizing_ship():
    t0 = x_C0 = vx0 = vy0 = theta0 = 0
    omega0 = 0.4344827586206897
    tend = 20

    w0 = np.array([x_C0, y_C0(), vx0, vy0, theta0, omega0])

    t, w = solve_and_set_capsized(t0, tend, w0, h=0.01)

    plot_states(t, *w.T)


def find_minimum_capsizing_omega0():
    t0 = x_C0 = vx0 = vy0 = theta0 = 0
    tend = 20
    omega0_array = np.linspace(0.4, 0.5, 30)

    results = (solve_ode(f, x0=t0, xend=tend, y0=np.array([x_C0, y_C0(), vx0, vy0, theta0, omega0]), h=0.01, method=rk4)
               for omega0 in omega0_array)
    capsizing_omega0 = (omega0 for omega0 in omega0_array
                        if has_capsized(next(results)[1].T).any())

    min_capsizing_omega0 = next(capsizing_omega0)
    return min_capsizing_omega0


if __name__ == '__main__':
    print(f"Min omega0 making ship capsize: {find_minimum_capsizing_omega0()}")
    capsizing_ship()
