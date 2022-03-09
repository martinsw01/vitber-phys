import numpy as np

from ode_solver import solve_ode, runge_kutta_4 as rk4
from variables import F_B, F_G, m, I_C, A, tau_B, y_C0
from viz import plot_states


# 2 a)
def f(t, w):
    x_C, y_C, vx, vy, theta, omega = w
    area = A(theta, y_C)
    ay = (F_G + F_B(area)) / m
    ax = 0
    alpha = tau_B(theta, area) / I_C
    return np.array([vx, vy, ax, ay, omega, alpha])


# 2 b)
def main():
    t0 = x_C0 = vx0 = vy0 = omega0 = 0
    theta0 = 20/180 * np.pi
    w0 = np.array([x_C0, y_C0(), vx0, vy0, theta0, omega0])

    t, w = solve_ode(f, x0=t0, xend=20, y0=w0, h=0.01, method=rk4)

    plot_states(t, *w.T)


if __name__ == '__main__':
    main()
