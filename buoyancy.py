import numpy as np

from ode_solver import solve_ode, runge_kutta_4 as rk4
from variables import F_B, F_G, m, H, A0, I_C, A
from viz import plot_states


# 2 a)
def f(t, w):
    x, y, vx, vy, theta, omega = w
    area = A(theta, y)
    ay = (F_G + F_B(area)) / m
    ax = 0
    alpha = - F_B(area) * H * np.sin(theta) / I_C
    return np.array([vx, vy, ax, ay, omega, alpha])



def main():
    x0 = y0 = vx0 = vy0 = omega0 = 0
    theta0 = 0.3
    w0 = np.array([x0, y0, vx0, vy0, theta0, omega0])

    t, w = solve_ode(f, x0=0, xend=20, y0=w0, h=0.01, method=rk4)

    plot_states(t, *w.T)


if __name__ == '__main__':
    main()
