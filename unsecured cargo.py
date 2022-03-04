import numpy as np

from ode_solver import solve_ode, runge_kutta_4 as rk4
from variables import A, F_G, F_B, m, I_C, tau_B, g, R, y_C0
from viz import plot_cargo


def tau_L(m_L, s_L):
    return m_L * g * s_L


def unsecured_cargo_f(m_L):
    def f(t, w):
        x_C, y_C, v_xC, v_yC, theta, omega, s_L, v_L = w
        area = A(theta, y_C)
        a_yC = (F_G + F_B(area)) / m
        a_xC = 0
        if np.abs(s_L) < R:
            a_L = np.cos(theta) * g
            a_yC += a_L * s_L/R
            alpha = (tau_B(theta, area) + tau_L(m_L, s_L)) / I_C
            return np.array([v_xC, v_yC, a_xC, a_yC, omega, alpha, v_L, a_L])
        else:
            alpha = tau_B(theta, area) / I_C
            return np.array([v_xC, v_yC, a_xC, a_yC, omega, alpha, 0, 0])
    return f


def unsecured_cargo(m_L):
    t0 = x_C0 = vx0 = vy0 = omega0 = s_L0 = v_L0 = 0
    theta0 = 20 / 180 * np.pi
    tend = 20

    w0 = np.array([x_C0, y_C0, vx0, vy0, theta0, omega0, s_L0, v_L0])

    t, w = solve_ode(f=unsecured_cargo_f(m_L), x0=t0, xend=tend, y0=w0, h=0.01, method=rk4)

    plot_cargo(t, *w.T)


if __name__ == '__main__':
    unsecured_cargo(0.008*m)
    unsecured_cargo(0.001*m)

