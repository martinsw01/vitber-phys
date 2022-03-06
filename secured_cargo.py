import numpy as np

from ode_solver import solve_ode, runge_kutta_4 as rk4
from variables import A, F_G, F_B, m, I_C, tau_B, g, R, y_C0
from viz import plot_cargo
from unsecured_cargo import tau_L


def secured_cargo_f(m_L, dt):
    def f(t, w):
        x_C, y_C, v_xC, v_yC, theta, omega, s_L, v_L = w
        area = A(theta, y_C)
        a_yC = (F_G + F_B(area)) / m
        a_xC = 0
        a_L = -np.sin(theta) * g
        a_yC -= a_L * m_L / m
        alpha = (tau_B(theta, area) + tau_L(m_L, s_L)) / I_C
        if np.abs(s_L) >= R and v_L * s_L > 0:
            a_L = -v_L / dt
            v_L = 0
        return np.array([v_xC, v_yC, a_xC, a_yC, omega, alpha, v_L, a_L])

    return f

