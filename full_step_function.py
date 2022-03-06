import numpy as np
import matplotlib.pyplot as plt

from ode_solver import solve_ode, f, euler, runge_kutta_4 as rk4
from variables import R, A, calc_gamma, F_G, F_B, F_f, F_w, F_Lx, F_Ly, m, I_C, tau_B, g, R, y_C0, tau_L, tau_f, tau_w
from animation import animate_deck_movement

def full_f(m_L, k_f, F_w0, omega_w):
    def f(t, w):
        x_C, y_C, v_xC, v_yC, theta, omega, s_L, v_L = w
        area = A(theta, y_C)
        gamma = calc_gamma(theta, y_C)
        force_x = F_f(omega, gamma, k_f) + F_w(t, F_w0, omega_w)
        force_y = F_G + F_B(area)
        torque = tau_B(theta, area) + tau_f(omega, y_C, gamma, k_f) + tau_w(t, y_C, F_w0, omega_w)
        a_L = -np.sin(theta) * g
        a_xC = force_x / m
        a_yC = force_y / m
        alpha = torque / I_C
        if np.abs(s_L) >= R and v_L * np.sign(s_L) > 0:
            v_L = 0
        else:
            force_x += F_Lx(theta, m_L)
            force_y += F_Ly(theta, m_L)
            torque += tau_L(m_L, s_L)
        return np.array([v_xC, v_yC, a_xC, a_yC, omega, alpha, v_L, a_L])
    return f

if __name__ == '__main__':
    main()
