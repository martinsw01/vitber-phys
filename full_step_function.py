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
    #full_ship_physics(0.01 * m, 0.1, 0.01)

    t0 = 0
    tend = 20
    w0 = np.array([0, 0.2, 0, 0, 0.2, 0, 0.5, 2])
    h = 0.001
    t_num, w_num = solve_ode(full_f(0, 0.2, 1, 0.2), t0, tend, w0, h, method=rk4)
    #x_num, y_num = solve_ode(full_f(0.01*m, 0, 0), x0, xend, y0, h, method=euler)
    plt.title("Numerical solution")
    plt.plot(t_num, w_num[:, 6])
    plt.plot(t_num, w_num[:, 7])
    plt.legend(["s_L", "v_L"])
    plt.show()
    animate_deck_movement(t_num, w_num[:, 4], w_num[:, 0], w_num[:, 1], w_num[:, 6], gjerde=True, stepsize=0.01, vis_akse_verdier=False)
