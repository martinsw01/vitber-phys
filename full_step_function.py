import matplotlib.pyplot as plt
import numpy as np

from ode_solver import solve_ode, runge_kutta_4 as rk4
from variables import A, calc_gamma, F_G, F_B, F_f, F_w, F_Lx, F_Ly, m, I_C, tau_B, g, R, tau_L, tau_f, tau_w, omega_0
from animation import animate_deck_movement
from capsizing import has_capsized


def full_f(m_L, k_f, F_w0, omega_w, dt):
    def f(t, w):
        x_C, y_C, v_xC, v_yC, theta, omega, s_L, v_L = w
        a_L = -np.sin(theta) * g

        area = A(theta, y_C)
        gamma = calc_gamma(theta, y_C)
        force_x = F_f(omega, gamma, k_f) + F_w(t, F_w0, omega_w)
        force_y = F_G + F_B(area)
        torque = tau_B(theta, area) + tau_f(omega, y_C, gamma, k_f) + tau_w(t, y_C, F_w0, omega_w)

        if np.abs(s_L) >= R and v_L * s_L > 0:
            a_L += -v_L / dt
            v_L = 0
        else:
            force_x += F_Lx(theta, m_L)
            force_y += F_Ly(theta, m_L)
            torque += tau_L(m_L, s_L)

        a_xC = force_x / m
        a_yC = force_y / m
        alpha = torque / I_C

        return np.array([v_xC, v_yC, a_xC, a_yC, omega, alpha, v_L, a_L])

    return f


def test_multiple_friction_coefficients():
    t0 = 0
    tend = 20
    w0 = np.array([0, 0, 0, 0, 0, 0.4, 0, 0])
    h = 0.001
    t_num1, w_num1 = solve_ode(full_f(0, 0, 0, 0, h), t0, tend, w0, h, method=rk4)
    t_num2, w_num2 = solve_ode(full_f(0, 50, 0, 0, h), t0, tend, w0, h, method=rk4)
    t_num3, w_num3 = solve_ode(full_f(0, 200, 0, 0, h), t0, tend, w0, h, method=rk4)
    plt.title("Numerical solution")
    plt.plot(t_num1, w_num1[:, 4])
    plt.plot(t_num2, w_num2[:, 4])
    plt.plot(t_num3, w_num3[:, 4])
    plt.legend(["0", "50", "200"])
    plt.show()


def plot_effect_of_harmonic_occilation():
    t0 = 0
    tend = 240
    w0 = np.array([0, 0, 0, 0, 0, 2 * np.pi / 180, 0, 0])
    h = 0.01
    t_num, w_num = solve_ode(full_f(0, 100, 0.625, 0.93 * omega_0, h), t0, tend, w0, h, method=rk4)
    plt.title("Numerical solution")
    plt.plot(t_num, w_num[:, 4])

    plt.show()

def capsizing_angle(theta, y_C):
    return (np.pi - calc_gamma(theta, y_C)) / 2

def main():
    # 2f)
    #test_multiple_friction_coefficients()

    # 2g)
    #plot_effect_of_harmonic_occilation()



    t0 = 0
    tend = 50
    w0 = np.array([0, 0, 0, 0, 0, 2 * np.pi / 180, 0, 0])
    h1 = 0.01
    h2 = 0.001
    t_num1, w_num1 = solve_ode(full_f(0.08*m, 100, 0.625 * m * g, 0.93 * omega_0, h1), t0, tend, w0, h1, method=rk4)
    t_num2, w_num2 = solve_ode(full_f(0.08*m, 100, 0.625 * m * g, 0.93 * omega_0, h2), t0, tend, w0, h2, method=rk4)
    cap_angle1 = capsizing_angle(w_num1[:, 4], w_num1[:, 1])
    cap_angle2 = capsizing_angle(w_num2[:, 4], w_num2[:, 1])
    plt.plot(t_num1, w_num1[:, 4])
    plt.plot(t_num2, w_num2[:, 4])
    plt.plot(t_num1, cap_angle1)
    plt.plot(t_num2, cap_angle2)

    animate_deck_movement(t_num1, w_num1[:, 4], w_num1[:, 0], w_num1[:, 1], w_num1[:, 6], gjerde=False, stepsize=0.01, vis_akse_verdier=False)
    animate_deck_movement(t_num2, w_num2[:, 4], w_num2[:, 0], w_num2[:, 1], w_num2[:, 6], gjerde=False, stepsize=0.01, vis_akse_verdier=False)

    plt.show()


if __name__ == '__main__':
    main()
