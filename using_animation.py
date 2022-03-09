import numpy as np
from matplotlib import pyplot as plt

from animation import animate_deck_movement
from ode_solver import solve_ode, runge_kutta_4
from secured_cargo import secured_cargo_f
from unsecured_cargo import unsecured_cargo_f
from variables import m, R, y_C0


def animate_unsecured_cargo():
    t0 = x_C0 = vx0 = vy0 = omega0 = v_L0 = 0
    s_L0 = - R * 0.8
    theta0 = 20 / 180 * np.pi
    tend = 20
    m_L = 0.001 * m

    w0 = np.array([x_C0, y_C0, vx0, vy0, theta0, omega0, s_L0, v_L0])

    t, w = solve_ode(f=unsecured_cargo_f(m_L), x0=t0, xend=tend, y0=w0, h=0.01, method=runge_kutta_4)
    x_C, y_C, vx, vy, theta, omega, s_L, v_L = w.T
    animate_deck_movement(t, theta, x_C, y_C, s_L)


def animate_secured_cargo():
    t0 = x_C0 = vx0 = vy0 = omega0 = v_L0 = 0
    s_L0 = -R * 0.8
    theta0 = 20 / 180 * np.pi
    tend = 20
    m_L = 0.001 * m
    beta = calc_beta(m_L)

    w0 = np.array([x_C0, y_C0(beta), vx0, vy0, theta0, omega0, s_L0, v_L0])

    t, w = solve_ode(f=secured_cargo_f(m_L, 0.001), x0=t0, xend=tend, y0=w0, h=0.001, method=runge_kutta_4)
    x_C, y_C, vx, vy, theta, omega, s_L, v_L = w.T
    plt.plot(t, s_L, label="s_L")
    plt.plot(t, v_L, label="v_L")
    animate_deck_movement(t, theta, x_C, y_C, s_L, gjerde=True)


def main():
    animate_secured_cargo()


if __name__ == "__main__":
    main()
