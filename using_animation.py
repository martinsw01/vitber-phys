from animation import animate_deck_movement
import numpy as np
from ode_solver import solve_ode, runge_kutta_4
from unsecured_cargo import unsecured_cargo_f
from variables import A, F_G, F_B, m, I_C, tau_B, g, R, y_C0

def animate_unscured_cargo():
    t0 = x_C0 = vx0 = vy0 = omega0 = s_L0 = v_L0 = 0
    theta0 = 20 / 180 * np.pi
    tend = 20
    m_L = 0.001*m

    w0 = np.array([x_C0, y_C0, vx0, vy0, theta0, omega0, s_L0, v_L0])

    t, w = solve_ode(f=unsecured_cargo_f(m_L), x0=t0, xend=tend, y0=w0, h=0.01, method=runge_kutta_4)
    x_C, y_C, vx, vy, theta, omega, s_L, v_L = w.T
    print(s_L)
    animate_deck_movement(t, theta, x_C, y_C, s_L)

def main():
    animate_unscured_cargo()

if __name__=="__main__":
    main()