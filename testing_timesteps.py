import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

from full_step_function import full_f
from ode_solver import solve_ode
from ode_solver import runge_kutta_4 as rk4
from variables import A, calc_gamma, F_G, F_B, F_f, F_w, F_Lx, F_Ly, m, I_C, tau_B, g, R, tau_L, tau_f, tau_w, omega_0


def sup_norm(t1, t2, w1, w2):

    f = interpolate.interp1d(t1, w1.T)
    return np.linalg.norm(f(t2[:-1]).T-w2[:-1], ord=np.inf)


def l2_norm(t1, t2, w1, w2):
    f = interpolate.interp1d(t1, w1.T)
    return np.linalg.norm(f(t2[:-1]).T - w2[:-1])/len(t2)


def test_time_steps(method=rk4):
    N = 100
    t0, tend = 0, 200
    h0 = 0.01
    # h_array = h0*2**np.arange(0, N, 1)
    h_array = np.linspace(h0, 0.3, N)
    w0 = np.array([0, 0, 0, 0, 0, np.pi/8, 0, 0])
    l2_array = np.zeros(N)
    sup_array = np.zeros(N)
    t_h0, w_h0 = solve_ode(full_f(0, 0, 0, 0, h0), t0, tend, w0, h0, method=method)

    for ind, h in enumerate(h_array[1:]):
        t, w = solve_ode(full_f(0, 0, 0, 0, h), t0, tend, w0, h, method=method)
        l2_array[ind] = l2_norm(t_h0, t, w_h0, w)
        sup_array[ind] = sup_norm(t_h0, t, w_h0, w)
        print(h)
    plt.figure(1)
    fig, ax1 = plt.subplots()
    ax1.plot(h_array[1:], l2_array[1:], label=r"$l_2-norm$")
    ax2 = plt.twinx(ax1)
    ax2.plot(h_array[1:], sup_array[1:], label=r"$sup-norm$", color="tab:red")
    fig.tight_layout()
    plt.title(r"l_2-norm vs sup-norm")
    plt.xlabel("h [s]")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    test_time_steps()
