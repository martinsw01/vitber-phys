import numpy as np

from analytic_approximation import analytic_solution
from ode_solver import solve_ode, euler, runge_kutta_4, f
from variables import FB, H, IC, A0
from viz import plot_differences


def compare_methods(t, f, t0, y0, h_array):
    y_analytic = analytic_solution(FB(A0), H, IC, y0)(t)

    euler_diff_array = np.array([calc_difference_at_t(t, euler, f, t0, y0, h, y_analytic) for h in h_array])
    rk4_diff_array = np.array([calc_difference_at_t(t, runge_kutta_4, f, t0, y0, h, y_analytic) for h in h_array])

    return euler_diff_array, rk4_diff_array


def calc_difference_at_t(t, method, f, t0, theta0, h, y_analytic):
    _, y = solve_ode(f, t0, t, theta0, h, method=method)
    return np.linalg.norm(y[-1] - y_analytic)


def main():
    h_array = np.exp(np.linspace(*np.log([0.001, 0.5])))  # Evenly spaced on a log-scaled x-axis

    theta0, w0 = 0.1, 0
    y0 = np.array([theta0, w0])
    euler_diff_array, rk4_diff_array = compare_methods(t=20, f=f, t0=0, y0=y0, h_array=h_array)
    plot_differences(h_array, euler_diff_array, rk4_diff_array)


if __name__ == '__main__':
    main()
