import time

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

from ode_solver import solve_ode, runge_kutta_4 as rk4
from full_step_function import full_f
from bdf_ode_solver import solve_ode_bdf
from variables import A, calc_gamma, F_G, F_B, F_f, F_w, F_Lx, F_Ly, m, I_C, tau_B, g, R, tau_L, tau_f, tau_w, omega_0
from animation import animate_deck_movement
from capsizing import has_capsized


def find_capsizing_time(x, w):
    for xn, wn in zip(x, w):
        if has_capsized(wn[:6]):
            return xn
    return 0


def compare_time_steps():
    N = 100
    h0 = 0.01
    h_array = np.linspace(h0, 0.3, N)
    t0 = 0
    tend = 300
    m_L = 0 * m
    w0 = np.array([0, 0, 0, 0, 0, 0 * np.pi / 180, 0, 0])
    capsizing_times_rk4 = np.zeros(N)
    capsizing_times_bdf = np.zeros(N)
    for ind, h in enumerate(h_array):
        # t, w = solve_ode(full_f(0.08 * m, 100, 0.625 * m * g, 0.93 * omega_0, h), t0, tend, w0, h, method=method)
        t, w = solve_ode(full_f(m_L, 100, 0.65 * m * g, omega_0, h), t0, tend, w0, h, method=rk4)
        capsizing_times_rk4[ind] = find_capsizing_time(t, w)
        t_bdf, w_bdf = solve_ode_bdf(full_f(m_L, 100, 0.65 * m * g, omega_0, h), t0, tend, w0, h, method=rk4)
        capsizing_times_bdf[ind] = find_capsizing_time(t_bdf, w_bdf)
        print(f"{(ind+1)/len(h_array)*100}% done")

    plt.figure(0)
    plt.plot(h_array, capsizing_times_rk4, label="rk4")
    plt.plot(h_array, capsizing_times_bdf, label="bdf")
    plt.legend()
    plt.show()


def compare_friction_coefficients():
    N = 10
    friction_array = np.linspace(100, 300,  N)
    h = 0.2
    t0 = 0
    tend = 300
    m_L = 0 * m
    w0 = np.array([0, 0, 0, 0, 0, 0 * np.pi / 180, 0, 0])
    capsizing_times_rk4 = np.zeros(N)
    capsizing_times_bdf = np.zeros(N)
    for ind, friction in enumerate(friction_array):
        # t, w = solve_ode(full_f(0.08 * m, 100, 0.625 * m * g, 0.93 * omega_0, h), t0, tend, w0, h, method=method)
        t, w = solve_ode(full_f(m_L, friction, 0.65 * m * g, omega_0, h), t0, tend, w0, h, method=rk4)
        capsizing_times_rk4[ind] = find_capsizing_time(t, w)
        t_bdf, w_bdf = solve_ode_bdf(full_f(m_L, friction, 0.65 * m * g, omega_0, h), t0, tend, w0, h, method=rk4)
        capsizing_times_bdf[ind] = find_capsizing_time(t_bdf, w_bdf)
        print(f"{(ind+1)/len(friction_array)*100}% done")

    plt.figure(0)
    plt.plot(friction_array, capsizing_times_rk4, label="rk4")
    plt.plot(friction_array, capsizing_times_bdf, label="bdf")
    plt.legend()
    plt.show()


def compare_wave_size():
    h_array = [0.005 , 0.01, 0.1]
    t0 = 0
    tend = 300
    m_L = 0 * m
    w0 = np.array([0, 0, 0, 0, 0, 0 * np.pi / 180, 0, 0])
    wave_force_array = np.arange(0.1, 1, 0.01)
    rk = []
    bdf = []
    for h in h_array:
        for wave_force in wave_force_array:
            t, w = solve_ode(full_f(m_L, 100, wave_force * m * g, omega_0, h), t0, tend, w0, h, method=rk4)
            if find_capsizing_time(t, w):
                rk.append(wave_force)
                break
            print(f"RK waveforce {wave_force}")
        for wave_force in wave_force_array:
            t, w = solve_ode_bdf(full_f(m_L, 100, wave_force * m * g, omega_0, h), t0, tend, w0, h, method=rk4)
            if find_capsizing_time(t, w):
                bdf.append(wave_force)
                break
            print(f"BDF waveforce {wave_force}")
        print(f"Done with h = {h}")

    print(f"Minimum wave force for capsizing:\nRK4: \ndt = {h_array[0]} : {rk[0]}\ndt = {h_array[1]} : {rk[1]}\ndt = {h_array[2]} : {rk[2]}")
    print(f"BDF:\ndt = {h_array[0]} : {bdf[0]}\ndt = {h_array[1]} : {bdf[1]}\ndt = {h_array[2]} : {bdf[2]}")


"""def find_capsizing_force(friction):
    h = 0.1
    t0 = 0
    tend = 100
    m_L = 0.0 * m
    w0 = np.array([0, 0, 0, 0, 0, 0 * np.pi / 180, 0, 0])

    l_bound = 0
    u_bound = 1
    t, w = solve_ode(full_f(0, friction, u_bound * m * g, 0.9*omega_0, h), t0, tend, w0, h, method=rk4)
    b = len(t)
    while not find_capsizing_time(t, w):
        t, w = solve_ode(full_f(0, friction, u_bound * m * g, 0.9*omega_0, h), t0, tend, w0, h, method=rk4)
        u_bound += 1
        l_bound += 1



    while abs(u_bound-l_bound)>0.01:
        t, w = solve_ode(full_f(0, friction, (u_bound + l_bound) / 2 * m * g, 0.9*omega_0, h), t0, tend, w0, h,
                         method=rk4)
        if find_capsizing_time(t, w):
            u_bound = (u_bound+l_bound)/2
        else:
            l_bound = (u_bound+l_bound)/2
    return (u_bound+l_bound)/2, b"""

def find_capsizing_force_bdf(friction):
    h = 0.1
    t0 = 0
    tend = 100
    m_L = 0.0 * m
    w0 = np.array([0, 0, 0, 0, 0, 0 * np.pi / 180, 0, 0])

    l_bound = 0
    u_bound = 1
    t, w = solve_ode_bdf(full_f(0, friction, u_bound * m * g, 0.9*omega_0, h), t0, tend, w0, h, method=rk4)
    while not find_capsizing_time(t, w):
        t, w = solve_ode_bdf(full_f(0, friction, u_bound * m * g, 0.9*omega_0, h), t0, tend, w0, h, method=rk4)
        u_bound += 0.5
        l_bound += 0.5

    while abs(u_bound - l_bound) > 0.01:
        t, w = solve_ode_bdf(full_f(0, friction, (u_bound + l_bound) / 2 * m * g, 0.9*omega_0, h), t0, tend, w0, h,
                         method=rk4)
        a = len(t)
        if find_capsizing_time(t, w):
            u_bound = (u_bound + l_bound) / 2
        else:
            l_bound = (u_bound + l_bound) / 2
    return (u_bound + l_bound) / 2, a


def find_capsizing_force(ode_solver, friction):
    h = 0.1
    t0 = 0
    tend = 100
    w0 = np.array([0, 0, 0, 0, 0, 0 * np.pi / 180, 0, 0])

    l_bound = 0
    u_bound = 1
    t, w = ode_solver(full_f(0, friction, u_bound * m * g, 0.9*omega_0, h), t0, tend, w0, h, method=rk4)
    while not find_capsizing_time(t, w):
        t, w = ode_solver(full_f(0, friction, u_bound * m * g, 0.9*omega_0, h), t0, tend, w0, h, method=rk4)
        u_bound += 0.5
        l_bound += 0.5

    while abs(u_bound - l_bound) > 0.01:
        t, w = ode_solver(full_f(0, friction, (u_bound + l_bound) / 2 * m * g, 0.9*omega_0, h), t0, tend, w0, h,
                         method=rk4)
        if find_capsizing_time(t, w):
            u_bound = (u_bound + l_bound) / 2
        else:
            l_bound = (u_bound + l_bound) / 2
    return (u_bound + l_bound) / 2


def wavesize_vs_friction():
    N = 10
    friction_array = np.linspace(10, 1000, N)

    capsizing_force_rk4 = np.zeros(N)
    capsizing_force_bdf = np.zeros(N)
    # t_0 = time.perf_counter()
    for ind, friction in enumerate(friction_array):

        # t, w = solve_ode(full_f(0.08 * m, 100, 0.625 * m * g, 0.93 * omega_0, h), t0, tend, w0, h, method=method)

        """t, w = solve_ode(full_f(m_L, friction, waveforce_rk * m * g, omega_0, h), t0, tend, w0, h, method=rk4)
        while not find_capsizing_time(t, w):
            t, w = solve_ode(full_f(m_L, friction, waveforce_rk * m * g, omega_0, h), t0, tend, w0, h, method=rk4)
            waveforce_rk += 0.01"""
        capsizing_force_rk4[ind], b = find_capsizing_force(solve_ode, friction)
        capsizing_force_bdf[ind]

        print(f"{friction / friction_array[-1] * 100}% done")
    """t_rk = time.perf_counter()-t_0
    t_0 = time.perf_counter()
    for ind, friction in enumerate(friction_array):
        t, w = solve_ode_bdf(full_f(m_L, friction, waveforce_bdf * m * g, omega_0, h), t0, tend, w0, h, method=rk4)
        while not find_capsizing_time(t, w):
            t, w = solve_ode_bdf(full_f(m_L, friction, waveforce_bdf * m * g, omega_0, h), t0, tend, w0, h, method=rk4)
            waveforce_bdf += 0.01
        capsizing_force_bdf[ind], a = find_capsizing_force_bdf(friction)
        print(f"{friction/friction_array[-1]*100}% done")"""
    t_bdf = time.perf_counter() - t_0
    print(f"RK brukte {t_rk} sekunder, BDF brukte {t_bdf} sekunder")
    print(a)
    print(b)
    plt.figure(0)
    plt.plot(friction_array, capsizing_force_rk4, label="rk4")
    plt.plot(friction_array, capsizing_force_bdf, label="bdf")
    plt.legend()
    plt.show()


"""def find_capsizing_force(frequency):
    h = 0.05
    t0 = 0
    tend = 100
    m_L = 0.0 * m
    w0 = np.array([0, 0, 0, 0, 0, 0 * np.pi / 180, 0, 0])

    l_bound = 0
    u_bound = 1
    t, w = solve_ode(full_f(0, 100, u_bound * m * g, frequency * omega_0, h), t0, tend, w0, h, method=rk4)
    while not find_capsizing_time(t, w):
        t, w = solve_ode(full_f(0, 100, u_bound * m * g, frequency * omega_0, h), t0, tend, w0, h, method=rk4)
        u_bound += 1
        l_bound +=1



    while abs(u_bound-l_bound)>0.1:
        t, w = solve_ode(full_f(0, 100, (u_bound + l_bound) / 2 * m * g, frequency * omega_0, h), t0, tend, w0, h,
                         method=rk4)
        if find_capsizing_time(t, w):
            u_bound = (u_bound+l_bound)/2
        else:
            l_bound = (u_bound+l_bound)/2
    return (u_bound+l_bound)/2


def find_capsizing_force_bdf(frequency):
    h = 0.2
    t0 = 0
    tend = 100
    m_L = 0.0 * m
    w0 = np.array([0, 0, 0, 0, 0, 0 * np.pi / 180, 0, 0])

    l_bound = 0
    u_bound = 1
    t, w = solve_ode_bdf(full_f(0, 100, u_bound * m * g, frequency * omega_0, h), t0, tend, w0, h, method=rk4)
    while not find_capsizing_time(t, w):
        t, w = solve_ode_bdf(full_f(0, 100, u_bound * m * g, frequency * omega_0, h), t0, tend, w0, h, method=rk4)
        u_bound += 0.5
        l_bound += 0.5

    while abs(u_bound - l_bound) > 0.1:
        t, w = solve_ode_bdf(full_f(0, 100, (u_bound + l_bound) / 2 * m * g, frequency * omega_0, h), t0, tend, w0, h,
                         method=rk4)
        if find_capsizing_time(t, w):
            u_bound = (u_bound + l_bound) / 2
        else:
            l_bound = (u_bound + l_bound) / 2
    return (u_bound + l_bound) / 2"""


def powerful_waves():
    N = 5
    frequency_array = np.linspace(1, 6, N)

    capsizing_force_rk4 = np.zeros(N)
    capsizing_force_bdf = np.zeros(N)

    t_0 = time.perf_counter()
    for ind, frequency in enumerate(frequency_array):
        capsizing_force_rk4[ind] = find_capsizing_force(frequency)
        print(f"{ind / len(frequency_array-1) * 100}% done")
    t_rk = time.perf_counter() - t_0
    t_0 = time.perf_counter()
    for ind, frequency in enumerate(frequency_array):
        capsizing_force_bdf[ind] = find_capsizing_force_bdf(frequency)
        print(f"{ind / len(frequency_array-1) * 100}% done")
    t_bdf = time.perf_counter() - t_0

    plt.figure(0)
    plt.plot(frequency_array, capsizing_force_rk4, label="rk4")
    plt.plot(frequency_array, capsizing_force_bdf, label="bdf")
    plt.legend()
    plt.show()
    print(f"RK brukte {t_rk} sekunder, BDF brukte {t_bdf} sekunder")





if __name__ == "__main__":
    # compare_time_steps()
    # compare_wave_size() # denne tok lang tid men har lagret resultatet
    # powerful_waves()
    # compare_friction_coefficients()
    wavesize_vs_friction()