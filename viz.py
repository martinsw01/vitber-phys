import matplotlib.pyplot as plt
import numpy as np


def plot_difference(ax, h, difference, **plot_args):
    ax.plot(h, difference, marker=".", **plot_args)


def plot_differences(h, euler, rk4):
    _, ax = plt.subplots(1)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("h")

    plot_difference(ax, h, euler, label="euler")
    plot_difference(ax, h, rk4, label="rk4")

    ax.legend()
    plt.show()


def plot_theta(ax, t, theta, **plot_args):
    ax.plot(t, theta, **plot_args)


def plot_omega(ax, t, omega, **plot_args):
    ax.plot(t, omega, **plot_args)


def plot_center_of_gravity(ax, t, x, y):
    ax.plot(t, x, label="y_C")
    ax.plot(t, y, label="x_C")


def plot_vel(ax, t, vx, vy):
    ax.plot(t, vx, label="vx")
    ax.plot(t, vy, label="vy")


def plot_states(t, x, y, vx, vy, theta, omega):
    _, (ax1, ax2, ax3) = plt.subplots(3)

    plot_theta(ax1, t, theta, label="theta")
    plot_omega(ax1, t, omega, label="omega")
    plot_center_of_gravity(ax2, t, x, y)
    plot_vel(ax3, t, vx, vy)

    for ax in (ax1, ax2, ax3):
        ax.legend()

    plt.tight_layout()
    plt.show()


def plot_cargo(t, x_C, y_C, v_xC, v_yC, theta, omega, s_L, v_L):
    # _, (ax1, ax2, ax3, ax4) = plt.subplots(4)

    plot_states(t, x_C, y_C, v_xC, v_yC, theta, omega)
    _, ax = plt.subplots(1)
    ax.plot(t, s_L, label="s_L")
    ax.plot(t, v_L, label="v_L")
    ax.legend()
    plt.show()

