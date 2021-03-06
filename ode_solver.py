import matplotlib.pyplot as plt
import numpy as np

from variables import A0, tau_B, I_C


def f(x, y):
    theta, w = y
    tau = tau_B(theta, A0)
    return np.array([w, tau / I_C])


def euler(f, x, y, h):
    y_next = y + h * f(x, y)
    x_next = x + h
    return x_next, y_next


def runge_kutta_4(f, x, y, h):
    k1 = f(x, y)
    k2 = f(x + h / 2, y + h / 2 * k1)
    k3 = f(x + h / 2, y + h / 2 * k2)
    k4 = f(x + h, y + h * k3)

    y_next = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    x_next = x + h
    return x_next, y_next


def solve_ode(f, x0, xend, y0, h, method=euler):
    # Initializing:
    y_num = np.array([y0])  # Array for the solution y
    x_num = np.array([x0])  # Array for the x-values

    xn = x0  # Running values for x and y
    yn = y0

    # Main loop
    while xn < xend - 1.e-10:  # Buffer for truncation errors
        xn, yn = method(f, xn, yn, h)  # Do one step by the method of choice

        # Extend the arrays for x and y
        y_num = np.concatenate((y_num, np.array([yn])))
        x_num = np.append(x_num, xn)

    return x_num, y_num


def main():
    x0 = 0
    xend = 20
    y0 = np.array([20 / 180 * np.pi, 0])
    h = 0.001
    x_num, y_num = solve_ode(f, x0, xend, y0, h, method=euler)
    plt.title("Numerical solution")
    plt.plot(x_num, y_num[:, 0], 'r')
    plt.plot(x_num, y_num[:, 1], 'b')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(['theta', 'w'])
    plt.show()

if __name__ == '__main__':
    main()
