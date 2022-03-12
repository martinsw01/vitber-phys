import numpy as np
import scipy.integrate
from variables import R


def solve_ode_bdf(f, x0, xend, y0, h, method=0):
    bdf = scipy.integrate.BDF(f, x0, y0, xend, max_step=h)
    x_num = np.array([x0])
    y_num = np.array([y0])
    while bdf.status == "running":
        bdf.step()

        y_num = np.concatenate((y_num, np.array([bdf.y])))
        x_num = np.append(x_num, bdf.t)

    return x_num, y_num

