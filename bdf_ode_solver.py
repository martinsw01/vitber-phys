import numpy as np
import scipy.integrate
from variables import R


def solve_ode_bdf(f, x0, xend, y0, h, method=0):
    bdf = scipy.integrate.BDF(f, x0, y0, xend, max_step=h)
    x_num = np.array([x0])
    y_num = np.array([y0])
    while bdf.status == "running":
        bdf.step()
        yn = bdf.y
        xn = bdf.t
        if len(yn) > 6:
            if np.abs(yn[6]) >= R and yn[6] * yn[7] > 0:
                yn[6] = R * np.sign(yn[6])
                yn[7] = 0
        y_num = np.concatenate((y_num, np.array([yn])))
        x_num = np.append(x_num, xn)

    return x_num, y_num

