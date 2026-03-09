"""
solver.py - Homotopy regressor for nonlinear ODEs.

Solves:
    1st order:  y' + f(y) = u(t)
    2nd order:  y'' + f(y, y') = u(t)

Using 3-point backward discrete derivatives and 3-term homotopy series.
No matrices, no iteration. Suitable for microcontroller.

Author: Rodolfo H. Rodrigo - UNSJ
"""

import numpy as np


def solve_order1(f, df, d2f, d3f, u, y0, y1, T, n):
    """
    Solve y' + f(y) = u(t) using 3-point homotopy regressor.

    Parameters
    ----------
    f    : callable f(y) -> float
    df   : callable f'(y) -> float
    d2f  : callable f''(y) -> float
    d3f  : callable f'''(y) -> float
    u    : array of length n, excitation u(t_k)
    y0   : float, y(t_0) initial condition
    y1   : float, y(t_1) initial condition
    T    : float, sampling period
    n    : int, number of points

    Returns
    -------
    y : array of length n
    """
    y = np.zeros(n)
    y[0] = y0
    y[1] = y1

    for k in range(2, n):
        y[k] = y[k-1]

        # g = y'_k discretized + f(y_k) - u_k
        g = (3/2)*y[k]/T - 2*y[k-1]/T + (1/2)*y[k-2]/T + f(y[k]) - u[k]
        gp = 3/(2*T) + df(y[k])

        # z1: Newton step
        y[k] = y[k] - g / gp

        # z2: second order correction
        g = (3/2)*y[k]/T - 2*y[k-1]/T + (1/2)*y[k-2]/T + f(y[k]) - u[k]
        gp = 3/(2*T) + df(y[k])
        gpp = d2f(y[k])
        y[k] = y[k] - (1/2) * g**2 * gpp / gp**3

        # z3: third order correction
        g = (3/2)*y[k]/T - 2*y[k-1]/T + (1/2)*y[k-2]/T + f(y[k]) - u[k]
        gp = 3/(2*T) + df(y[k])
        gpp = d2f(y[k])
        gppp = d3f(y[k])
        y[k] = y[k] - (1/6) * g**3 * (-gppp * gp + 3 * gpp**2) / gp**5

    return y


def solve_order2(f, df_dy, df_dyp, d2f_dy2, d2f_dydyp, d2f_dyp2,
                 d3f_dy3, u, y0, y1, T, n):
    """
    Solve y'' + f(y, y') = u(t) using 3-point homotopy regressor.

    Discrete derivatives (3 points):
        y''_k = (y_k - 2*y_{k-1} + y_{k-2}) / T^2
        y'_k  = (3*y_k - 4*y_{k-1} + y_{k-2}) / (2*T)

    Parameters
    ----------
    f          : callable f(y, yp) -> float
    df_dy      : callable df/dy(y, yp) -> float
    df_dyp     : callable df/dy'(y, yp) -> float
    d2f_dy2    : callable d²f/dy²(y, yp) -> float
    d2f_dydyp  : callable d²f/dydyp(y, yp) -> float
    d2f_dyp2   : callable d²f/dyp²(y, yp) -> float
    d3f_dy3    : callable d³f/dy³(y, yp) -> float  (can be None -> 0)
    u          : array of length n
    y0, y1     : initial conditions y(t_0), y(t_1)
    T          : sampling period
    n          : number of points

    Returns
    -------
    y : array of length n
    """
    y = np.zeros(n)
    y[0] = y0
    y[1] = y1

    for k in range(2, n):
        y[k] = y[k-1]

        # y'_k approximation
        yp_k = (3*y[k] - 4*y[k-1] + y[k-2]) / (2*T)

        # g = y''_k + f(y_k, y'_k) - u_k
        g = (y[k] - 2*y[k-1] + y[k-2])/T**2 + f(y[k], yp_k) - u[k]

        # g' = dg/dy_k = 1/T² + df/dy + df/dy' * (3/(2T))
        gp = 1/T**2 + df_dy(y[k], yp_k) + df_dyp(y[k], yp_k) * 3/(2*T)

        # z1
        y[k] = y[k] - g / gp

        # Recalculate
        yp_k = (3*y[k] - 4*y[k-1] + y[k-2]) / (2*T)
        g = (y[k] - 2*y[k-1] + y[k-2])/T**2 + f(y[k], yp_k) - u[k]
        gp = 1/T**2 + df_dy(y[k], yp_k) + df_dyp(y[k], yp_k) * 3/(2*T)

        # g'' = d²g/dy_k²
        gpp = (d2f_dy2(y[k], yp_k)
               + 2 * d2f_dydyp(y[k], yp_k) * 3/(2*T)
               + d2f_dyp2(y[k], yp_k) * (3/(2*T))**2)

        # z2
        y[k] = y[k] - (1/2) * g**2 * gpp / gp**3

        # Recalculate
        yp_k = (3*y[k] - 4*y[k-1] + y[k-2]) / (2*T)
        g = (y[k] - 2*y[k-1] + y[k-2])/T**2 + f(y[k], yp_k) - u[k]
        gp = 1/T**2 + df_dy(y[k], yp_k) + df_dyp(y[k], yp_k) * 3/(2*T)
        gpp = (d2f_dy2(y[k], yp_k)
               + 2 * d2f_dydyp(y[k], yp_k) * 3/(2*T)
               + d2f_dyp2(y[k], yp_k) * (3/(2*T))**2)

        # g'''
        if d3f_dy3 is not None:
            gppp = d3f_dy3(y[k], yp_k)
        else:
            gppp = 0.0

        # z3
        y[k] = y[k] - (1/6) * g**3 * (-gppp * gp + 3 * gpp**2) / gp**5

    return y


def solve_order1_numeric(f, u, y0, y1, T, n, h=1e-5):
    """
    Solve y' + f(y) = u(t) with numerical derivatives of f.

    For unknown f (RBF, polynomial, lookup table, etc.)

    Parameters
    ----------
    f  : callable f(y) -> float (the only thing you need)
    u  : array of length n
    y0, y1 : initial conditions
    T  : sampling period
    n  : number of points
    h  : step for numerical derivatives

    Returns
    -------
    y : array of length n
    """
    def df(y):
        return (8*(f(y+h) - f(y-h)) - (f(y+2*h) - f(y-2*h))) / (12*h)

    def d2f(y):
        return (-f(y+2*h) + 16*f(y+h) - 30*f(y) + 16*f(y-h) - f(y-2*h)) / (12*h**2)

    def d3f(y):
        return (-f(y+2*h) + 2*f(y+h) - 2*f(y-h) + f(y-2*h)) / (2*h**3)

    return solve_order1(f, df, d2f, d3f, u, y0, y1, T, n)
