"""
Benchmark COMPLETO: 3 puntos vs 4 puntos

Configuraciones a comparar (8 total):
- 3 puntos: 0i-2p-3pt, 0i-3p-3pt, 1i-2p-3pt, 1i-3p-3pt
- 4 puntos: 0i-2p-4pt, 0i-3p-4pt, 1i-2p-4pt, 1i-3p-4pt
"""

import numpy as np
from scipy.integrate import odeint, solve_ivp
from scipy.interpolate import Rbf
from numpy.polynomial import Polynomial


def solve_order1_configurable(f, df, d2f, d3f, u, y0, y1, y2, T, n, n_terms=3, n_iterations=1, n_points=3):
    """
    Solve y' + f(y) = u(t) con configuración flexible.

    Parameters
    ----------
    n_points : int (3 or 4)
        Número de puntos backward para diferencias finitas.
    """
    y = np.zeros(n)
    y[0] = y0
    y[1] = y1

    if n_points == 4:
        y[2] = y2
        start_k = 3
    else:
        start_k = 2

    for k in range(start_k, n):
        y[k] = y[k-1]

        if n_points == 3:
            # 3 puntos: y'_k = (3*y_k - 4*y_{k-1} + y_{k-2}) / (2T)
            if n_iterations == 0:
                g = (3/2)*y[k]/T - 2*y[k-1]/T + (1/2)*y[k-2]/T + f(y[k]) - u[k]
                gp = 3/(2*T) + df(y[k])
                gpp = d2f(y[k])
                gppp = d3f(y[k])

                delta = - g / gp
                if n_terms >= 2:
                    delta += - (1/2) * g**2 * gpp / gp**3
                if n_terms >= 3:
                    delta += - (1/6) * g**3 * (-gppp * gp + 3 * gpp**2) / gp**5
                y[k] = y[k] + delta

            else:
                g = (3/2)*y[k]/T - 2*y[k-1]/T + (1/2)*y[k-2]/T + f(y[k]) - u[k]
                gp = 3/(2*T) + df(y[k])
                y[k] = y[k] - g / gp

                if n_terms >= 2:
                    g = (3/2)*y[k]/T - 2*y[k-1]/T + (1/2)*y[k-2]/T + f(y[k]) - u[k]
                    gp = 3/(2*T) + df(y[k])
                    gpp = d2f(y[k])
                    y[k] = y[k] - (1/2) * g**2 * gpp / gp**3

                if n_terms >= 3:
                    g = (3/2)*y[k]/T - 2*y[k-1]/T + (1/2)*y[k-2]/T + f(y[k]) - u[k]
                    gp = 3/(2*T) + df(y[k])
                    gpp = d2f(y[k])
                    gppp = d3f(y[k])
                    y[k] = y[k] - (1/6) * g**3 * (-gppp * gp + 3 * gpp**2) / gp**5

        else:  # n_points == 4
            # 4 puntos: y'_k = (11*y_k - 18*y_{k-1} + 9*y_{k-2} - 2*y_{k-3}) / (6T)
            if n_iterations == 0:
                g = (11/6)*y[k]/T - 3*y[k-1]/T + (3/2)*y[k-2]/T - (1/3)*y[k-3]/T + f(y[k]) - u[k]
                gp = 11/(6*T) + df(y[k])
                gpp = d2f(y[k])
                gppp = d3f(y[k])

                delta = - g / gp
                if n_terms >= 2:
                    delta += - (1/2) * g**2 * gpp / gp**3
                if n_terms >= 3:
                    delta += - (1/6) * g**3 * (-gppp * gp + 3 * gpp**2) / gp**5
                y[k] = y[k] + delta

            else:
                g = (11/6)*y[k]/T - 3*y[k-1]/T + (3/2)*y[k-2]/T - (1/3)*y[k-3]/T + f(y[k]) - u[k]
                gp = 11/(6*T) + df(y[k])
                y[k] = y[k] - g / gp

                if n_terms >= 2:
                    g = (11/6)*y[k]/T - 3*y[k-1]/T + (3/2)*y[k-2]/T - (1/3)*y[k-3]/T + f(y[k]) - u[k]
                    gp = 11/(6*T) + df(y[k])
                    gpp = d2f(y[k])
                    y[k] = y[k] - (1/2) * g**2 * gpp / gp**3

                if n_terms >= 3:
                    g = (11/6)*y[k]/T - 3*y[k-1]/T + (3/2)*y[k-2]/T - (1/3)*y[k-3]/T + f(y[k]) - u[k]
                    gp = 11/(6*T) + df(y[k])
                    gpp = d2f(y[k])
                    gppp = d3f(y[k])
                    y[k] = y[k] - (1/6) * g**3 * (-gppp * gp + 3 * gpp**2) / gp**5

    return y


def solve_order2_configurable(f, df_dy, df_dyp, d2f_dy2, d2f_dydyp, d2f_dyp2,
                                d3f_dy3, u, y0, y1, y2, T, n, n_terms=3, n_iterations=1, n_points=3):
    """Solve y'' + f(y, y') = u(t) con configuración flexible."""
    y = np.zeros(n)
    y[0] = y0
    y[1] = y1

    if n_points == 4:
        y[2] = y2
        start_k = 3
    else:
        start_k = 2

    for k in range(start_k, n):
        y[k] = y[k-1]

        if n_points == 3:
            # 3 puntos
            if n_iterations == 0:
                yp_k = (3*y[k] - 4*y[k-1] + y[k-2]) / (2*T)
                g = (y[k] - 2*y[k-1] + y[k-2])/T**2 + f(y[k], yp_k) - u[k]
                gp = 1/T**2 + df_dy(y[k], yp_k) + df_dyp(y[k], yp_k) * 3/(2*T)
                gpp = (d2f_dy2(y[k], yp_k)
                       + 2 * d2f_dydyp(y[k], yp_k) * 3/(2*T)
                       + d2f_dyp2(y[k], yp_k) * (3/(2*T))**2)
                gppp = d3f_dy3(y[k], yp_k) if d3f_dy3 is not None else 0.0

                delta = - g / gp
                if n_terms >= 2:
                    delta += - (1/2) * g**2 * gpp / gp**3
                if n_terms >= 3:
                    delta += - (1/6) * g**3 * (-gppp * gp + 3 * gpp**2) / gp**5
                y[k] = y[k] + delta

            else:
                yp_k = (3*y[k] - 4*y[k-1] + y[k-2]) / (2*T)
                g = (y[k] - 2*y[k-1] + y[k-2])/T**2 + f(y[k], yp_k) - u[k]
                gp = 1/T**2 + df_dy(y[k], yp_k) + df_dyp(y[k], yp_k) * 3/(2*T)
                y[k] = y[k] - g / gp

                if n_terms >= 2:
                    yp_k = (3*y[k] - 4*y[k-1] + y[k-2]) / (2*T)
                    g = (y[k] - 2*y[k-1] + y[k-2])/T**2 + f(y[k], yp_k) - u[k]
                    gp = 1/T**2 + df_dy(y[k], yp_k) + df_dyp(y[k], yp_k) * 3/(2*T)
                    gpp = (d2f_dy2(y[k], yp_k)
                           + 2 * d2f_dydyp(y[k], yp_k) * 3/(2*T)
                           + d2f_dyp2(y[k], yp_k) * (3/(2*T))**2)
                    y[k] = y[k] - (1/2) * g**2 * gpp / gp**3

                if n_terms >= 3:
                    yp_k = (3*y[k] - 4*y[k-1] + y[k-2]) / (2*T)
                    g = (y[k] - 2*y[k-1] + y[k-2])/T**2 + f(y[k], yp_k) - u[k]
                    gp = 1/T**2 + df_dy(y[k], yp_k) + df_dyp(y[k], yp_k) * 3/(2*T)
                    gpp = (d2f_dy2(y[k], yp_k)
                           + 2 * d2f_dydyp(y[k], yp_k) * 3/(2*T)
                           + d2f_dyp2(y[k], yp_k) * (3/(2*T))**2)
                    gppp = d3f_dy3(y[k], yp_k) if d3f_dy3 is not None else 0.0
                    y[k] = y[k] - (1/6) * g**3 * (-gppp * gp + 3 * gpp**2) / gp**5

        else:  # n_points == 4
            # 4 puntos: y'_k = (11*y_k - 18*y_{k-1} + 9*y_{k-2} - 2*y_{k-3}) / (6T)
            #           y''_k = (2*y_k - 5*y_{k-1} + 4*y_{k-2} - y_{k-3}) / T²
            if n_iterations == 0:
                yp_k = (11*y[k] - 18*y[k-1] + 9*y[k-2] - 2*y[k-3]) / (6*T)
                g = (2*y[k] - 5*y[k-1] + 4*y[k-2] - y[k-3])/T**2 + f(y[k], yp_k) - u[k]
                gp = 2/T**2 + df_dy(y[k], yp_k) + df_dyp(y[k], yp_k) * 11/(6*T)
                gpp = (d2f_dy2(y[k], yp_k)
                       + 2 * d2f_dydyp(y[k], yp_k) * 11/(6*T)
                       + d2f_dyp2(y[k], yp_k) * (11/(6*T))**2)
                gppp = d3f_dy3(y[k], yp_k) if d3f_dy3 is not None else 0.0

                delta = - g / gp
                if n_terms >= 2:
                    delta += - (1/2) * g**2 * gpp / gp**3
                if n_terms >= 3:
                    delta += - (1/6) * g**3 * (-gppp * gp + 3 * gpp**2) / gp**5
                y[k] = y[k] + delta

            else:
                yp_k = (11*y[k] - 18*y[k-1] + 9*y[k-2] - 2*y[k-3]) / (6*T)
                g = (2*y[k] - 5*y[k-1] + 4*y[k-2] - y[k-3])/T**2 + f(y[k], yp_k) - u[k]
                gp = 2/T**2 + df_dy(y[k], yp_k) + df_dyp(y[k], yp_k) * 11/(6*T)
                y[k] = y[k] - g / gp

                if n_terms >= 2:
                    yp_k = (11*y[k] - 18*y[k-1] + 9*y[k-2] - 2*y[k-3]) / (6*T)
                    g = (2*y[k] - 5*y[k-1] + 4*y[k-2] - y[k-3])/T**2 + f(y[k], yp_k) - u[k]
                    gp = 2/T**2 + df_dy(y[k], yp_k) + df_dyp(y[k], yp_k) * 11/(6*T)
                    gpp = (d2f_dy2(y[k], yp_k)
                           + 2 * d2f_dydyp(y[k], yp_k) * 11/(6*T)
                           + d2f_dyp2(y[k], yp_k) * (11/(6*T))**2)
                    y[k] = y[k] - (1/2) * g**2 * gpp / gp**3

                if n_terms >= 3:
                    yp_k = (11*y[k] - 18*y[k-1] + 9*y[k-2] - 2*y[k-3]) / (6*T)
                    g = (2*y[k] - 5*y[k-1] + 4*y[k-2] - y[k-3])/T**2 + f(y[k], yp_k) - u[k]
                    gp = 2/T**2 + df_dy(y[k], yp_k) + df_dyp(y[k], yp_k) * 11/(6*T)
                    gpp = (d2f_dy2(y[k], yp_k)
                           + 2 * d2f_dydyp(y[k], yp_k) * 11/(6*T)
                           + d2f_dyp2(y[k], yp_k) * (11/(6*T))**2)
                    gppp = d3f_dy3(y[k], yp_k) if d3f_dy3 is not None else 0.0
                    y[k] = y[k] - (1/6) * g**3 * (-gppp * gp + 3 * gpp**2) / gp**5

    return y


def run_ejemplo_1(n_terms, n_iterations, n_points):
    n = 500
    t = np.linspace(0, 10, n)
    T = t[1] - t[0]
    sol = odeint(lambda y, t: -y**2 + np.sin(5*t), -0.2, t).ravel()
    f, df, d2f, d3f = lambda y: y**2, lambda y: 2*y, lambda y: 2.0, lambda y: 0.0
    u = np.sin(5*t)
    y = solve_order1_configurable(f, df, d2f, d3f, u, sol[0], sol[1], sol[2], T, n, n_terms, n_iterations, n_points)
    return np.max(np.abs(y - sol))


def run_ejemplo_2(n_terms, n_iterations, n_points):
    n = 500
    t = np.linspace(0, 10, n)
    T = t[1] - t[0]
    sol = odeint(lambda y, t: -np.sin(y)**2 + np.sin(5*t), -0.2, t).ravel()
    f = lambda y: np.sin(y)**2
    df = lambda y: 2*np.sin(y)*np.cos(y)
    d2f = lambda y: 2*np.cos(y)**2 - 2*np.sin(y)**2
    d3f = lambda y: -8*np.sin(y)*np.cos(y)
    u = np.sin(5*t)
    y = solve_order1_configurable(f, df, d2f, d3f, u, sol[0], sol[1], sol[2], T, n, n_terms, n_iterations, n_points)
    return np.max(np.abs(y - sol))


def run_ejemplo_3(n_terms, n_iterations, n_points):
    n = 100
    t = np.linspace(-1, 1, n)
    T = t[1] - t[0]
    beta = lambda y: -1/10*y**3 + 1/10*y**2 + y - 1
    db = lambda y: -3/10*y**2 + 2/10*y + 1
    db2 = lambda y: -6/10*y + 2/10
    db3 = lambda y: -6/10
    sol = odeint(lambda y, t: -beta(y) + np.sin(5*t), -0.2, t).ravel()
    u = np.sin(5*t)
    y = solve_order1_configurable(beta, db, db2, db3, u, sol[0], sol[1], sol[2], T, n, n_terms, n_iterations, n_points)
    return np.max(np.abs(y - sol))


def run_ejemplo_5(n_terms, n_iterations, n_points):
    mu = 0.1
    n = 1000
    t = np.linspace(0, 10, n)
    T = t[1] - t[0]
    def model(z, t):
        y, dydt = z
        return [dydt, -mu*dydt - np.sin(y) + np.sin(3*t)]
    sol = odeint(model, [0.5, 0], t)[:, 0]
    f = lambda y, yp: mu*yp + np.sin(y)
    df_dy = lambda y, yp: np.cos(y)
    df_dyp = lambda y, yp: mu
    d2f_dy2 = lambda y, yp: -np.sin(y)
    d2f_dydyp = lambda y, yp: 0.0
    d2f_dyp2 = lambda y, yp: 0.0
    d3f_dy3 = lambda y, yp: -np.cos(y)
    u = np.sin(3*t)
    y = solve_order2_configurable(f, df_dy, df_dyp, d2f_dy2, d2f_dydyp, d2f_dyp2,
                                   d3f_dy3, u, sol[0], sol[1], sol[2], T, n, n_terms, n_iterations, n_points)
    return np.max(np.abs(y - sol))


def run_ejemplo_A(n_terms, n_iterations, n_points):
    a, b, c = 1, 2, 3
    n = 1000
    t_eval = np.linspace(0, 10, n)
    T = t_eval[1] - t_eval[0]
    def sistema(t, z):
        y1, y2 = z
        return [y2, np.sin(t) - a*y2 - b*y2*(y1**2 - 1) - c*y1*y2 - y1]
    sol_ivp = solve_ivp(sistema, [0, 10], [0, 0], t_eval=t_eval, method='RK45')
    sol = sol_ivp.y[0]
    f = lambda y, yp: a*yp + b*yp*(y**2 - 1) + c*y*yp + y
    df_dy = lambda y, yp: 2*b*yp*y + c*yp + 1
    df_dyp = lambda y, yp: a + b*(y**2 - 1) + c*y
    d2f_dy2 = lambda y, yp: 2*b*yp
    d2f_dydyp = lambda y, yp: 2*b*y + c
    d2f_dyp2 = lambda y, yp: 0.0
    d3f_dy3 = lambda y, yp: 0.0
    u = np.sin(t_eval)
    y = solve_order2_configurable(f, df_dy, df_dyp, d2f_dy2, d2f_dydyp, d2f_dyp2,
                                   d3f_dy3, u, sol[0], sol[1], sol[2], T, n, n_terms, n_iterations, n_points)
    return np.max(np.abs(y - sol))


if __name__ == "__main__":
    ejemplos = {
        'Ej1 (y\'+y²)': run_ejemplo_1,
        'Ej2 (y\'+sin²y)': run_ejemplo_2,
        'Ej3 (y\'+β(y))': run_ejemplo_3,
        'Ej5 (péndulo)': run_ejemplo_5,
        'EjA (y\'\' cmplx)': run_ejemplo_A,
    }

    # 8 configuraciones: 4 con 3pt + 4 con 4pt
    configs = [
        ('0i-2p-3pt', 2, 0, 3),
        ('0i-3p-3pt', 3, 0, 3),
        ('1i-2p-3pt', 2, 1, 3),
        ('1i-3p-3pt', 3, 1, 3),
        ('0i-2p-4pt', 2, 0, 4),
        ('0i-3p-4pt', 3, 0, 4),
        ('1i-2p-4pt', 2, 1, 4),
        ('1i-3p-4pt', 3, 1, 4),
    ]

    print("\n" + "="*110)
    print("COMPARACIÓN 3 PUNTOS vs 4 PUNTOS")
    print("="*110)
    header = f"{'Ejemplo':<17}"
    for cfg in configs:
        header += f" {cfg[0]:<12}"
    print(header)
    print("-"*110)

    for nombre, func in ejemplos.items():
        errores = []
        for config_name, n_terms, n_iterations, n_points in configs:
            try:
                error = func(n_terms, n_iterations, n_points)
                errores.append(error)
            except Exception as e:
                errores.append(float('nan'))
                print(f"ERROR en {nombre} con {config_name}: {e}")

        row = f"{nombre:<17}"
        for err in errores:
            row += f" {err:<12.2e}"
        print(row)

    print("="*110)
    print("\nLeyenda: 0i=0 iter, 1i=1 iter, 2p=2 términos, 3p=3 términos, 3pt=3 puntos, 4pt=4 puntos")
