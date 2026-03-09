"""
Benchmark COMPLETO: Todos los ejemplos (incluyendo B, C, fricción).
"""

import numpy as np
from scipy.integrate import odeint, solve_ivp
from scipy.interpolate import Rbf
from numpy.polynomial import Polynomial


def solve_order1_configurable(f, df, d2f, d3f, u, y0, y1, T, n, n_terms=3, n_iterations=1):
    """Solve y' + f(y) = u(t) con configuración flexible."""
    y = np.zeros(n)
    y[0] = y0
    y[1] = y1

    for k in range(2, n):
        y[k] = y[k-1]

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

    return y


def solve_order2_configurable(f, df_dy, df_dyp, d2f_dy2, d2f_dydyp, d2f_dyp2,
                                d3f_dy3, u, y0, y1, T, n, n_terms=3, n_iterations=1):
    """Solve y'' + f(y, y') = u(t) con configuración flexible."""
    y = np.zeros(n)
    y[0] = y0
    y[1] = y1

    for k in range(2, n):
        y[k] = y[k-1]

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

    return y


def run_ejemplo_1(n_terms, n_iterations):
    n = 500
    t = np.linspace(0, 10, n)
    T = t[1] - t[0]
    sol = odeint(lambda y, t: -y**2 + np.sin(5*t), -0.2, t).ravel()
    f, df, d2f, d3f = lambda y: y**2, lambda y: 2*y, lambda y: 2.0, lambda y: 0.0
    u = np.sin(5*t)
    y = solve_order1_configurable(f, df, d2f, d3f, u, sol[0], sol[1], T, n, n_terms, n_iterations)
    return np.max(np.abs(y - sol))


def run_ejemplo_2(n_terms, n_iterations):
    n = 500
    t = np.linspace(0, 10, n)
    T = t[1] - t[0]
    sol = odeint(lambda y, t: -np.sin(y)**2 + np.sin(5*t), -0.2, t).ravel()
    f = lambda y: np.sin(y)**2
    df = lambda y: 2*np.sin(y)*np.cos(y)
    d2f = lambda y: 2*np.cos(y)**2 - 2*np.sin(y)**2
    d3f = lambda y: -8*np.sin(y)*np.cos(y)
    u = np.sin(5*t)
    y = solve_order1_configurable(f, df, d2f, d3f, u, sol[0], sol[1], T, n, n_terms, n_iterations)
    return np.max(np.abs(y - sol))


def run_ejemplo_3(n_terms, n_iterations):
    n = 100
    t = np.linspace(-1, 1, n)
    T = t[1] - t[0]
    beta = lambda y: -1/10*y**3 + 1/10*y**2 + y - 1
    db = lambda y: -3/10*y**2 + 2/10*y + 1
    db2 = lambda y: -6/10*y + 2/10
    db3 = lambda y: -6/10
    sol = odeint(lambda y, t: -beta(y) + np.sin(5*t), -0.2, t).ravel()
    u = np.sin(5*t)
    y = solve_order1_configurable(beta, db, db2, db3, u, sol[0], sol[1], T, n, n_terms, n_iterations)
    return np.max(np.abs(y - sol))


def run_ejemplo_5(n_terms, n_iterations):
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
                                   d3f_dy3, u, sol[0], sol[1], T, n, n_terms, n_iterations)
    return np.max(np.abs(y - sol))


def run_ejemplo_A(n_terms, n_iterations):
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
                                   d3f_dy3, u, sol[0], sol[1], T, n, n_terms, n_iterations)
    return np.max(np.abs(y - sol))


def run_ejemplo_B(n_terms, n_iterations):
    x_ = np.array([0, 0.2, 0.5, 0.8, 1])
    y_ = np.array([0, 0.32, 0.5, 0.7, 1])
    polynomial = Polynomial.fit(x_, y_, len(x_) - 1)
    e, d, c, b, a = np.polyfit(x_, y_, 4)
    n = 1000
    t = np.linspace(0, 2, n)
    T = t[1] - t[0]
    sol = odeint(lambda y, t: -polynomial(y) + np.sin(5*t) + 1, -0.5, t).ravel()
    f = lambda y: a + b*y + c*y**2 + d*y**3 + e*y**4
    df = lambda y: b + 2*c*y + 3*d*y**2 + 4*e*y**3
    d2f = lambda y: 2*c + 6*d*y + 12*e*y**2
    d3f = lambda y: 6*d + 24*e*y
    u = np.sin(5*t) + 1
    y = solve_order1_configurable(f, df, d2f, d3f, u, sol[0], sol[1], T, n, n_terms, n_iterations)
    return np.max(np.abs(y - sol))


def run_ejemplo_C(n_terms, n_iterations):
    """Ejemplo C: RBF con derivadas numéricas"""
    x_ = np.array([0, 0.2, 0.5, 0.8, 1])
    y_ = np.array([0, 0.32, 0.5, 0.7, 1])
    polynomial = Polynomial.fit(x_, y_, len(x_) - 1)
    rbf = Rbf(x_, y_, function='multiquadric', epsilon=2)
    n = 1000
    t = np.linspace(0, 2, n)
    T = t[1] - t[0]
    sol = odeint(lambda y, t: -polynomial(y) + np.sin(5*t) + 1, -0.5, t).ravel()
    u = np.sin(5*t) + 1

    # Derivadas numéricas
    h = 1e-5
    df = lambda y: (8*(rbf(y+h) - rbf(y-h)) - (rbf(y+2*h) - rbf(y-2*h))) / (12*h)
    d2f = lambda y: (-rbf(y+2*h) + 16*rbf(y+h) - 30*rbf(y) + 16*rbf(y-h) - rbf(y-2*h)) / (12*h**2)
    d3f = lambda y: (-rbf(y+2*h) + 2*rbf(y+h) - 2*rbf(y-h) + rbf(y-2*h)) / (2*h**3)

    y = solve_order1_configurable(rbf, df, d2f, d3f, u, sol[0], sol[1], T, n, n_terms, n_iterations)
    return np.max(np.abs(y - sol))


def run_ejemplo_friccion(n_terms, n_iterations):
    k1, k2, k3, vs = 0.5, 0.3, 0.2, 0.1
    n = 2000
    t = np.linspace(0, 10, n)
    T = t[1] - t[0]
    def model(z, t):
        y, yp = z
        fric = k1*yp + k2*np.sign(yp) + k3*yp*np.exp(-np.abs(yp)/vs)
        return [yp, -fric + np.sin(2*t)]
    sol = odeint(model, [0, 0.5], t)[:, 0]

    def smsign(x, eps=1e-3):
        return x / np.sqrt(x**2 + eps**2)
    def dsmsign(x, eps=1e-3):
        return eps**2 / (x**2 + eps**2)**1.5

    f = lambda y, yp: k1*yp + k2*smsign(yp) + k3*yp*np.exp(-np.abs(yp)/vs)
    df_dy = lambda y, yp: 0.0
    df_dyp = lambda y, yp: k1 + k2*dsmsign(yp) + k3*np.exp(-np.abs(yp)/vs)*(1 - np.abs(yp)/vs)
    d2f_dy2 = lambda y, yp: 0.0
    d2f_dydyp = lambda y, yp: 0.0
    d2f_dyp2 = lambda y, yp: (k2 * (-3*yp) / (yp**2 + 1e-6)**2.5 * 1e-6
                              + k3*np.exp(-np.abs(yp)/vs)*(-2*smsign(yp)/vs + yp*smsign(yp)**2/vs**2))
    u = np.sin(2*t)
    y = solve_order2_configurable(f, df_dy, df_dyp, d2f_dy2, d2f_dydyp, d2f_dyp2,
                                   None, u, sol[0], sol[1], T, n, n_terms, n_iterations)
    return np.max(np.abs(y - sol))


if __name__ == "__main__":
    ejemplos = {
        'Ej1 (y\'+y²)': run_ejemplo_1,
        'Ej2 (y\'+sin²y)': run_ejemplo_2,
        'Ej3 (y\'+β(y))': run_ejemplo_3,
        'Ej5 (péndulo)': run_ejemplo_5,
        'EjA (y\'\' cmplx)': run_ejemplo_A,
        'EjB (polinomio)': run_ejemplo_B,
        'EjC (RBF)': run_ejemplo_C,
        'Fricción': run_ejemplo_friccion,
    }

    configs = [('0i-2p', 2, 0), ('0i-3p', 3, 0), ('1i-2p', 2, 1), ('1i-3p', 3, 1)]

    print("\n" + "="*75)
    print("BENCHMARK COMPLETO - 8 EJEMPLOS × 4 CONFIGURACIONES")
    print("="*75)
    print(f"{'Ejemplo':<20} {'0i-2p':<13} {'0i-3p':<13} {'1i-2p':<13} {'1i-3p':<13}")
    print("-"*75)

    for nombre, func in ejemplos.items():
        errores = []
        for config_name, n_terms, n_iterations in configs:
            try:
                error = func(n_terms, n_iterations)
                errores.append(error)
            except Exception as e:
                errores.append(float('nan'))

        print(f"{nombre:<20} {errores[0]:<13.2e} {errores[1]:<13.2e} {errores[2]:<13.2e} {errores[3]:<13.2e}")

    print("="*75)
