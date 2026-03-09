"""
Benchmark: Comparación de configuraciones de series de homotopía.

Compara 4 configuraciones:
- 0 iteraciones, 2 términos (0i-2p)
- 0 iteraciones, 3 términos (0i-3p)
- 1 iteración, 2 términos (1i-2p)
- 1 iteración, 3 términos (1i-3p)

Iteración: recalcular g después de cada término z_i
"""

import numpy as np
from scipy.integrate import odeint, solve_ivp


def solve_order1_configurable(f, df, d2f, d3f, u, y0, y1, T, n, n_terms=3, n_iterations=1):
    """
    Solve y' + f(y) = u(t) con configuración flexible.

    Parameters
    ----------
    n_terms : int (2 or 3)
        Número de términos de la serie de homotopía.
    n_iterations : int (0 or 1)
        0 = calcular todos los términos con g inicial (sin recalcular)
        1 = recalcular g después de cada término (actual)
    """
    y = np.zeros(n)
    y[0] = y0
    y[1] = y1

    for k in range(2, n):
        y[k] = y[k-1]

        if n_iterations == 0:
            # 0 iteraciones: calcular todos los términos con g inicial
            g = (3/2)*y[k]/T - 2*y[k-1]/T + (1/2)*y[k-2]/T + f(y[k]) - u[k]
            gp = 3/(2*T) + df(y[k])
            gpp = d2f(y[k])
            gppp = d3f(y[k])

            # z1
            delta = - g / gp

            # z2
            if n_terms >= 2:
                delta += - (1/2) * g**2 * gpp / gp**3

            # z3
            if n_terms >= 3:
                delta += - (1/6) * g**3 * (-gppp * gp + 3 * gpp**2) / gp**5

            y[k] = y[k] + delta

        else:
            # 1 iteración: recalcular g después de cada término
            # z1
            g = (3/2)*y[k]/T - 2*y[k-1]/T + (1/2)*y[k-2]/T + f(y[k]) - u[k]
            gp = 3/(2*T) + df(y[k])
            y[k] = y[k] - g / gp

            # z2
            if n_terms >= 2:
                g = (3/2)*y[k]/T - 2*y[k-1]/T + (1/2)*y[k-2]/T + f(y[k]) - u[k]
                gp = 3/(2*T) + df(y[k])
                gpp = d2f(y[k])
                y[k] = y[k] - (1/2) * g**2 * gpp / gp**3

            # z3
            if n_terms >= 3:
                g = (3/2)*y[k]/T - 2*y[k-1]/T + (1/2)*y[k-2]/T + f(y[k]) - u[k]
                gp = 3/(2*T) + df(y[k])
                gpp = d2f(y[k])
                gppp = d3f(y[k])
                y[k] = y[k] - (1/6) * g**3 * (-gppp * gp + 3 * gpp**2) / gp**5

    return y


def solve_order2_configurable(f, df_dy, df_dyp, d2f_dy2, d2f_dydyp, d2f_dyp2,
                                d3f_dy3, u, y0, y1, T, n, n_terms=3, n_iterations=1):
    """
    Solve y'' + f(y, y') = u(t) con configuración flexible.
    """
    y = np.zeros(n)
    y[0] = y0
    y[1] = y1

    for k in range(2, n):
        y[k] = y[k-1]

        if n_iterations == 0:
            # 0 iteraciones
            yp_k = (3*y[k] - 4*y[k-1] + y[k-2]) / (2*T)
            g = (y[k] - 2*y[k-1] + y[k-2])/T**2 + f(y[k], yp_k) - u[k]
            gp = 1/T**2 + df_dy(y[k], yp_k) + df_dyp(y[k], yp_k) * 3/(2*T)
            gpp = (d2f_dy2(y[k], yp_k)
                   + 2 * d2f_dydyp(y[k], yp_k) * 3/(2*T)
                   + d2f_dyp2(y[k], yp_k) * (3/(2*T))**2)
            gppp = d3f_dy3(y[k], yp_k) if d3f_dy3 is not None else 0.0

            # z1
            delta = - g / gp

            # z2
            if n_terms >= 2:
                delta += - (1/2) * g**2 * gpp / gp**3

            # z3
            if n_terms >= 3:
                delta += - (1/6) * g**3 * (-gppp * gp + 3 * gpp**2) / gp**5

            y[k] = y[k] + delta

        else:
            # 1 iteración
            # z1
            yp_k = (3*y[k] - 4*y[k-1] + y[k-2]) / (2*T)
            g = (y[k] - 2*y[k-1] + y[k-2])/T**2 + f(y[k], yp_k) - u[k]
            gp = 1/T**2 + df_dy(y[k], yp_k) + df_dyp(y[k], yp_k) * 3/(2*T)
            y[k] = y[k] - g / gp

            # z2
            if n_terms >= 2:
                yp_k = (3*y[k] - 4*y[k-1] + y[k-2]) / (2*T)
                g = (y[k] - 2*y[k-1] + y[k-2])/T**2 + f(y[k], yp_k) - u[k]
                gp = 1/T**2 + df_dy(y[k], yp_k) + df_dyp(y[k], yp_k) * 3/(2*T)
                gpp = (d2f_dy2(y[k], yp_k)
                       + 2 * d2f_dydyp(y[k], yp_k) * 3/(2*T)
                       + d2f_dyp2(y[k], yp_k) * (3/(2*T))**2)
                y[k] = y[k] - (1/2) * g**2 * gpp / gp**3

            # z3
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


# ============================================================
# Ejemplos de prueba
# ============================================================

def run_ejemplo_1(n_terms, n_iterations):
    """Ejemplo 1: y' + y² = sin(5t)"""
    n = 500
    t = np.linspace(0, 10, n)
    T = t[1] - t[0]

    sol = odeint(lambda y, t: -y**2 + np.sin(5*t), -0.2, t).ravel()

    f   = lambda y: y**2
    df  = lambda y: 2*y
    d2f = lambda y: 2.0
    d3f = lambda y: 0.0
    u = np.sin(5*t)

    y = solve_order1_configurable(f, df, d2f, d3f, u, sol[0], sol[1], T, n,
                                   n_terms=n_terms, n_iterations=n_iterations)

    return np.max(np.abs(y - sol))


def run_ejemplo_2(n_terms, n_iterations):
    """Ejemplo 2: y' + sin²(y) = sin(5t)"""
    n = 500
    t = np.linspace(0, 10, n)
    T = t[1] - t[0]

    sol = odeint(lambda y, t: -np.sin(y)**2 + np.sin(5*t), -0.2, t).ravel()

    f   = lambda y: np.sin(y)**2
    df  = lambda y: 2*np.sin(y)*np.cos(y)
    d2f = lambda y: 2*np.cos(y)**2 - 2*np.sin(y)**2
    d3f = lambda y: -8*np.sin(y)*np.cos(y)
    u = np.sin(5*t)

    y = solve_order1_configurable(f, df, d2f, d3f, u, sol[0], sol[1], T, n,
                                   n_terms=n_terms, n_iterations=n_iterations)

    return np.max(np.abs(y - sol))


def run_ejemplo_3(n_terms, n_iterations):
    """Ejemplo 3: y' + β(y) = sin(5t)"""
    n = 100
    t = np.linspace(-1, 1, n)
    T = t[1] - t[0]

    def beta(y): return -1/10*y**3 + 1/10*y**2 + y - 1
    def db(y):   return -3/10*y**2 + 2/10*y + 1
    def db2(y):  return -6/10*y + 2/10
    def db3(y):  return -6/10

    sol = odeint(lambda y, t: -beta(y) + np.sin(5*t), -0.2, t).ravel()
    u = np.sin(5*t)

    y = solve_order1_configurable(beta, db, db2, db3, u, sol[0], sol[1], T, n,
                                   n_terms=n_terms, n_iterations=n_iterations)

    return np.max(np.abs(y - sol))


def run_ejemplo_5(n_terms, n_iterations):
    """Ejemplo 5: y'' + 0.1y' + sin(y) = sin(3t)"""
    mu = 0.1
    n = 1000
    t = np.linspace(0, 10, n)
    T = t[1] - t[0]

    def model(z, t):
        y, dydt = z
        return [dydt, -mu*dydt - np.sin(y) + np.sin(3*t)]

    sol = odeint(model, [0.5, 0], t)[:, 0]

    f          = lambda y, yp: mu*yp + np.sin(y)
    df_dy      = lambda y, yp: np.cos(y)
    df_dyp     = lambda y, yp: mu
    d2f_dy2    = lambda y, yp: -np.sin(y)
    d2f_dydyp  = lambda y, yp: 0.0
    d2f_dyp2   = lambda y, yp: 0.0
    d3f_dy3    = lambda y, yp: -np.cos(y)
    u = np.sin(3*t)

    y = solve_order2_configurable(f, df_dy, df_dyp, d2f_dy2, d2f_dydyp, d2f_dyp2,
                                   d3f_dy3, u, sol[0], sol[1], T, n,
                                   n_terms=n_terms, n_iterations=n_iterations)

    return np.max(np.abs(y - sol))


def run_ejemplo_A(n_terms, n_iterations):
    """Ejemplo A: y'' + ay' + by'(y²-1) + cyy' + y = sin(t)"""
    a, b, c = 1, 2, 3
    n = 1000
    t_eval = np.linspace(0, 10, n)
    T = t_eval[1] - t_eval[0]

    def sistema(t, z):
        y1, y2 = z
        return [y2, np.sin(t) - a*y2 - b*y2*(y1**2 - 1) - c*y1*y2 - y1]

    sol_ivp = solve_ivp(sistema, [0, 10], [0, 0], t_eval=t_eval, method='RK45')
    sol = sol_ivp.y[0]

    def f(y, yp):
        return a*yp + b*yp*(y**2 - 1) + c*y*yp + y

    def df_dy(y, yp):
        return 2*b*yp*y + c*yp + 1

    def df_dyp(y, yp):
        return a + b*(y**2 - 1) + c*y

    def d2f_dy2(y, yp):
        return 2*b*yp

    def d2f_dydyp(y, yp):
        return 2*b*y + c

    def d2f_dyp2(y, yp):
        return 0.0

    def d3f_dy3(y, yp):
        return 0.0

    u = np.sin(t_eval)

    y = solve_order2_configurable(f, df_dy, df_dyp, d2f_dy2, d2f_dydyp, d2f_dyp2,
                                   d3f_dy3, u, sol[0], sol[1], T, n,
                                   n_terms=n_terms, n_iterations=n_iterations)

    return np.max(np.abs(y - sol))


# ============================================================
# Ejecutar benchmark
# ============================================================

if __name__ == "__main__":
    ejemplos = {
        'Ejemplo 1 (y\'+y²)': run_ejemplo_1,
        'Ejemplo 2 (y\'+sin²y)': run_ejemplo_2,
        'Ejemplo 3 (y\'+β(y))': run_ejemplo_3,
        'Ejemplo 5 (péndulo)': run_ejemplo_5,
        'Ejemplo A (y\'\' complejo)': run_ejemplo_A,
    }

    configs = [
        ('0i-2p', 2, 0),
        ('0i-3p', 3, 0),
        ('1i-2p', 2, 1),
        ('1i-3p', 3, 1),
    ]

    print("\n" + "="*80)
    print("COMPARACIÓN DE CONFIGURACIONES DE SERIES DE HOMOTOPÍA")
    print("="*80)
    print(f"{'Ejemplo':<30} {'0i-2p':<15} {'0i-3p':<15} {'1i-2p':<15} {'1i-3p':<15}")
    print("-"*80)

    for nombre, func in ejemplos.items():
        errores = []
        for config_name, n_terms, n_iterations in configs:
            try:
                error = func(n_terms, n_iterations)
                errores.append(error)
            except Exception as e:
                errores.append(float('nan'))
                print(f"ERROR en {nombre} con {config_name}: {e}")

        print(f"{nombre:<30} {errores[0]:<15.2e} {errores[1]:<15.2e} {errores[2]:<15.2e} {errores[3]:<15.2e}")

    print("="*80)
    print("\nLeyenda:")
    print("  0i-2p: 0 iteraciones, 2 términos")
    print("  0i-3p: 0 iteraciones, 3 términos")
    print("  1i-2p: 1 iteración, 2 términos")
    print("  1i-3p: 1 iteración, 3 términos")
    print("\nIteración = recalcular g después de cada término z_i")
