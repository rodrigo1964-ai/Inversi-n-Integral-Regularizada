"""
examples.py - All thesis examples using homotopy_regressors library.

Each example compares the homotopy regressor against RK45.

Author: Rodolfo H. Rodrigo - UNSJ
"""

import numpy as np
from scipy.integrate import odeint, solve_ivp
from solver import solve_order1, solve_order2, solve_order1_numeric


def ejemplo_1():
    """
    Ejemplo 1: y' + y² = sin(5t)
    f(y) = y², f'= 2y, f''= 2, f'''= 0
    """
    print("Ejemplo 1: y' + y² = sin(5t)")

    n = 500
    t = np.linspace(0, 10, n)
    T = t[1] - t[0]

    # RK reference
    sol = odeint(lambda y, t: -y**2 + np.sin(5*t), -0.2, t).ravel()

    # Homotopy regressor
    f   = lambda y: y**2
    df  = lambda y: 2*y
    d2f = lambda y: 2.0
    d3f = lambda y: 0.0
    u = np.sin(5*t)

    y = solve_order1(f, df, d2f, d3f, u, sol[0], sol[1], T, n)

    err = np.max(np.abs(y - sol))
    print(f"  Error máximo: {err:.2e}")
    return t, sol, y


def ejemplo_2():
    """
    Ejemplo 2: y' + sin²(y) = sin(5t)
    f(y) = sin²(y)
    """
    print("Ejemplo 2: y' + sin²(y) = sin(5t)")

    n = 500
    t = np.linspace(0, 10, n)
    T = t[1] - t[0]

    sol = odeint(lambda y, t: -np.sin(y)**2 + np.sin(5*t), -0.2, t).ravel()

    f    = lambda y: np.sin(y)**2
    df   = lambda y: 2*np.sin(y)*np.cos(y)
    d2f  = lambda y: 2*np.cos(y)**2 - 2*np.sin(y)**2
    d3f  = lambda y: -8*np.sin(y)*np.cos(y)
    u = np.sin(5*t)

    y = solve_order1(f, df, d2f, d3f, u, sol[0], sol[1], T, n)

    err = np.max(np.abs(y - sol))
    print(f"  Error máximo: {err:.2e}")
    return t, sol, y


def ejemplo_3():
    """
    Ejemplo 3: y' + β(y) = sin(5t)
    β(y) = -y³/10 + y²/10 + y - 1
    """
    print("Ejemplo 3: y' + β(y) = sin(5t)")

    n = 100
    t = np.linspace(-1, 1, n)
    T = t[1] - t[0]

    def beta(y): return -1/10*y**3 + 1/10*y**2 + y - 1
    def db(y):   return -3/10*y**2 + 2/10*y + 1
    def db2(y):  return -6/10*y + 2/10
    def db3(y):  return -6/10

    sol = odeint(lambda y, t: -beta(y) + np.sin(5*t), -0.2, t).ravel()
    u = np.sin(5*t)

    y = solve_order1(beta, db, db2, db3, u, sol[0], sol[1], T, n)

    err = np.max(np.abs(y - sol))
    print(f"  Error máximo: {err:.2e}")
    return t, sol, y


def ejemplo_5():
    """
    Ejemplo 5: y'' + μy' + sin(y) = sin(3t)  (péndulo amortiguado)
    f(y, y') = μ*y' + sin(y),  μ = 0.1
    """
    print("Ejemplo 5: y'' + 0.1y' + sin(y) = sin(3t)")

    mu = 0.1
    n = 1000
    t = np.linspace(0, 10, n)
    T = t[1] - t[0]

    def model(z, t):
        y, dydt = z
        return [dydt, -mu*dydt - np.sin(y) + np.sin(3*t)]

    sol = odeint(model, [0.5, 0], t)[:, 0]

    # f(y, yp) = mu*yp + sin(y)
    f          = lambda y, yp: mu*yp + np.sin(y)
    df_dy      = lambda y, yp: np.cos(y)
    df_dyp     = lambda y, yp: mu
    d2f_dy2    = lambda y, yp: -np.sin(y)
    d2f_dydyp  = lambda y, yp: 0.0
    d2f_dyp2   = lambda y, yp: 0.0
    d3f_dy3    = lambda y, yp: -np.cos(y)
    u = np.sin(3*t)

    y = solve_order2(f, df_dy, df_dyp, d2f_dy2, d2f_dydyp, d2f_dyp2,
                     d3f_dy3, u, sol[0], sol[1], T, n)

    err = np.max(np.abs(y - sol))
    print(f"  Error máximo: {err:.2e}")
    return t, sol, y


def ejemplo_A():
    """
    Ejemplo A: y'' + ay' + by'(y²-1) + cyy' + y = sin(t)
    a=1, b=2, c=3
    """
    print("Ejemplo A: y'' + ay' + by'(y²-1) + cyy' + y = sin(t)")

    a, b, c = 1, 2, 3
    n = 1000
    t_eval = np.linspace(0, 10, n)
    T = t_eval[1] - t_eval[0]

    def sistema(t, z):
        y1, y2 = z
        return [y2, np.sin(t) - a*y2 - b*y2*(y1**2 - 1) - c*y1*y2 - y1]

    sol_ivp = solve_ivp(sistema, [0, 10], [0, 0], t_eval=t_eval, method='RK45')
    sol = sol_ivp.y[0]

    # f(y, yp) = a*yp + b*yp*(y²-1) + c*y*yp + y
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

    y = solve_order2(f, df_dy, df_dyp, d2f_dy2, d2f_dydyp, d2f_dyp2,
                     d3f_dy3, u, sol[0], sol[1], T, n)

    err = np.max(np.abs(y - sol))
    print(f"  Error máximo: {err:.2e}")
    return t_eval, sol, y


def ejemplo_B():
    """
    Ejemplo B: y' + P(y) = sin(5t) + 1
    P(y) = polinomio interpolado por puntos de colocación
    """
    print("Ejemplo B: y' + P(y) = sin(5t) + 1  (polinomio)")

    from numpy.polynomial import Polynomial

    x_ = np.array([0, 0.2, 0.5, 0.8, 1])
    y_ = np.array([0, 0.32, 0.5, 0.7, 1])
    polynomial = Polynomial.fit(x_, y_, len(x_) - 1)

    e, d, c, b, a = np.polyfit(x_, y_, 4)

    n = 1000
    t = np.linspace(0, 2, n)
    T = t[1] - t[0]

    sol = odeint(lambda y, t: -polynomial(y) + np.sin(5*t) + 1, -0.5, t).ravel()

    # f(y) = a + b*y + c*y² + d*y³ + e*y⁴
    f    = lambda y: a + b*y + c*y**2 + d*y**3 + e*y**4
    df   = lambda y: b + 2*c*y + 3*d*y**2 + 4*e*y**3
    d2f  = lambda y: 2*c + 6*d*y + 12*e*y**2
    d3f  = lambda y: 6*d + 24*e*y
    u = np.sin(5*t) + 1

    y = solve_order1(f, df, d2f, d3f, u, sol[0], sol[1], T, n)

    err = np.max(np.abs(y - sol))
    print(f"  Error máximo: {err:.2e}")
    return t, sol, y


def ejemplo_C():
    """
    Ejemplo C: y' + RBF(y) = sin(5t) + 1
    RBF ajustada a datos — derivadas numéricas
    """
    print("Ejemplo C: y' + RBF(y) = sin(5t) + 1  (derivadas numéricas)")

    from scipy.interpolate import Rbf
    from numpy.polynomial import Polynomial

    x_ = np.array([0, 0.2, 0.5, 0.8, 1])
    y_ = np.array([0, 0.32, 0.5, 0.7, 1])
    polynomial = Polynomial.fit(x_, y_, len(x_) - 1)

    rbf = Rbf(x_, y_, function='multiquadric', epsilon=2)

    n = 1000
    t = np.linspace(0, 2, n)
    T = t[1] - t[0]

    # Reference: solve with polynomial (same as Ejemplo B)
    sol = odeint(lambda y, t: -polynomial(y) + np.sin(5*t) + 1, -0.5, t).ravel()

    u = np.sin(5*t) + 1

    # Use numeric derivatives — only pass rbf
    y = solve_order1_numeric(rbf, u, sol[0], sol[1], T, n)

    err = np.max(np.abs(y - sol))
    print(f"  Error máximo: {err:.2e}")
    return t, sol, y


def ejemplo_friccion():
    """
    Ejemplo fricción compuesta:
    y'' + k1*y' + k2*sign(y') + k3*y'*exp(-|y'|/vs) = u(t)
    
    k1: viscosa, k2: Coulomb, k3: Stribeck
    """
    print("Ejemplo Fricción: y'' + k1*y' + k2*sign(y') + k3*y'*exp(-|y'|/vs) = F(t)")

    k1, k2, k3 = 0.5, 0.3, 0.2
    vs = 0.1  # velocidad Stribeck
    n = 2000
    t = np.linspace(0, 10, n)
    T = t[1] - t[0]

    def model(z, t):
        y, yp = z
        fric = k1*yp + k2*np.sign(yp) + k3*yp*np.exp(-np.abs(yp)/vs)
        return [yp, -fric + np.sin(2*t)]

    sol = odeint(model, [0, 0.5], t)[:, 0]

    # Smooth sign approximation for derivatives
    def smsign(x, eps=1e-3):
        return x / np.sqrt(x**2 + eps**2)

    def dsmsign(x, eps=1e-3):
        return eps**2 / (x**2 + eps**2)**1.5

    def f(y, yp):
        return k1*yp + k2*smsign(yp) + k3*yp*np.exp(-np.abs(yp)/vs)

    def df_dy(y, yp):
        return 0.0

    def df_dyp(y, yp):
        e = np.exp(-np.abs(yp)/vs)
        return k1 + k2*dsmsign(yp) + k3*e*(1 - np.abs(yp)/vs)

    def d2f_dy2(y, yp):
        return 0.0

    def d2f_dydyp(y, yp):
        return 0.0

    def d2f_dyp2(y, yp):
        e = np.exp(-np.abs(yp)/vs)
        s = smsign(yp)
        return (k2 * (-3*yp) / (yp**2 + 1e-6)**2.5 * 1e-6
                + k3*e*(-2*s/vs + yp*s**2/vs**2))

    u = np.sin(2*t)

    y = solve_order2(f, df_dy, df_dyp, d2f_dy2, d2f_dydyp, d2f_dyp2,
                     None, u, sol[0], sol[1], T, n)

    err = np.max(np.abs(y - sol))
    print(f"  Error máximo: {err:.2e}")
    return t, sol, y


# ============================================================
# Run all
# ============================================================
if __name__ == "__main__":
    results = {}

    for name, func in [
        ("Ejemplo 1", ejemplo_1),
        ("Ejemplo 2", ejemplo_2),
        ("Ejemplo 3", ejemplo_3),
        ("Ejemplo 5", ejemplo_5),
        ("Ejemplo A", ejemplo_A),
        ("Ejemplo B", ejemplo_B),
        ("Ejemplo C", ejemplo_C),
        ("Fricción",  ejemplo_friccion),
    ]:
        print()
        try:
            t, sol, y = func()
            results[name] = (t, sol, y)
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    for name, (t, sol, y) in results.items():
        err = np.max(np.abs(y - sol))
        print(f"  {name:20s}: error máx = {err:.2e}")
