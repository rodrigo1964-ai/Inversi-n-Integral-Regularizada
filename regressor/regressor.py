"""
regressor.py - Genera automáticamente el regresor homotópico de 3 puntos
               para una ODE dada.

Entrada: la ODE en forma simbólica
Salida: función Python numérica lista para usar

Soporta:
    1er orden: y' + f(y) = u
    2do orden: y'' + f(y, y') = u

Author: Rodolfo H. Rodrigo - UNSJ
"""

from sympy import (
    Symbol, Function, symbols, diff, simplify,
    lambdify, Rational, pprint
)
import numpy as np


def build_regressor_order1(f_expr, y_sym=None):
    """
    Genera el regresor homotópico para y' + f(y) = u, 3 puntos.

    Parameters
    ----------
    f_expr : sympy expression
        Expresión simbólica de f en función de y.
        Ejemplo: y**2, sin(y)**2, -y**3/10 + y**2/10 + y - 1
    y_sym : Symbol or None
        Símbolo de la variable y. Si None, busca 'y' en f_expr.

    Returns
    -------
    regressor : callable(u, y0, y1, T, n) -> array
        Función numérica que resuelve la ODE.
    info : dict
        Diccionario con f, f', f'', f''' simbólicos y numéricos.

    Example
    -------
    >>> from sympy import Symbol, sin
    >>> y = Symbol('y')
    >>> regressor, info = build_regressor_order1(y**2, y)
    >>> # Resuelve y' + y² = sin(5t)
    >>> import numpy as np
    >>> t = np.linspace(0, 10, 500)
    >>> u = np.sin(5*t)
    >>> sol = regressor(u, -0.2, y1_from_rk, t[1]-t[0], 500)
    """
    from sympy import sin, cos, exp, sqrt

    if y_sym is None:
        free = f_expr.free_symbols
        if len(free) == 1:
            y_sym = free.pop()
        else:
            raise ValueError("Especificá y_sym, hay más de un símbolo libre")

    # Derivadas simbólicas
    f   = f_expr
    df  = diff(f, y_sym)
    d2f = diff(df, y_sym)
    d3f = diff(d2f, y_sym)

    # Simplificar
    f   = simplify(f)
    df  = simplify(df)
    d2f = simplify(d2f)
    d3f = simplify(d3f)

    # Mostrar
    print("f(y)    =", f)
    print("f'(y)   =", df)
    print("f''(y)  =", d2f)
    print("f'''(y) =", d3f)

    # Convertir a funciones numéricas
    f_num   = lambdify(y_sym, f, modules='numpy')
    df_num  = lambdify(y_sym, df, modules='numpy')
    d2f_num = lambdify(y_sym, d2f, modules='numpy')
    d3f_num = lambdify(y_sym, d3f, modules='numpy')

    def regressor(u, y0, y1, T, n):
        y = np.zeros(n)
        y[0] = y0
        y[1] = y1

        for k in range(2, n):
            y[k] = y[k-1]

            # z1
            g  = (3/2)*y[k]/T - 2*y[k-1]/T + (1/2)*y[k-2]/T + f_num(y[k]) - u[k]
            gp = 3/(2*T) + df_num(y[k])
            y[k] = y[k] - g / gp

            # z2
            g  = (3/2)*y[k]/T - 2*y[k-1]/T + (1/2)*y[k-2]/T + f_num(y[k]) - u[k]
            gp = 3/(2*T) + df_num(y[k])
            gpp = d2f_num(y[k])
            y[k] = y[k] - (1/2) * g**2 * gpp / gp**3

            # z3
            g  = (3/2)*y[k]/T - 2*y[k-1]/T + (1/2)*y[k-2]/T + f_num(y[k]) - u[k]
            gp = 3/(2*T) + df_num(y[k])
            gpp = d2f_num(y[k])
            gppp = d3f_num(y[k])
            y[k] = y[k] - (1/6) * g**3 * (-gppp * gp + 3 * gpp**2) / gp**5

        return y

    info = {
        'f': f, 'df': df, 'd2f': d2f, 'd3f': d3f,
        'f_num': f_num, 'df_num': df_num, 'd2f_num': d2f_num, 'd3f_num': d3f_num,
        'y_sym': y_sym, 'order': 1
    }

    return regressor, info


def build_regressor_order2(f_expr, y_sym=None, yp_sym=None):
    """
    Genera el regresor homotópico para y'' + f(y, y') = u, 3 puntos.

    Parameters
    ----------
    f_expr : sympy expression
        Expresión simbólica de f en función de y, yp.
        Ejemplo: 0.1*yp + sin(y)  (péndulo amortiguado)
    y_sym : Symbol or None
        Símbolo para y.
    yp_sym : Symbol or None
        Símbolo para y'.

    Returns
    -------
    regressor : callable(u, y0, y1, T, n) -> array
    info : dict

    Example
    -------
    >>> from sympy import Symbol, sin
    >>> y, yp = Symbol('y'), Symbol('yp')
    >>> regressor, info = build_regressor_order2(0.1*yp + sin(y), y, yp)
    """
    from sympy import sin, cos, exp, sqrt

    if y_sym is None or yp_sym is None:
        raise ValueError("Especificá y_sym y yp_sym para segundo orden")

    # Derivadas parciales simbólicas
    f = f_expr

    df_dy   = diff(f, y_sym)
    df_dyp  = diff(f, yp_sym)

    d2f_dy2   = diff(df_dy, y_sym)
    d2f_dydyp = diff(df_dy, yp_sym)
    d2f_dyp2  = diff(df_dyp, yp_sym)

    d3f_dy3 = diff(d2f_dy2, y_sym)

    # Simplificar
    derivs = {}
    for name, expr in [
        ('f', f), ('df_dy', df_dy), ('df_dyp', df_dyp),
        ('d2f_dy2', d2f_dy2), ('d2f_dydyp', d2f_dydyp),
        ('d2f_dyp2', d2f_dyp2), ('d3f_dy3', d3f_dy3)
    ]:
        derivs[name] = simplify(expr)

    # Mostrar
    print("f(y,y')       =", derivs['f'])
    print("df/dy         =", derivs['df_dy'])
    print("df/dy'        =", derivs['df_dyp'])
    print("d²f/dy²       =", derivs['d2f_dy2'])
    print("d²f/dydy'     =", derivs['d2f_dydyp'])
    print("d²f/dy'²      =", derivs['d2f_dyp2'])
    print("d³f/dy³       =", derivs['d3f_dy3'])

    # Convertir a numéricas
    syms = (y_sym, yp_sym)
    nums = {}
    for name, expr in derivs.items():
        nums[name] = lambdify(syms, expr, modules='numpy')

    def regressor(u, y0, y1, T, n):
        y = np.zeros(n)
        y[0] = y0
        y[1] = y1

        for k in range(2, n):
            y[k] = y[k-1]

            # z1
            yp_k = (3*y[k] - 4*y[k-1] + y[k-2]) / (2*T)
            g  = (y[k] - 2*y[k-1] + y[k-2])/T**2 + nums['f'](y[k], yp_k) - u[k]
            gp = 1/T**2 + nums['df_dy'](y[k], yp_k) + nums['df_dyp'](y[k], yp_k) * 3/(2*T)
            y[k] = y[k] - g / gp

            # z2
            yp_k = (3*y[k] - 4*y[k-1] + y[k-2]) / (2*T)
            g  = (y[k] - 2*y[k-1] + y[k-2])/T**2 + nums['f'](y[k], yp_k) - u[k]
            gp = 1/T**2 + nums['df_dy'](y[k], yp_k) + nums['df_dyp'](y[k], yp_k) * 3/(2*T)
            gpp = (nums['d2f_dy2'](y[k], yp_k)
                   + 2 * nums['d2f_dydyp'](y[k], yp_k) * 3/(2*T)
                   + nums['d2f_dyp2'](y[k], yp_k) * (3/(2*T))**2)
            y[k] = y[k] - (1/2) * g**2 * gpp / gp**3

            # z3
            yp_k = (3*y[k] - 4*y[k-1] + y[k-2]) / (2*T)
            g  = (y[k] - 2*y[k-1] + y[k-2])/T**2 + nums['f'](y[k], yp_k) - u[k]
            gp = 1/T**2 + nums['df_dy'](y[k], yp_k) + nums['df_dyp'](y[k], yp_k) * 3/(2*T)
            gpp = (nums['d2f_dy2'](y[k], yp_k)
                   + 2 * nums['d2f_dydyp'](y[k], yp_k) * 3/(2*T)
                   + nums['d2f_dyp2'](y[k], yp_k) * (3/(2*T))**2)
            gppp = nums['d3f_dy3'](y[k], yp_k)
            y[k] = y[k] - (1/6) * g**3 * (-gppp * gp + 3 * gpp**2) / gp**5

        return y

    info = {**derivs, **{k+'_num': v for k, v in nums.items()},
            'y_sym': y_sym, 'yp_sym': yp_sym, 'order': 2}

    return regressor, info


# ============================================================
# Test
# ============================================================
if __name__ == "__main__":
    from sympy import Symbol, sin, cos
    from scipy.integrate import odeint

    # --- Test 1: y' + y² = sin(5t) ---
    print("=" * 60)
    print("Test 1: y' + y² = sin(5t)")
    print("=" * 60)
    y = Symbol('y')
    regressor, info = build_regressor_order1(y**2, y)

    n = 500; t = np.linspace(0, 10, n); T = t[1]-t[0]
    sol = odeint(lambda y, t: -y**2 + np.sin(5*t), -0.2, t).ravel()
    u = np.sin(5*t)
    yr = regressor(u, sol[0], sol[1], T, n)
    print(f"Error: {np.max(np.abs(yr - sol)):.2e}\n")

    # --- Test 2: y' + sin²(y) = sin(5t) ---
    print("=" * 60)
    print("Test 2: y' + sin²(y) = sin(5t)")
    print("=" * 60)
    regressor2, info2 = build_regressor_order1(sin(y)**2, y)

    sol2 = odeint(lambda y, t: -np.sin(y)**2 + np.sin(5*t), -0.2, t).ravel()
    yr2 = regressor2(u, sol2[0], sol2[1], T, n)
    print(f"Error: {np.max(np.abs(yr2 - sol2)):.2e}\n")

    # --- Test 3: y'' + 0.1y' + sin(y) = sin(3t) ---
    print("=" * 60)
    print("Test 3: y'' + 0.1y' + sin(y) = sin(3t)")
    print("=" * 60)
    yp = Symbol('yp')
    regressor3, info3 = build_regressor_order2(0.1*yp + sin(y), y, yp)

    n3 = 1000; t3 = np.linspace(0, 10, n3); T3 = t3[1]-t3[0]
    sol3 = odeint(lambda z, t: [z[1], -0.1*z[1] - np.sin(z[0]) + np.sin(3*t)],
                  [0.5, 0], t3)[:, 0]
    u3 = np.sin(3*t3)
    yr3 = regressor3(u3, sol3[0], sol3[1], T3, n3)
    print(f"Error: {np.max(np.abs(yr3 - sol3)):.2e}\n")
