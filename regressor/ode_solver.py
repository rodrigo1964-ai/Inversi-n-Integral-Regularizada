"""
ode_solver.py - Homotopy series solver for nonlinear ODEs.

Solves dy/dt + f(y) = u(t) using backward finite differences
(3 or 4 points) combined with Newton homotopy series.

The discretized equation g(y_k) = 0 is solved via:
    y_k = y0 + z1 + z2 + z3 + ...

where each z_m is computed from the HAM expansion.

For linear f, z1 gives the exact solution.
For nonlinear f, each additional term corrects the nonlinearity.
"""

from sympy import (
    Symbol, Function, symbols, Derivative,
    Rational, factorial, simplify, together
)


def build_ode1_regressors(n_points=3, n_terms=3):
    """
    Build symbolic regressors for dy/dt + f(y) = u(t).
    
    Parameters
    ----------
    n_points : int
        Number of backward points (3 or 4).
    n_terms : int
        Number of homotopy terms (1, 2, or 3).
    
    Returns
    -------
    dict with keys:
        'z' : list of symbolic expressions [z1, z2, ...] 
        'y_k' : symbolic expression y0 + z1 + z2 + ...
        'symbols' : dict of symbols used
    """
    T = Symbol('T')
    y0 = Symbol('y0')
    uk = Symbol('u_k')
    f = Function('f')
    
    if n_points == 3:
        yk1, yk2 = symbols('y_{k-1} y_{k-2}')
        # y'_k = (3*y_k - 4*y_{k-1} + y_{k-2}) / (2T)
        # g(y_k) = 3*y_k/(2T) + f(y_k) - u_k - (4*y_{k-1} - y_{k-2})/(2T)
        g_y0 = 3*y0/(2*T) + f(y0) - uk - (4*yk1 - yk2)/(2*T)
        gp_y0 = Rational(3, 2)/T + Derivative(f(y0), y0)
        samples = {'y_{k-1}': yk1, 'y_{k-2}': yk2}
        
    elif n_points == 4:
        yk1, yk2, yk3 = symbols('y_{k-1} y_{k-2} y_{k-3}')
        # y'_k = (11*y_k - 18*y_{k-1} + 9*y_{k-2} - 2*y_{k-3}) / (6T)
        # g(y_k) = 11*y_k/(6T) + f(y_k) - u_k - (18*y_{k-1} - 9*y_{k-2} + 2*y_{k-3})/(6T)
        g_y0 = 11*y0/(6*T) + f(y0) - uk - (18*yk1 - 9*yk2 + 2*yk3)/(6*T)
        gp_y0 = Rational(11, 6)/T + Derivative(f(y0), y0)
        samples = {'y_{k-1}': yk1, 'y_{k-2}': yk2, 'y_{k-3}': yk3}
    else:
        raise ValueError("n_points must be 3 or 4")
    
    gpp_y0 = Derivative(f(y0), (y0, 2))
    gppp_y0 = Derivative(f(y0), (y0, 3))
    
    # z1 = -g / g'
    z1 = simplify(-g_y0 / gp_y0)
    
    terms = [z1]
    
    if n_terms >= 2:
        # z2 = -g² · g'' / (2 · g'³)
        z2 = simplify(-g_y0**2 * gpp_y0 / (2 * gp_y0**3))
        terms.append(z2)
    
    if n_terms >= 3:
        # z3 = (g'·g''' - 3·g''²) · g³ / (6 · g'⁵)
        z3 = simplify(
            (gp_y0 * gppp_y0 - 3 * gpp_y0**2) * g_y0**3
            / (6 * gp_y0**5)
        )
        terms.append(z3)
    
    yk_expr = y0
    for z in terms:
        yk_expr = yk_expr + z
    yk_expr = simplify(yk_expr)
    
    syms = {
        'T': T, 'y0': y0, 'u_k': uk, 'f': f,
        **samples
    }
    
    return {
        'z': terms,
        'y_k': yk_expr,
        'symbols': syms
    }


def ode1_step_3pt(yk1, yk2, uk, T, f, df, d2f, d3f, y0=None, n_terms=3):
    """
    Compute y_k for dy/dt + f(y) = u(t) using 3 backward points.
    
    Parameters
    ----------
    yk1 : float
        y_{k-1}
    yk2 : float
        y_{k-2}
    uk : float
        u(t_k)
    T : float
        Sampling period.
    f : callable
        f(y) -> float
    df : callable
        f'(y) -> float
    d2f : callable
        f''(y) -> float
    d3f : callable
        f'''(y) -> float
    y0 : float or None
        Initial approximation. If None, uses y_{k-1}.
    n_terms : int
        Number of homotopy terms (1, 2, or 3).
    
    Returns
    -------
    float
        y_k approximation.
    """
    if y0 is None:
        y0 = yk1
    
    g = 3*y0/(2*T) + f(y0) - uk - (4*yk1 - yk2)/(2*T)
    gp = 3/(2*T) + df(y0)
    
    z1 = -g / gp
    yk = y0 + z1
    
    if n_terms >= 2:
        gpp = d2f(y0)
        z2 = -g**2 * gpp / (2 * gp**3)
        yk += z2
    
    if n_terms >= 3:
        gpp = d2f(y0)
        gppp = d3f(y0)
        z3 = (gp * gppp - 3 * gpp**2) * g**3 / (6 * gp**5)
        yk += z3
    
    return yk


def ode1_step_4pt(yk1, yk2, yk3, uk, T, f, df, d2f, d3f, y0=None, n_terms=3):
    """
    Compute y_k for dy/dt + f(y) = u(t) using 4 backward points.
    
    Parameters
    ----------
    yk1 : float
        y_{k-1}
    yk2 : float
        y_{k-2}
    yk3 : float
        y_{k-3}
    uk : float
        u(t_k)
    T : float
        Sampling period.
    f, df, d2f, d3f : callable
        f(y), f'(y), f''(y), f'''(y)
    y0 : float or None
        Initial approximation. If None, uses y_{k-1}.
    n_terms : int
        Number of homotopy terms (1, 2, or 3).
    
    Returns
    -------
    float
        y_k approximation.
    """
    if y0 is None:
        y0 = yk1
    
    g = 11*y0/(6*T) + f(y0) - uk - (18*yk1 - 9*yk2 + 2*yk3)/(6*T)
    gp = 11/(6*T) + df(y0)
    
    z1 = -g / gp
    yk = y0 + z1
    
    if n_terms >= 2:
        gpp = d2f(y0)
        z2 = -g**2 * gpp / (2 * gp**3)
        yk += z2
    
    if n_terms >= 3:
        gpp = d2f(y0)
        gppp = d3f(y0)
        z3 = (gp * gppp - 3 * gpp**2) * g**3 / (6 * gp**5)
        yk += z3
    
    return yk


def solve_ode1(t_span, y_initial, u_func, T, f, df, d2f, d3f,
               n_points=3, n_terms=3):
    """
    Integrate dy/dt + f(y) = u(t) over a time span.
    
    Parameters
    ----------
    t_span : tuple
        (t_start, t_end)
    y_initial : list
        Initial values [y_0, y_1] for 3 points or [y_0, y_1, y_2] for 4 points.
    u_func : callable
        u(t) -> float
    T : float
        Sampling period.
    f, df, d2f, d3f : callable
        f(y), f'(y), f''(y), f'''(y)
    n_points : int
        Number of backward points (3 or 4).
    n_terms : int
        Number of homotopy terms (1, 2, or 3).
    
    Returns
    -------
    t_values : list
        Time points.
    y_values : list
        Solution values.
    """
    t_start, t_end = t_span
    n_init = len(y_initial)
    
    t_values = [t_start + i * T for i in range(n_init)]
    y_values = list(y_initial)
    
    k = n_init
    while t_values[-1] < t_end - T/2:
        t_k = t_start + k * T
        uk = u_func(t_k)
        
        if n_points == 3:
            yk = ode1_step_3pt(
                y_values[k-1], y_values[k-2],
                uk, T, f, df, d2f, d3f, n_terms=n_terms
            )
        elif n_points == 4:
            yk = ode1_step_4pt(
                y_values[k-1], y_values[k-2], y_values[k-3],
                uk, T, f, df, d2f, d3f, n_terms=n_terms
            )
        else:
            raise ValueError("n_points must be 3 or 4")
        
        y_values.append(yk)
        t_values.append(t_k)
        k += 1
    
    return t_values, y_values
