"""
parser.py - Traductor de ODE en texto a forma estándar del regresor.

Entrada (string):
    "3*y'' + 2*y' + y*sin(y) = cos(t)"

Salida:
    f_expr  = (2/3)*yp + (1/3)*y*sin(y)   (SymPy, en y, yp)
    u_expr  = cos(t)/3                      (SymPy, en t)
    order   = 2
    
Forma estándar:
    Orden 1:  y'  + f(y)    = u(t)
    Orden 2:  y'' + f(y,y') = u(t)

Convenciones de entrada:
    y''  o  y2   → segunda derivada
    y'   o  yp   → primera derivada
    y             → variable
    t             → tiempo (lado derecho)
    =             → separa ecuación

Ejemplo:
    >>> from homotopy_regressors.parser import parse_ode
    >>> f, u, order, info = parse_ode("3*y'' + 2*y' + y*sin(y) = cos(t)")
    >>> print(f"Orden {order}: y{'''*order} + {f} = {u}")
    Orden 2: y'' + 2*yp/3 + y*sin(y)/3 = cos(t)/3
"""

import re
from sympy import Symbol, symbols, sympify, cos, sin, exp, Rational, Abs
from sympy import collect, Poly, degree


def parse_ode(equation_str):
    """
    Parsea una ODE en string y devuelve forma estándar.
    
    Parameters
    ----------
    equation_str : str
        Ecuación en texto. Ejemplos:
            "y' + y**2 = sin(5*t)"
            "3*y'' + 2*y' + y*sin(y) = cos(t)"
            "y'' + 0.1*y' + sin(y) = sin(3*t)"
            "y' + sin(y)**2 = sin(5*t)"
    
    Returns
    -------
    f_expr : sympy.Expr
        Expresión f(y) o f(y,yp) en forma estándar
    u_expr : sympy.Expr
        Expresión u(t) de excitación
    order : int
        Orden de la ODE (1 o 2)
    info : dict
        Diccionario con detalles:
        - 'original': string original
        - 'coeff_max': coeficiente de la derivada mayor
        - 'lhs_normalized': lado izquierdo normalizado
        - 'y_sym', 'yp_sym', 'y2_sym', 't_sym': símbolos usados
    """
    y, yp, y2, t = symbols('y yp y2 t')
    
    original = equation_str.strip()
    
    # --- Paso 1: Separar por '=' ---
    if '=' not in equation_str:
        raise ValueError("La ecuación debe contener '='. Ejemplo: y' + y**2 = sin(5*t)")
    
    parts = equation_str.split('=')
    if len(parts) != 2:
        raise ValueError("La ecuación debe tener exactamente un '='")
    
    lhs_str = parts[0].strip()
    rhs_str = parts[1].strip()
    
    # --- Paso 2: Reemplazar notación de derivadas ---
    # y'' → y2, y' → yp (en ese orden para no confundir)
    lhs_str = _replace_derivatives(lhs_str)
    rhs_str = _replace_derivatives(rhs_str)
    
    # --- Paso 3: Parsear con SymPy ---
    local_dict = {
        'y': y, 'yp': yp, 'y2': y2, 't': t,
        'sin': sin, 'cos': cos, 'exp': exp, 'abs': Abs,
    }
    
    try:
        lhs_expr = sympify(lhs_str, locals=local_dict)
    except Exception as e:
        raise ValueError(f"Error parseando lado izquierdo '{lhs_str}': {e}")
    
    try:
        rhs_expr = sympify(rhs_str, locals=local_dict)
    except Exception as e:
        raise ValueError(f"Error parseando lado derecho '{rhs_str}': {e}")
    
    # --- Paso 4: Detectar orden ---
    has_y2 = lhs_expr.has(y2)
    has_yp = lhs_expr.has(yp)
    
    if has_y2:
        order = 2
        max_deriv = y2
    elif has_yp:
        order = 1
        max_deriv = yp
    else:
        raise ValueError("No se detectó y' ni y'' en el lado izquierdo")
    
    # --- Paso 5: Extraer coeficiente de la derivada mayor ---
    # Separar: lhs = coeff * max_deriv + resto
    coeff_max = _extract_coefficient(lhs_expr, max_deriv)
    
    if coeff_max == 0:
        raise ValueError(f"Coeficiente de {max_deriv} es cero")
    
    # --- Paso 6: Normalizar (dividir todo por coeff_max) ---
    lhs_normalized = lhs_expr / coeff_max
    rhs_normalized = rhs_expr / coeff_max
    
    # Simplificar
    from sympy import simplify, expand
    lhs_normalized = simplify(expand(lhs_normalized))
    rhs_normalized = simplify(expand(rhs_normalized))
    
    # --- Paso 7: Separar f de la derivada mayor ---
    # lhs_normalized = max_deriv + f(y, yp)
    # Entonces f = lhs_normalized - max_deriv
    f_expr = simplify(lhs_normalized - max_deriv)
    u_expr = rhs_normalized
    
    # --- Paso 8: Verificar que u no dependa de y, yp ---
    if u_expr.has(y) or u_expr.has(yp) or u_expr.has(y2):
        raise ValueError(
            f"El lado derecho u(t) = {u_expr} no debe depender de y.\n"
            f"Pasá los términos con y al lado izquierdo."
        )
    
    # --- Paso 9: Si orden 1, verificar que f no tenga yp residual ---
    if order == 1 and f_expr.has(y2):
        raise ValueError("ODE de orden 1 pero f contiene y''")
    
    info = {
        'original': original,
        'coeff_max': coeff_max,
        'lhs_normalized': lhs_normalized,
        'y_sym': y,
        'yp_sym': yp,
        'y2_sym': y2,
        't_sym': t,
        'standard_form': f"y{'pp' if order==2 else 'p'} + ({f_expr}) = {u_expr}",
    }
    
    return f_expr, u_expr, order, info


def parse_and_build(equation_str):
    """
    Parsea la ODE y construye el regresor directamente.
    
    Parameters
    ----------
    equation_str : str
        Ecuación en texto
    
    Returns
    -------
    regressor : callable
        Función regressor(u_array, y0, y1, T, n) → array solución
    info : dict
        Información completa (parser + regressor)
    
    Ejemplo
    -------
    >>> reg, info = parse_and_build("3*y'' + 2*y' + y*sin(y) = cos(t)")
    >>> # Definir excitación
    >>> import numpy as np
    >>> t = np.linspace(0, 10, 1000)
    >>> T = t[1] - t[0]
    >>> u = np.cos(t) / 3  # ¡ya normalizada!
    >>> sol = reg(u, 0.0, 0.0, T, len(t))
    """
    from .regressor import build_regressor_order1, build_regressor_order2
    
    f_expr, u_expr, order, parse_info = parse_ode(equation_str)
    
    y = parse_info['y_sym']
    yp = parse_info['yp_sym']
    
    if order == 1:
        regressor, reg_info = build_regressor_order1(f_expr, y)
    else:
        regressor, reg_info = build_regressor_order2(f_expr, y, yp)
    
    # Combinar info
    full_info = {**parse_info, **reg_info}
    full_info['u_expr'] = u_expr
    full_info['f_expr'] = f_expr
    full_info['order'] = order
    
    return regressor, full_info


def show(equation_str):
    """
    Muestra la traducción de la ecuación a forma estándar.
    Solo imprime, no construye regresor.
    
    >>> show("3*y'' + 2*y' + y*sin(y) = cos(t)")
    
    Original:   3*y'' + 2*y' + y*sin(y) = cos(t)
    Orden:      2
    Coef. máx:  3
    
    Forma estándar:
      y'' + (2*yp/3 + y*sin(y)/3) = cos(t)/3
    
    Para el regresor:
      f(y, yp) = 2*yp/3 + y*sin(y)/3
      u(t)     = cos(t)/3
    """
    f_expr, u_expr, order, info = parse_ode(equation_str)
    
    y = info['y_sym']
    yp = info['yp_sym']
    
    deriv_str = "y''" if order == 2 else "y'"
    vars_str = "y, y'" if order == 2 else "y"
    
    print(f"  Original:   {info['original']}")
    print(f"  Orden:      {order}")
    print(f"  Coef. máx:  {info['coeff_max']}")
    print()
    print(f"  Forma estándar:")
    print(f"    {deriv_str} + ({f_expr}) = {u_expr}")
    print()
    print(f"  Para el regresor:")
    print(f"    f({vars_str}) = {f_expr}")
    print(f"    u(t)     = {u_expr}")


# ==================== Funciones internas ====================

def _replace_derivatives(s):
    """
    Reemplaza notación de derivadas en string.
    
    y'''  → no soportado (solo hasta orden 2)
    y''   → y2
    y'    → yp
    
    Maneja casos como:
        3*y''  → 3*y2
        -y'    → -yp
        y'**2  → yp**2
    """
    # Primero y'' → y2 (antes que y' para no confundir)
    s = re.sub(r"y''", 'y2', s)
    
    # Luego y' → yp
    s = re.sub(r"y'", 'yp', s)
    
    return s


def _extract_coefficient(expr, symbol):
    """
    Extrae el coeficiente lineal de `symbol` en `expr`.
    
    Para expr = 3*y2 + 2*yp + y*sin(y):
        _extract_coefficient(expr, y2) → 3
    
    Funciona incluso si symbol aparece multiplicado por constantes
    pero no dentro de funciones.
    """
    from sympy import collect, Add
    
    # Expandir y colectar por el símbolo
    collected = collect(expr.expand(), symbol)
    
    # Si es una suma, buscar el término con symbol
    if isinstance(collected, Add):
        for term in collected.args:
            coeff = term.as_coefficient(symbol)
            if coeff is not None:
                return coeff
    else:
        coeff = collected.as_coefficient(symbol)
        if coeff is not None:
            return coeff
    
    # Fallback: derivada
    from sympy import diff
    coeff = diff(expr, symbol)
    # Verificar que sea constante respecto a symbol
    if not coeff.has(symbol):
        return coeff
    
    raise ValueError(
        f"No pude extraer coeficiente lineal de {symbol} en {expr}.\n"
        f"La derivada mayor debe aparecer linealmente (no dentro de sin, exp, etc.)"
    )


# ==================== Pruebas ====================

if __name__ == '__main__':
    
    tests = [
        "y' + y**2 = sin(5*t)",
        "y' + sin(y)**2 = sin(5*t)",
        "3*y'' + 2*y' + y*sin(y) = cos(t)",
        "y'' + 0.1*y' + sin(y) = sin(3*t)",
        "y'' + 0.5*y'*(y**2 - 1) + y = sin(t)",
    ]
    
    print("=" * 60)
    print("TRADUCTOR DE ODE → FORMA ESTÁNDAR")
    print("=" * 60)
    
    for eq in tests:
        print()
        print(f"{'─' * 50}")
        show(eq)
