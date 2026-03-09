"""
derivatives.py - Discrete derivative approximations via Taylor matrix inversion.

Constructs the Taylor-based Vandermonde matrix:

    A[i,j] = (-(i-1)*T)^(j-1) / (j-1)!

where row i corresponds to sample y_{k-i+1} and column j to the j-th derivative.
Inverting A yields backward finite difference formulas for y'_k, y''_k, etc.

Based on: builMatrizDerivadas.mw (Maple worksheet)
"""

from sympy import Symbol, Matrix, factorial, simplify, Rational


def build_taylor_matrix(n, T=None):
    """
    Build the n x n Taylor matrix for backward differences.
    
    A[i,j] = (-(i-1)*T)^(j-1) / (j-1)!
    
    Parameters
    ----------
    n : int
        Number of sample points (2, 3, 4, ...).
    T : Symbol or None
        Sampling period. If None, creates Symbol('T').
    
    Returns
    -------
    A : sympy.Matrix
        The n x n Taylor matrix.
    T : Symbol
        The sampling period symbol used.
    """
    if T is None:
        T = Symbol('T')
    
    A = Matrix(n, n, lambda i, j: (-(i) * T)**j / factorial(j))
    return A, T


def discrete_derivatives(n, T=None):
    """
    Compute discrete derivative formulas using n backward points.
    
    Given n sample points y_k, y_{k-1}, ..., y_{k-n+1}, returns
    symbolic expressions for y'_k, y''_k, ... y^(n-1)_k.
    
    Parameters
    ----------
    n : int
        Number of sample points (2, 3, or 4).
    T : Symbol or None
        Sampling period. If None, creates Symbol('T').
    
    Returns
    -------
    dict
        Dictionary with keys 1, 2, ..., n-1 mapping derivative order
        to its symbolic expression in terms of y_k, y_{k-1}, etc.
    
    Examples
    --------
    >>> d = discrete_derivatives(3)
    >>> print(d[1])  # y'_k  = (3*y_k - 4*y_{k-1} + y_{k-2}) / (2*T)
    >>> print(d[2])  # y''_k = (y_k - 2*y_{k-1} + y_{k-2}) / T**2
    """
    A, T = build_taylor_matrix(n, T)
    
    # Sample vector: [y_k, y_{k-1}, ..., y_{k-n+1}]
    y = _make_sample_vector(n)
    
    # Invert and multiply: [y_k, y'_k, y''_k, ...] = A^{-1} * [y_k, y_{k-1}, ...]
    A_inv = A.inv()
    derivs_vec = simplify(A_inv * y)
    
    # Return derivative orders 1 through n-1
    result = {}
    for order in range(1, n):
        result[order] = simplify(derivs_vec[order])
    
    return result


def _make_sample_vector(n):
    """Create the sample vector [y_k, y_{k-1}, ..., y_{k-n+1}]."""
    from sympy import symbols
    
    names = ['y_k'] + [f'y_{{k-{i}}}' for i in range(1, n)]
    syms = symbols(' '.join(names))
    
    if n == 1:
        return Matrix([syms])
    return Matrix(list(syms))


def print_formulas(n, T=None):
    """
    Print all derivative formulas for n points.
    
    Parameters
    ----------
    n : int
        Number of sample points.
    T : Symbol or None
        Sampling period.
    """
    d = discrete_derivatives(n, T)
    
    labels = {1: "y'_k", 2: "y''_k", 3: "y'''_k", 4: "y''''_k"}
    
    print(f"{n} puntos:")
    for order, expr in d.items():
        label = labels.get(order, f"y^({order})_k")
        print(f"  {label:8s} = {expr}")
