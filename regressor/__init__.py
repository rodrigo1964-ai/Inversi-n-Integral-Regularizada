"""
homotopy_regressors - Nonlinear ODE solver via discrete homotopy series.
Author: Rodolfo H. Rodrigo - UNSJ
"""

from .solver import solve_order1, solve_order2, solve_order1_numeric
from .derivatives import discrete_derivatives

__version__ = "0.1.0"
