"""
regressor_system.py - Symbolic regressor builder for coupled ODE systems.

Automatically generates homotopy regressor from symbolic expressions of F, G, H
using SymPy for automatic differentiation.
Jacobian, Hessian, and third-order tensor are ALL computed symbolically
with full chain-rule composition for discrete derivatives.

Author: Rodolfo H. Rodrigo - UNSJ
"""

from sympy import Symbol, diff, simplify, lambdify, S
import numpy as np
from solver_system import solve_system


def build_system_regressor(func_exprs, state_syms, order=1):
    """
    Build homotopy regressor for a system of N coupled ODEs.

    Computes symbolically:
      - Jacobian  J[i][j]       = dgi/dqj[k]       (with chain rule)
      - Hessian   H[i][j][l]    = d²gi/dqj dql[k]  (with chain rule)
      - Tensor    T[i][j][l][m] = d³gi/dqj dql dqm  (with chain rule)

    The chain rule for discrete derivatives maps each ∂/∂q_j[k] to:
        ∂/∂q_j + (3/(2T)) · ∂/∂q'_j + (1/T²) · ∂/∂q''_j

    Parameters
    ----------
    func_exprs : list of sympy.Expr
        [F_expr, G_expr, H_expr] in terms of state symbols.

    state_syms : list of sympy.Symbol
        Always 10 symbols: [x, y, z, xp, yp, zp, xpp, ypp, zpp, t]

    order : int
        Maximum order of derivatives present (1 or 2).

    Returns
    -------
    regressor : callable with signature (excitations, initial_conditions, T, n)
    info : dict with all symbolic and numeric derivatives
    """
    N = len(func_exprs)

    # Extraer simbolos de variables.
    # state_syms siempre tiene estructura fija de 10 elementos:
    #   [x, y, z, xp, yp, zp, xpp, ypp, zpp, t]
    # Las posiciones son fijas: var=[0:3], varp=[3:6], varpp=[6:9], t=[9]
    var_syms = state_syms[0:N]     # [x, y] o [x, y, z]
    varp_syms = state_syms[3:3+N]  # [xp, yp] o [xp, yp, zp]
    if order == 2:
        varpp_syms = state_syms[6:6+N]  # [xpp, ypp] o [xpp, ypp, zpp]
    else:
        varpp_syms = [Symbol('_dummy_pp_%d' % i) for i in range(N)]  # no aparecen

    t_sym = state_syms[-1]
    all_syms = state_syms  # 10 simbolos para lambdify

    print("=" * 60)
    print("Building system regressor (fully symbolic)")
    print("=" * 60)
    print(f"Number of equations: {N}")
    print(f"Order: {order}")
    print(f"Variables: {var_syms}")
    print()

    # ------------------------------------------------------------------
    # Operador derivada discreta D_j[expr]:
    #   D_j[expr] = d(expr)/dq_j + c1 * d(expr)/dqp_j + c2 * d(expr)/dqpp_j
    # donde c1 = 3/(2T) y c2 = 1/T^2 se aplican numericamente despues.
    # Guardamos las tres derivadas parciales por separado.
    # ------------------------------------------------------------------

    def _chain_rule_partials(expr, j, do_simplify=True):
        """Retorna (d_expr/dq_j, d_expr/dqp_j, d_expr/dqpp_j)."""
        _simp = simplify if do_simplify else lambda e: e
        d_q = _simp(diff(expr, var_syms[j]))
        d_qp = _simp(diff(expr, varp_syms[j]))
        if order == 2:
            d_qpp = _simp(diff(expr, varpp_syms[j]))
        else:
            d_qpp = S.Zero
        return d_q, d_qp, d_qpp

    # ------------------------------------------------------------------
    # JACOBIANO simbolico: J_ij = D_j[F_i]
    # Almacenamos las 3 derivadas parciales de F_i respecto a q_j, qp_j, qpp_j
    # ------------------------------------------------------------------
    print("Computing Jacobian (symbolic)...")

    # jac_parts[i][j] = (dFi/dqj, dFi/dqpj, dFi/dqppj) como expresiones sympy
    jac_parts = []
    for i in range(N):
        row = []
        for j in range(N):
            row.append(_chain_rule_partials(func_exprs[i], j))
        jac_parts.append(row)

    # Expresion simbolica combinada del Jacobiano (con c1, c2 simbolicos para info)
    # J_ij_expr = dFi/dqj + c1*dFi/dqpj + c2*dFi/dqppj
    # Pero para lambdify guardamos las 3 partes por separado

    # Lambdify de las partes del Jacobiano
    jac_nums = []
    for i in range(N):
        row = []
        for j in range(N):
            d_q, d_qp, d_qpp = jac_parts[i][j]
            row.append((
                lambdify(all_syms, d_q, modules='numpy'),
                lambdify(all_syms, d_qp, modules='numpy'),
                lambdify(all_syms, d_qpp, modules='numpy'),
            ))
        jac_nums.append(row)

    print("  Jacobian OK.")

    # ------------------------------------------------------------------
    # HESSIANO simbolico: H_ijl = D_l[ J_ij ]
    #
    # J_ij es una funcion de (q, qp, qpp), y D_l aplica la regla de la
    # cadena otra vez. Como J_ij = dFi/dqj + c1*dFi/dqpj + c2*dFi/dqppj,
    # necesitamos D_l de cada una de las 3 partes.
    #
    # H_ijl = D_l[dFi/dqj] + c1 * D_l[dFi/dqpj] + c2 * D_l[dFi/dqppj]
    #
    # Cada D_l[·] a su vez produce 3 derivadas parciales que se combinan
    # con c1, c2 en runtime. En total tenemos 3 x 3 = 9 derivadas parciales
    # por componente (i,j,l).
    # ------------------------------------------------------------------
    print("Computing Hessian (symbolic)...")

    # hess_parts[i][j][l] = lista de 9 expresiones sympy
    # Organizadas como 3 bloques (uno por cada parte del Jacobiano):
    #   bloque_q:   D_l[dFi/dqj]   -> (d2Fi/dqj_dql, d2Fi/dqj_dqpl, d2Fi/dqj_dqppl)
    #   bloque_qp:  D_l[dFi/dqpj]  -> (d2Fi/dqpj_dql, d2Fi/dqpj_dqpl, d2Fi/dqpj_dqppl)
    #   bloque_qpp: D_l[dFi/dqppj] -> (d2Fi/dqppj_dql, d2Fi/dqppj_dqpl, d2Fi/dqppj_dqppl)
    hess_parts = []
    for i in range(N):
        hess_i = []
        for j in range(N):
            hess_ij = []
            d_q, d_qp, d_qpp = jac_parts[i][j]
            for l in range(N):
                # D_l aplicado a cada parte del Jacobiano (sin simplify extra)
                blk_q = _chain_rule_partials(d_q, l, do_simplify=False)
                blk_qp = _chain_rule_partials(d_qp, l, do_simplify=False)
                blk_qpp = _chain_rule_partials(d_qpp, l, do_simplify=False)
                hess_ij.append((blk_q, blk_qp, blk_qpp))
            hess_i.append(hess_ij)
        hess_parts.append(hess_i)

    print("  Hessian OK.")

    # ------------------------------------------------------------------
    # TENSOR de 3er orden simbolico: T_ijlm = D_m[ H_ijl ]
    #
    # H_ijl se combina con c1, c2 de forma cuadratica. La estructura
    # completa tiene 27 derivadas parciales por componente (i,j,l,m).
    # Para evitar explosion combinatoria, usamos un enfoque mas compacto:
    #
    # Definimos la expresion simbolica combinada del Jacobiano usando
    # simbolos auxiliares para c1, c2, y luego diferenciamos 2 veces mas.
    # ------------------------------------------------------------------
    print("Computing third-order tensor (symbolic)...")

    # Simbolos auxiliares para los coeficientes de la regla de la cadena
    _c1 = Symbol('_c1')  # sera 3/(2T)
    _c2 = Symbol('_c2')  # sera 1/T^2

    # Expresion simbolica completa del Jacobiano combinado
    # Jij_full = dFi/dqj + _c1*dFi/dqpj + _c2*dFi/dqppj
    jac_full_exprs = []
    for i in range(N):
        row = []
        for j in range(N):
            d_q, d_qp, d_qpp = jac_parts[i][j]
            row.append(d_q + _c1 * d_qp + _c2 * d_qpp)
        jac_full_exprs.append(row)

    # Operador D_l completo (con _c1, _c2) sobre una expresion
    def _D_l(expr, l):
        return (diff(expr, var_syms[l])
                + _c1 * diff(expr, varp_syms[l])
                + _c2 * diff(expr, varpp_syms[l]) if order == 2
                else diff(expr, var_syms[l]) + _c1 * diff(expr, varp_syms[l]))

    # Corregir: para orden 1 no hay varpp en las expresiones reales,
    # pero varpp_syms son dummies que no aparecen, asi que diff da 0.
    # Podemos usar la version general siempre.
    def _D_full(expr, l):
        """Operador D_l completo con _c1, _c2 simbolicos."""
        return (diff(expr, var_syms[l])
                + _c1 * diff(expr, varp_syms[l])
                + _c2 * diff(expr, varpp_syms[l]))

    # Hessiano simbolico completo (con _c1, _c2)
    hess_full_exprs = []
    for i in range(N):
        hess_i = []
        for j in range(N):
            hess_ij = []
            for l in range(N):
                expr = _D_full(jac_full_exprs[i][j], l)
                hess_ij.append(expr)
            hess_i.append(hess_ij)
        hess_full_exprs.append(hess_i)

    # Tensor simbolico completo: T_ijlm = D_m[ H_ijl ]
    tens_full_exprs = []
    tens_all_zero = True
    for i in range(N):
        tens_i = []
        for j in range(N):
            tens_ij = []
            for l in range(N):
                tens_ijl = []
                for m in range(N):
                    expr = _D_full(hess_full_exprs[i][j][l], m)
                    tens_ijl.append(expr)
                    if expr != S.Zero:
                        tens_all_zero = False
                tens_ij.append(tens_ijl)
            tens_i.append(tens_ij)
        tens_full_exprs.append(tens_i)

    # Lambdify del tensor (sustituir _c1, _c2 en runtime)
    # Para evaluar: sustituimos _c1 -> 3/(2T), _c2 -> 1/T^2, luego evaluamos
    # Usamos all_syms + [_c1, _c2] como argumentos del lambdify
    all_syms_ext = list(all_syms) + [_c1, _c2]

    tens_nums = None
    if not tens_all_zero:
        tens_nums = []
        for i in range(N):
            tens_i = []
            for j in range(N):
                tens_ij = []
                for l in range(N):
                    tens_ijl = []
                    for m in range(N):
                        tens_ijl.append(
                            lambdify(all_syms_ext, tens_full_exprs[i][j][l][m],
                                     modules='numpy')
                        )
                    tens_ij.append(tens_ijl)
                tens_i.append(tens_ij)
            tens_nums.append(tens_i)

    if tens_all_zero:
        print("  Tensor: all components are zero, will skip z3 correction.")
    else:
        print("  Tensor OK (non-zero components found).")

    # Lambdify del Hessiano completo (con _c1, _c2)
    hess_full_nums = []
    for i in range(N):
        hess_i = []
        for j in range(N):
            hess_ij = []
            for l in range(N):
                hess_ij.append(
                    lambdify(all_syms_ext, hess_full_exprs[i][j][l],
                             modules='numpy')
                )
            hess_i.append(hess_ij)
        hess_full_nums.append(hess_i)

    # Lambdify funciones principales
    func_nums = [lambdify(all_syms, simplify(f), modules='numpy') for f in func_exprs]

    print()

    # ------------------------------------------------------------------
    # Construir el regressor callable
    # ------------------------------------------------------------------
    def regressor(excitations, initial_conditions, T, n):
        """
        Solve the system using fully symbolic derivatives.

        Parameters
        ----------
        excitations : list of np.ndarray
        initial_conditions : list of list of float
        T : float
        n : int

        Returns
        -------
        results : list of np.ndarray
        """
        c1 = 3.0 / (2.0 * T)
        c2 = 1.0 / (T ** 2)

        # --- Jacobiano: combinar las 3 partes con c1, c2 ---
        def make_jac_func(i, j):
            f_q, f_qp, f_qpp = jac_nums[i][j]

            def jac_ij(*args):
                return f_q(*args) + c1 * f_qp(*args) + c2 * f_qpp(*args)

            return jac_ij

        jac_funcs = [[make_jac_func(i, j) for j in range(N)] for i in range(N)]

        # --- Hessiano: evaluar expresion completa con c1, c2 ---
        def make_hess_func(i, j, l):
            f_hijl = hess_full_nums[i][j][l]

            def hess_ijl(*args):
                # Agregar c1, c2 al final de los argumentos
                return f_hijl(*args, c1, c2)

            return hess_ijl

        hess_funcs = [[[make_hess_func(i, j, l) for l in range(N)]
                       for j in range(N)] for i in range(N)]

        # --- Tensor de 3er orden ---
        if tens_nums is not None:
            def make_tens_func(i, j, l, m):
                f_tijlm = tens_nums[i][j][l][m]

                def tens_ijlm(*args):
                    return f_tijlm(*args, c1, c2)

                return tens_ijlm

            t_funcs = [[[[make_tens_func(i, j, l, m) for m in range(N)]
                         for l in range(N)]
                        for j in range(N)] for i in range(N)]
        else:
            t_funcs = None

        # Llamar a solve_system con derivadas simbolicas exactas
        return solve_system(func_nums, jac_funcs, hess_funcs, t_funcs,
                            excitations, initial_conditions, T, n)

    # ------------------------------------------------------------------
    # Info para inspeccion
    # ------------------------------------------------------------------
    info = {
        'N': N,
        'order': order,
        'func_exprs': func_exprs,
        'jac_parts': jac_parts,
        'hess_full_exprs': hess_full_exprs,
        'tens_full_exprs': tens_full_exprs,
        'tens_all_zero': tens_all_zero,
        'func_nums': func_nums,
        'jac_nums': jac_nums,
        'hess_full_nums': hess_full_nums,
        'tens_nums': tens_nums,
        'var_syms': var_syms,
        'varp_syms': varp_syms,
        'varpp_syms': varpp_syms if order == 2 else None,
        't_sym': t_sym,
        '_c1': _c1,
        '_c2': _c2,
    }

    print("=" * 60)
    print("Regressor built successfully! (fully symbolic)")
    print("=" * 60)
    print()

    return regressor, info


# ============================================================
# Test
# ============================================================
if __name__ == "__main__":
    from sympy import symbols
    from scipy.integrate import solve_ivp

    # ---- Test 1: Lotka-Volterra (1er orden, bilineal) ----
    print("\n" + "=" * 60)
    print("Test 1: Lotka-Volterra (1st order, bilinear)")
    print("x' = x - 0.1*x*y")
    print("y' = 0.075*x*y - 1.5*y")
    print("=" * 60)

    x, y, z = symbols('x y z')
    xp, yp, zp = symbols('xp yp zp')
    xpp, ypp, zpp = symbols('xpp ypp zpp')
    t = Symbol('t')

    F_expr = xp - x + 0.1*x*y
    G_expr = yp - 0.075*x*y + 1.5*y

    state_syms = [x, y, z, xp, yp, zp, xpp, ypp, zpp, t]
    regressor, info = build_system_regressor([F_expr, G_expr], state_syms, order=1)

    t_span = (0, 30)
    n_points = 5000
    t_arr = np.linspace(t_span[0], t_span[1], n_points)
    T_step = t_arr[1] - t_arr[0]

    def lv_rhs(t, q):
        x, y = q
        return [x - 0.1*x*y, 0.075*x*y - 1.5*y]

    sol_ref = solve_ivp(lv_rhs, t_span, [10.0, 5.0], t_eval=t_arr,
                        method='RK45', rtol=1e-9, atol=1e-9)

    u = np.zeros(n_points)
    v = np.zeros(n_points)
    ic = [[10.0, sol_ref.y[0, 1]], [5.0, sol_ref.y[1, 1]]]

    results = regressor([u, v], ic, T_step, n_points)

    err_x = np.max(np.abs(results[0] - sol_ref.y[0]))
    err_y = np.max(np.abs(results[1] - sol_ref.y[1]))

    print(f"Error en x: {err_x:.6e}")
    print(f"Error en y: {err_y:.6e}")
    print(f"{'✓ PASS' if (err_x < 1e-2 and err_y < 1e-2) else '✗ FAIL'}")

    # ---- Test 2: Duffing acoplado (2do orden, cubico) ----
    print("\n" + "=" * 60)
    print("Test 2: Coupled Duffing (2nd order, cubic)")
    print("x'' + 0.1*x' + x + 0.2*x^3 + 0.5*(x-y) = 0.5*cos(1.2*t)")
    print("y'' + 0.1*y' + y + 0.2*y^3 - 0.5*(x-y) = 0")
    print("=" * 60)

    from sympy import cos as sym_cos

    F2_expr = xpp + 0.1*xp + x + 0.2*x**3 + 0.5*(x - y)
    G2_expr = ypp + 0.1*yp + y + 0.2*y**3 - 0.5*(x - y)

    regressor2, info2 = build_system_regressor([F2_expr, G2_expr], state_syms, order=2)

    # Verificar expresiones del tensor
    print("\nTensor expressions (non-zero):")
    for i in range(2):
        for j in range(2):
            for l in range(2):
                for m in range(2):
                    expr = info2['tens_full_exprs'][i][j][l][m]
                    if expr != S.Zero:
                        print(f"  T[{i}][{j}][{l}][{m}] = {expr}")

    t_span2 = (0, 50)
    n2 = 500000
    t_arr2 = np.linspace(t_span2[0], t_span2[1], n2)
    T2 = t_arr2[1] - t_arr2[0]

    def duffing_rhs(t, q):
        x, xp, y, yp = q
        xpp = -0.1*xp - x - 0.2*x**3 - 0.5*(x-y) + 0.5*np.cos(1.2*t)
        ypp = -0.1*yp - y - 0.2*y**3 + 0.5*(x-y)
        return [xp, xpp, yp, ypp]

    sol_ref2 = solve_ivp(duffing_rhs, t_span2, [0.0, 0.5, 0.2, 0.0],
                         t_eval=t_arr2, method='RK45', rtol=1e-9, atol=1e-9)

    u2 = 0.5 * np.cos(1.2 * t_arr2)
    v2 = np.zeros(n2)
    ic2 = [[0.0, sol_ref2.y[0, 1]], [0.2, sol_ref2.y[2, 1]]]

    results2 = regressor2([u2, v2], ic2, T2, n2)

    err_x2 = np.max(np.abs(results2[0] - sol_ref2.y[0]))
    err_y2 = np.max(np.abs(results2[1] - sol_ref2.y[2]))

    print(f"\nError en x: {err_x2:.6e}")
    print(f"Error en y: {err_y2:.6e}")
    print(f"{'✓ PASS' if (err_x2 < 1e-2 and err_y2 < 1e-2) else '✗ FAIL'}")
