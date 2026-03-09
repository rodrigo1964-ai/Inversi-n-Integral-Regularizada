#!/usr/bin/env python3
"""
shooting_jacobian.py - Cálculo analítico del Jacobiano del shooting para sistemas lineales

Para el sistema estado-coestado lineal:
    dx/dt = Ac*x - Ec*λ
    dλ/dt = -Qc*x - Ac^T*λ

El mapeo λ_0 → x_N es AFÍN:
    x_N = Φ_xx * x_0 + Φ_xλ * λ_0 + ψ

Este módulo calcula Φ_xλ = ∂x_N/∂λ_0 simbólicamente usando el regressor HAM,
eliminando la necesidad de shooting iterativo.

Author: Rodolfo H. Rodrigo - UNSJ
"""

import numpy as np
from sympy import Symbol, symbols
from regressor_system import build_system_regressor


def compute_shooting_jacobian_analytic(Ac, Bc, Q, R, x0, N, T):
    """
    Calcula el Jacobiano del shooting Φ_xλ = ∂x_N/∂λ_0 simbólicamente
    para el sistema LQR estado-coestado lineal.

    Para sistemas lineales, el regressor produce un mapeo afín, por lo que
    el Jacobiano es CONSTANTE y puede calcularse propagando perturbaciones
    unitarias a través del regressor.

    Parameters
    ----------
    Ac, Bc : np.ndarray
        Matrices del sistema continuo
    Q, R : np.ndarray
        Matrices de costo
    x0 : np.ndarray
        Estado inicial (shape (n,))
    N : int
        Horizonte
    T : float
        Tiempo total

    Returns
    -------
    Phi_xlam : np.ndarray, shape (n, n)
        Jacobiano ∂x_N/∂λ_0
    Phi_xx : np.ndarray, shape (n, n)
        Jacobiano ∂x_N/∂x_0
    psi : np.ndarray, shape (n,)
        Término constante (cero para LQR homogéneo)
    regressor : callable
        El regressor construido (para reutilizar)
    """
    n = Ac.shape[0]
    Ec = Bc @ np.linalg.inv(R) @ Bc.T
    Qc = Q
    h = T / N

    # ========================================================================
    # CONSTRUIR REGRESSOR SIMBÓLICO (una sola vez)
    # ========================================================================

    x_sym, y_sym, z_sym, w_sym = symbols('x y z w')
    xp_sym, yp_sym, zp_sym, wp_sym = symbols('xp yp zp wp')
    xpp_sym, ypp_sym, zpp_sym, wpp_sym = symbols('xpp ypp zpp wpp')
    t_sym = Symbol('t')

    state_syms = [x_sym, y_sym, z_sym, w_sym,
                  xp_sym, yp_sym, zp_sym, wp_sym,
                  xpp_sym, ypp_sym, zpp_sym, wpp_sym, t_sym]

    # Ecuaciones del sistema estado-coestado (n=2 → 4 ecuaciones)
    F1_expr = (xp_sym - (Ac[0,0]*x_sym + Ac[0,1]*y_sym
                         - Ec[0,0]*z_sym - Ec[0,1]*w_sym))
    F2_expr = (yp_sym - (Ac[1,0]*x_sym + Ac[1,1]*y_sym
                         - Ec[1,0]*z_sym - Ec[1,1]*w_sym))
    F3_expr = (zp_sym - (-Qc[0,0]*x_sym - Qc[0,1]*y_sym
                         - Ac[0,0]*z_sym - Ac[1,0]*w_sym))
    F4_expr = (wp_sym - (-Qc[1,0]*x_sym - Qc[1,1]*y_sym
                         - Ac[0,1]*z_sym - Ac[1,1]*w_sym))

    func_exprs = [F1_expr, F2_expr, F3_expr, F4_expr]

    print("Building regressor for Jacobian computation...")
    regressor, info = build_system_regressor(func_exprs, state_syms, order=1)
    print("  Regressor built.\n")

    # ========================================================================
    # CALCULAR Φ_xλ PROPAGANDO PERTURBACIONES UNITARIAS
    # ========================================================================

    # Para un sistema lineal, el mapeo es:
    #   [x_N]   [Φ_xx  Φ_xλ] [x_0]   [ψ_x]
    #   [λ_N] = [Φ_λx  Φ_λλ] [λ_0] + [ψ_λ]

    # Calculamos Φ_xλ propagando con x_0 = 0, λ_0 = e_i (vector unitario)

    Phi_xlam = np.zeros((n, n))
    Phi_xx = np.zeros((n, n))

    excitations = [np.zeros(N+1) for _ in range(4)]

    # Aproximación inicial para x1, lambda_1 (necesitamos 2 puntos iniciales)
    def get_initial_point(x0_val, lambda0_val):
        x1_approx = x0_val + h * (Ac @ x0_val - Ec @ lambda0_val)
        lambda_1_approx = lambda0_val + h * (-Qc @ x0_val - Ac.T @ lambda0_val)
        return [
            [x0_val[0], x1_approx[0]],
            [x0_val[1], x1_approx[1]],
            [lambda0_val[0], lambda_1_approx[0]],
            [lambda0_val[1], lambda_1_approx[1]]
        ]

    # 1. Calcular Φ_xλ: perturbar λ_0, mantener x_0 = 0
    for i in range(n):
        lambda0_pert = np.zeros(n)
        lambda0_pert[i] = 1.0  # Perturbación unitaria en λ_0[i]

        initial_conditions = get_initial_point(np.zeros(n), lambda0_pert)
        results = regressor(excitations, initial_conditions, h, N+1)

        # x_N está en results[0][-1], results[1][-1]
        Phi_xlam[0, i] = results[0][-1]
        Phi_xlam[1, i] = results[1][-1]

    # 2. Calcular Φ_xx: perturbar x_0, mantener λ_0 = 0
    for i in range(n):
        x0_pert = np.zeros(n)
        x0_pert[i] = 1.0  # Perturbación unitaria en x_0[i]

        initial_conditions = get_initial_point(x0_pert, np.zeros(n))
        results = regressor(excitations, initial_conditions, h, N+1)

        Phi_xx[0, i] = results[0][-1]
        Phi_xx[1, i] = results[1][-1]

    # 3. Calcular ψ: propagar con x_0 = 0, λ_0 = 0
    initial_conditions = get_initial_point(np.zeros(n), np.zeros(n))
    results = regressor(excitations, initial_conditions, h, N+1)
    psi = np.array([results[0][-1], results[1][-1]])

    print("Jacobiano del shooting calculado:")
    print(f"  ||Φ_xλ|| = {np.linalg.norm(Phi_xlam):.6f}")
    print(f"  ||Φ_xx|| = {np.linalg.norm(Phi_xx):.6f}")
    print(f"  ||ψ||    = {np.linalg.norm(psi):.6e}")
    print()

    return Phi_xlam, Phi_xx, psi, regressor


def solve_lqr_linear_direct(Ac, Bc, Q, R, x0, xf, N, T):
    """
    Resuelve el problema LQR lineal SIN shooting iterativo.

    Usa el Jacobiano analítico Φ_xλ para calcular λ_0 directamente:
        λ_0 = Φ_xλ^{-1} * (xf - Φ_xx * x0 - ψ)

    Luego propaga forward una sola vez.

    Returns
    -------
    x_traj, u_traj, lambda_traj : np.ndarray
        Trayectorias óptimas
    lambda_0 : np.ndarray
        Coestado inicial calculado
    """
    n = Ac.shape[0]
    m = Bc.shape[1]
    h = T / N

    # Calcular Jacobiano analíticamente
    Phi_xlam, Phi_xx, psi, regressor = compute_shooting_jacobian_analytic(
        Ac, Bc, Q, R, x0, N, T
    )

    # Resolver para λ_0 (sin iteraciones!)
    rhs = xf - Phi_xx @ x0 - psi
    lambda_0 = np.linalg.solve(Phi_xlam, rhs)

    print(f"λ_0 calculado directamente (sin shooting):")
    print(f"  λ_0 = [{lambda_0[0]:.6f}, {lambda_0[1]:.6f}]")
    print()

    # Propagar forward con λ_0 óptimo
    Ec = Bc @ np.linalg.inv(R) @ Bc.T
    Qc = Q

    x1_approx = x0 + h * (Ac @ x0 - Ec @ lambda_0)
    lambda_1_approx = lambda_0 + h * (-Qc @ x0 - Ac.T @ lambda_0)

    initial_conditions = [
        [x0[0], x1_approx[0]],
        [x0[1], x1_approx[1]],
        [lambda_0[0], lambda_1_approx[0]],
        [lambda_0[1], lambda_1_approx[1]]
    ]

    excitations = [np.zeros(N+1) for _ in range(4)]
    results = regressor(excitations, initial_conditions, h, N+1)

    # Extraer trayectorias
    x_traj = np.column_stack([results[0], results[1]])
    lambda_traj = np.column_stack([results[2], results[3]])

    # Calcular control
    u_traj = np.zeros((N, m))
    for k in range(N):
        u_traj[k] = -np.linalg.inv(R) @ Bc.T @ lambda_traj[k+1]

    # Verificar error final
    error = np.linalg.norm(x_traj[-1] - xf)
    print(f"Error final: ||x_N - x_f|| = {error:.2e}")
    print()

    return x_traj, u_traj, lambda_traj, lambda_0


if __name__ == "__main__":
    # Test con sistema de 2do orden
    print("=" * 70)
    print("TEST: Cálculo de Jacobiano del Shooting (Analítico vs Numérico)")
    print("=" * 70)
    print()

    # Sistema de prueba
    Ac = np.array([[0.0, 1.0], [0.0, 0.0]])
    Bc = np.array([[0.0], [1.0]])
    Q = np.eye(2)
    R = np.array([[1.0]])

    x0 = np.array([1.0, 0.0])
    xf = np.array([0.0, 0.0])
    N = 30
    T = 5.0

    # Calcular Jacobiano analíticamente
    Phi_xlam, Phi_xx, psi, regressor = compute_shooting_jacobian_analytic(
        Ac, Bc, Q, R, x0, N, T
    )

    print("Φ_xλ (Jacobiano del shooting):")
    print(Phi_xlam)
    print()

    # Resolver LQR directamente (sin shooting)
    print("Resolviendo LQR sin shooting iterativo...")
    x_traj, u_traj, lambda_traj, lambda_0 = solve_lqr_linear_direct(
        Ac, Bc, Q, R, x0, xf, N, T
    )

    print("✓ TEST COMPLETADO")
