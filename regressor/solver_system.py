"""
solver_system.py - Homotopy regressor for coupled nonlinear ODE systems.

Solves systems of 2 or 3 coupled ODEs:
    F(x, y, z, x', y', z', x'', y'', z'', t) = u(t)
    G(x, y, z, x', y', z', x'', y'', z'', t) = v(t)
    H(x, y, z, x', y', z', x'', y'', z'', t) = w(t)

Using 3-point backward discrete derivatives and 3-term homotopy series.
Extension of scalar solver.py to vector case.

Author: Rodolfo H. Rodrigo - UNSJ
"""

import numpy as np


def solve_system(funcs, jac_funcs, hess_funcs, tens_funcs,
                 excitations, initial_conditions, T, n):
    """
    Solve a system of N coupled ODEs using homotopy regressor.

    Parameters
    ----------
    funcs : list of callable, length N
        [F, G, H] where each function has signature:
            F(x, y, z, xp, yp, zp, xpp, ypp, zpp, t) -> float
        For 1st order: xpp=ypp=zpp=0 always.
        For 2 equations: z=zp=zpp=0 always.

    jac_funcs : list of list of callable, shape NxN
        jac_funcs[i][j] = dgi/dqj already combined with chain rule.
        For example, jac_funcs[0][0] = dg1/dx[k] includes contributions
        from dx, dx', dx'' via chain rule.

    hess_funcs : list of list of list of callable, shape NxNxN
        hess_funcs[i][j][l] = d²gi/dqj dql at current point.

    tens_funcs : list or None
        Third-order tensor. If None, z3 correction is skipped.

    excitations : list of np.ndarray, length N
        [u, v, w] arrays of length n. Pass np.zeros(n) for autonomous.

    initial_conditions : list of list of float
        [[x0,x1], [y0,y1], [z0,z1]]  (two initial conditions per variable)

    T : float
        Sampling period.

    n : int
        Total number of points.

    Returns
    -------
    results : list of np.ndarray, length N
        [x, y, z] arrays of length n.
    """
    N = len(funcs)

    # Inicializar arrays
    results = [np.zeros(n) for _ in range(N)]
    for i in range(N):
        results[i][0] = initial_conditions[i][0]
        results[i][1] = initial_conditions[i][1]

    # Vector tiempo
    t = np.linspace(0, (n-1)*T, n)

    # Loop principal
    for k in range(2, n):
        # Inicializar: q = [x[k], y[k], z[k]] = [x[k-1], y[k-1], z[k-1]]
        q = np.array([results[i][k-1] for i in range(N)])

        # Calcular derivadas discretas de los valores pasados (constantes en este paso)
        # Estos son los términos conocidos en las aproximaciones discretas

        # --- Corrección z1 (Newton vectorial) ---
        G_vec, J_mat = _compute_residual_and_jacobian(
            funcs, jac_funcs, q, results, k, T, t[k], excitations, N
        )

        dz = _solve_linear_system(J_mat, G_vec, N)
        q = q - dz

        # --- Corrección z2 (curvatura - Hessiano) ---
        G_vec, J_mat = _compute_residual_and_jacobian(
            funcs, jac_funcs, q, results, k, T, t[k], excitations, N
        )

        H_vec = _compute_hessian_product(hess_funcs, q, results, k, T, t[k], dz, dz, N)
        dz2 = -0.5 * _solve_linear_system(J_mat, H_vec, N)
        q = q + dz2

        # --- Corrección z3 (tercer orden - tensor) ---
        if tens_funcs is not None:
            G_vec, J_mat = _compute_residual_and_jacobian(
                funcs, jac_funcs, q, results, k, T, t[k], excitations, N
            )

            H_vec = _compute_hessian_product(hess_funcs, q, results, k, T, t[k], dz, dz2, N)
            T_vec = _compute_tensor_product(tens_funcs, q, results, k, T, t[k], dz, dz, dz, N)

            dz3 = -_solve_linear_system(J_mat, H_vec + (1/6)*T_vec, N)
            q = q + dz3

        # Guardar resultado
        for i in range(N):
            results[i][k] = q[i]

    return results


def _compute_residual_and_jacobian(funcs, jac_funcs, q, results, k, T, tk, excitations, N):
    """
    Compute residual vector G and Jacobian matrix J at current q.

    Parameters
    ----------
    funcs : list of callable
    jac_funcs : list of list of callable
    q : np.ndarray, current state [x[k], y[k], z[k]]
    results : list of np.ndarray, all trajectories
    k : int, current time index
    T : float, sampling period
    tk : float, current time
    excitations : list of np.ndarray
    N : int, system dimension (2 or 3)

    Returns
    -------
    G_vec : np.ndarray, shape (N,)
    J_mat : np.ndarray, shape (N, N)
    """
    # Derivadas discretas de q usando valores pasados (conocidos)
    # Primera derivada: xp[k] = (3*x[k] - 4*x[k-1] + x[k-2]) / (2*T)
    # Segunda derivada: xpp[k] = (x[k] - 2*x[k-1] + x[k-2]) / T^2

    qp = np.zeros(N)   # primera derivada
    qpp = np.zeros(N)  # segunda derivada

    for i in range(N):
        qp[i] = (3*q[i] - 4*results[i][k-1] + results[i][k-2]) / (2*T)
        qpp[i] = (q[i] - 2*results[i][k-1] + results[i][k-2]) / (T**2)

    # Construir argumentos para las funciones (rellenar con ceros hasta 4 variables)
    # Formato: (x, y, z, w, xp, yp, zp, wp, xpp, ypp, zpp, wpp, t)
    q_padded = np.pad(q, (0, max(0, 4 - N)), 'constant')
    qp_padded = np.pad(qp, (0, max(0, 4 - N)), 'constant')
    qpp_padded = np.pad(qpp, (0, max(0, 4 - N)), 'constant')

    args = tuple(q_padded) + tuple(qp_padded) + tuple(qpp_padded) + (tk,)

    # Vector residuo
    G_vec = np.zeros(N)
    for i in range(N):
        G_vec[i] = funcs[i](*args) - excitations[i][k]

    # Jacobiano
    J_mat = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            J_mat[i, j] = jac_funcs[i][j](*args)

    return G_vec, J_mat


def _compute_hessian_product(hess_funcs, q, results, k, T, tk, dz1, dz2, N):
    """
    Compute bilinear Hessian product H[dz1, dz2].

    H_i = sum_j sum_l (d²gi/dqj dql) * dz1[j] * dz2[l]

    Returns
    -------
    H_vec : np.ndarray, shape (N,)
    """
    # Derivadas discretas
    qp = np.zeros(N)
    qpp = np.zeros(N)

    for i in range(N):
        qp[i] = (3*q[i] - 4*results[i][k-1] + results[i][k-2]) / (2*T)
        qpp[i] = (q[i] - 2*results[i][k-1] + results[i][k-2]) / (T**2)

    # Construir argumentos (rellenar hasta 4 variables)
    q_padded = np.pad(q, (0, max(0, 4 - N)), 'constant')
    qp_padded = np.pad(qp, (0, max(0, 4 - N)), 'constant')
    qpp_padded = np.pad(qpp, (0, max(0, 4 - N)), 'constant')

    args = tuple(q_padded) + tuple(qp_padded) + tuple(qpp_padded) + (tk,)

    H_vec = np.zeros(N)
    for i in range(N):
        for j in range(N):
            for l in range(N):
                H_vec[i] += hess_funcs[i][j][l](*args) * dz1[j] * dz2[l]

    return H_vec


def _compute_tensor_product(tens_funcs, q, results, k, T, tk, dz1, dz2, dz3, N):
    """
    Compute trilinear tensor product T[dz1, dz2, dz3].

    T_i = sum_j sum_l sum_m (d³gi/dqj dql dqm) * dz1[j] * dz2[l] * dz3[m]

    Returns
    -------
    T_vec : np.ndarray, shape (N,)
    """
    qp = np.zeros(N)
    qpp = np.zeros(N)

    for i in range(N):
        qp[i] = (3*q[i] - 4*results[i][k-1] + results[i][k-2]) / (2*T)
        qpp[i] = (q[i] - 2*results[i][k-1] + results[i][k-2]) / (T**2)

    # Construir argumentos (rellenar hasta 4 variables)
    q_padded = np.pad(q, (0, max(0, 4 - N)), 'constant')
    qp_padded = np.pad(qp, (0, max(0, 4 - N)), 'constant')
    qpp_padded = np.pad(qpp, (0, max(0, 4 - N)), 'constant')

    args = tuple(q_padded) + tuple(qp_padded) + tuple(qpp_padded) + (tk,)

    T_vec = np.zeros(N)
    for i in range(N):
        for j in range(N):
            for l in range(N):
                for m in range(N):
                    T_vec[i] += tens_funcs[i][j][l][m](*args) * dz1[j] * dz2[l] * dz3[m]

    return T_vec


def _solve_linear_system(J_mat, vec, N):
    """
    Solve J_mat * x = vec.

    For 2x2: use analytic formula (no np.linalg.solve in loop).
    For 3x3: use np.linalg.solve (acceptable).

    Parameters
    ----------
    J_mat : np.ndarray, shape (N, N)
    vec : np.ndarray, shape (N,)
    N : int, system dimension

    Returns
    -------
    x : np.ndarray, shape (N,)
    """
    if N == 2:
        # Fórmula analítica para 2x2
        # J = [[a,b],[c,d]]; det = a*d - b*c
        # J^{-1} * v = (1/det) * [d*v1 - b*v2, -c*v1 + a*v2]
        a, b = J_mat[0, 0], J_mat[0, 1]
        c, d = J_mat[1, 0], J_mat[1, 1]
        det = a*d - b*c

        if abs(det) < 1e-14:
            raise ValueError("Jacobiano singular en sistema 2x2")

        x = np.array([
            (d * vec[0] - b * vec[1]) / det,
            (-c * vec[0] + a * vec[1]) / det
        ])
        return x
    else:
        # Para 3x3 usar np.linalg.solve
        return np.linalg.solve(J_mat, vec)


def solve_system_numeric(funcs, excitations, initial_conditions, T, n, h=1e-5):
    """
    Solve a system with numerically computed Jacobian, Hessian, and tensor.

    Uses 4th-order centered finite differences with chain rule for discrete derivatives.
    Only requires [F, G, H] functions.

    Parameters
    ----------
    funcs : list of callable, length N
        [F, G, H] where each function has signature:
            F(x, y, z, xp, yp, zp, xpp, ypp, zpp, t) -> float

    excitations : list of np.ndarray, length N
        [u, v, w] arrays of length n.

    initial_conditions : list of list of float
        [[x0,x1], [y0,y1], [z0,z1]]

    T : float
        Sampling period.

    n : int
        Number of points.

    h : float
        Step for numerical derivatives (default 1e-5).

    Returns
    -------
    results : list of np.ndarray, length N
        [x, y, z] arrays of length n.
    """
    N = len(funcs)

    # Construir funciones Jacobiano numéricas con regla de la cadena
    # dgi/dqj[k] = dFi/dqj + dFi/dqpj * d(qpj)/dqj[k] + dFi/dqppj * d(qppj)/dqj[k]
    #            = dFi/dqj + dFi/dqpj * 3/(2T) + dFi/dqppj * 1/T^2

    def make_jac(i, j):
        def jac_ij(x, y, z, xp, yp, zp, xpp, ypp, zpp, t):
            args = [x, y, z, xp, yp, zp, xpp, ypp, zpp, t]

            # dFi/dqj (índice j en [0,1,2] para x,y,z)
            def eval_func_q(delta):
                args_p = args.copy()
                args_p[j] += delta
                return funcs[i](*args_p)
            dFi_dqj = (8*(eval_func_q(h) - eval_func_q(-h)) -
                       (eval_func_q(2*h) - eval_func_q(-2*h))) / (12*h)

            # dFi/dqpj (índice j+3 en [3,4,5] para xp,yp,zp)
            def eval_func_qp(delta):
                args_p = args.copy()
                args_p[j + 3] += delta
                return funcs[i](*args_p)
            dFi_dqpj = (8*(eval_func_qp(h) - eval_func_qp(-h)) -
                        (eval_func_qp(2*h) - eval_func_qp(-2*h))) / (12*h)

            # dFi/dqppj (índice j+6 en [6,7,8] para xpp,ypp,zpp)
            def eval_func_qpp(delta):
                args_p = args.copy()
                args_p[j + 6] += delta
                return funcs[i](*args_p)
            dFi_dqppj = (8*(eval_func_qpp(h) - eval_func_qpp(-h)) -
                         (eval_func_qpp(2*h) - eval_func_qpp(-2*h))) / (12*h)

            # Aplicar regla de la cadena
            return dFi_dqj + dFi_dqpj * 3/(2*T) + dFi_dqppj * 1/(T**2)

        return jac_ij

    jac_funcs = [[make_jac(i, j) for j in range(N)] for i in range(N)]

    # Construir funciones Hessiano numéricas (también con regla de la cadena)
    def make_hess(i, j, l):
        def hess_ijl(x, y, z, xp, yp, zp, xpp, ypp, zpp, t):
            # Derivada de jac_funcs[i][j] respecto a ql[k]
            # Usamos diferencias numéricas sobre jac_ij
            args = [x, y, z, xp, yp, zp, xpp, ypp, zpp, t]

            def eval_jac(delta):
                args_p = args.copy()
                args_p[l] += delta
                return jac_funcs[i][j](*args_p)

            return (8*(eval_jac(h) - eval_jac(-h)) -
                    (eval_jac(2*h) - eval_jac(-2*h))) / (12*h)
        return hess_ijl

    hess_funcs = [[[make_hess(i, j, l) for l in range(N)]
                    for j in range(N)] for i in range(N)]

    # Tensor de tercer orden: None por defecto (solo z1 y z2)
    tens_funcs = None

    return solve_system(funcs, jac_funcs, hess_funcs, tens_funcs,
                        excitations, initial_conditions, T, n)
