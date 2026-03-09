"""
Tikhonov regularization for unknown input reconstruction of a DC motor.

Given noisy measurements of omega(t), reconstruct the unknown input u(t)
by solving the optimization problem:

    min_u  sum_k ||omega_k^meas - omega_k^model(u)||^2  +  lambda * sum_k (u_{k+1} - u_k)^2

The forward model omega_k(u) is the HFNN direct regressor (solve_system)
applied to the coupled DC motor equations.

Author: Rodolfo H. Rodrigo
"""

import sys
sys.path.insert(0, '/home/rodo/10Paper/regressor')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from solver_system import solve_system
import motor_model as mm


# ---------------------------------------------------------------------------
# Forward model: given u array, return omega array via HFNN direct regressor
# ---------------------------------------------------------------------------

def build_motor_funcs(T):
    """
    Build residual functions, Jacobian, and Hessian for the DC motor system.
    Same formulation as test_direct.py (g2 multiplied by L(i) so excitation = u).
    """
    c = mm.C_LOAD

    # Residual functions
    def g1(x, y, z, w, xp, yp, zp, wp, xpp, ypp, zpp, wpp, t):
        omega, i = x, y
        return xp + (mm.b * omega + mm.N_load(omega) - mm.K_t * i) / mm.J

    def g2(x, y, z, w, xp, yp, zp, wp, xpp, ypp, zpp, wpp, t):
        omega, i = x, y
        i_p = yp
        L_i = mm.L_sat(i)
        return L_i * i_p + mm.R * i + mm.K_e * omega

    funcs = [g1, g2]

    # Jacobian (with discrete chain rule)
    def j00(x, y, z, w, xp, yp, zp, wp, xpp, ypp, zpp, wpp, t):
        return (mm.b + 2 * c * x) / mm.J + 3 / (2 * T)

    def j01(x, y, z, w, xp, yp, zp, wp, xpp, ypp, zpp, wpp, t):
        return -mm.K_t / mm.J

    def j10(x, y, z, w, xp, yp, zp, wp, xpp, ypp, zpp, wpp, t):
        return mm.K_e

    def j11(x, y, z, w, xp, yp, zp, wp, xpp, ypp, zpp, wpp, t):
        i = y
        i_p = yp
        L_i = mm.L_sat(i)
        dL = mm.dL_di(i)
        return dL * i_p + mm.R + L_i * 3 / (2 * T)

    jac_funcs = [[j00, j01], [j10, j11]]

    # Hessian
    def h000(x, y, z, w, xp, yp, zp, wp, xpp, ypp, zpp, wpp, t):
        return 2 * c / mm.J

    def zeros(*args):
        return 0.0

    def h111(x, y, z, w, xp, yp, zp, wp, xpp, ypp, zpp, wpp, t):
        i = y
        i_p = yp
        d2L = mm.d2L_di2(i)
        dL = mm.dL_di(i)
        return d2L * i_p + 2 * dL * 3 / (2 * T)

    hess_funcs = [
        [[h000, zeros], [zeros, zeros]],
        [[zeros, zeros], [zeros, h111]]
    ]

    return funcs, jac_funcs, hess_funcs


def forward_model(u_vec, T, n, ic_omega, ic_i, funcs, jac_funcs, hess_funcs):
    """
    Run the HFNN direct regressor with given u vector.
    Returns omega array of length n.
    """
    exc_g1 = np.zeros(n)
    exc_g2 = u_vec.copy()

    ic = [[ic_omega[0], ic_omega[1]],
          [ic_i[0], ic_i[1]]]

    omega_hfnn, i_hfnn = solve_system(
        funcs, jac_funcs, hess_funcs, None,
        [exc_g1, exc_g2], ic, T, n
    )

    return omega_hfnn, i_hfnn


def tikhonov_cost(u_vec, omega_meas, T, n, ic_omega, ic_i,
                  funcs, jac_funcs, hess_funcs, lam):
    """
    Tikhonov cost function:
        J(u) = sum (omega_meas - omega_model(u))^2  +  lam * sum (u_{k+1} - u_k)^2
    """
    omega_model, _ = forward_model(
        u_vec, T, n, ic_omega, ic_i, funcs, jac_funcs, hess_funcs
    )

    # Data fidelity
    residuals = omega_meas - omega_model
    fidelity = np.sum(residuals**2)

    # Roughness penalty (first-order differences)
    du = np.diff(u_vec)
    roughness = np.sum(du**2)

    return fidelity + lam * roughness


def reconstruct_input(omega_meas, T, n, ic_omega, ic_i, lam,
                      u_init=None, maxiter=200, verbose=True):
    """
    Reconstruct unknown input u(t) from noisy omega measurements
    using Tikhonov regularization with L-BFGS-B.

    Parameters
    ----------
    omega_meas : ndarray, shape (n,)
        Noisy omega measurements
    T : float
        Sampling period
    n : int
        Number of samples
    ic_omega : tuple (omega_0, omega_1)
        Initial conditions for omega (first 2 values)
    ic_i : tuple (i_0, i_1)
        Initial conditions for current (first 2 values)
    lam : float
        Regularization parameter
    u_init : ndarray or None
        Initial guess for u. If None, use zeros.
    maxiter : int
        Maximum iterations for L-BFGS-B

    Returns
    -------
    u_opt : ndarray, shape (n,)
        Reconstructed input
    omega_model : ndarray, shape (n,)
        Model output with reconstructed input
    result : OptimizeResult
        Full optimization result
    """
    funcs, jac_funcs, hess_funcs = build_motor_funcs(T)

    if u_init is None:
        u_init = np.zeros(n)

    # Bound u to physically reasonable range
    bounds = [(-1.0, 25.0)] * n

    if verbose:
        print(f"  Tikhonov reconstruction: lambda={lam:.1e}, n={n}, maxiter={maxiter}")

    iter_count = [0]

    def callback(xk):
        iter_count[0] += 1
        if verbose and iter_count[0] % 20 == 0:
            cost = tikhonov_cost(
                xk, omega_meas, T, n, ic_omega, ic_i,
                funcs, jac_funcs, hess_funcs, lam
            )
            print(f"    iter {iter_count[0]:4d}, cost = {cost:.6e}")

    result = minimize(
        tikhonov_cost,
        u_init,
        args=(omega_meas, T, n, ic_omega, ic_i, funcs, jac_funcs, hess_funcs, lam),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 50000},
        callback=callback
    )

    u_opt = result.x
    omega_model, i_model = forward_model(
        u_opt, T, n, ic_omega, ic_i, funcs, jac_funcs, hess_funcs
    )

    if verbose:
        print(f"    Converged: {result.success}, nit={result.nit}, "
              f"nfev={result.nfev}, final cost={result.fun:.6e}")

    return u_opt, omega_model, result


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------

def main():
    np.random.seed(42)

    T = 0.0001
    t_final = 0.2
    n = int(t_final / T)  # 2000

    # --- Ground truth via RK4 ---
    u_func = lambda t: 12.0 if t > 0.005 else 0.0
    x0 = np.array([0.0, 0.0])
    t_rk4, states_rk4, inputs_rk4 = mm.simulate(
        (0, t_final), x0, u_func, dt_rk4=1e-6
    )

    step = int(T / 1e-6)
    idx = np.arange(0, len(t_rk4), step)[:n]
    t_ref = t_rk4[idx]
    omega_true = states_rk4[idx, 0]
    i_true = states_rk4[idx, 1]
    u_true = inputs_rk4[idx]

    # Initial conditions (from ground truth, these are known)
    ic_omega = (omega_true[0], omega_true[1])
    ic_i = (i_true[0], i_true[1])

    # --- Noise levels and lambda values ---
    noise_levels = [0.1, 0.5, 1.0]
    # Lambda values to test per noise level
    lambda_values = {
        0.1: [1e2, 1e3, 1e4],
        0.5: [1e3, 1e4, 1e5],
        1.0: [1e4, 1e5, 1e6],
    }

    # Storage for results
    all_results = {}

    for sigma in noise_levels:
        np.random.seed(42)
        omega_meas = omega_true + np.random.normal(0, sigma, n)

        all_results[sigma] = {}

        for lam in lambda_values[sigma]:
            print(f"\n=== sigma={sigma}, lambda={lam:.0e} ===")

            # Initial guess: constant at mean of likely input range
            u_init = np.ones(n) * 6.0

            u_opt, omega_model, result = reconstruct_input(
                omega_meas, T, n, ic_omega, ic_i, lam,
                u_init=u_init, maxiter=300, verbose=True
            )

            # Error metrics (skip transient: first 100 samples = 10ms)
            skip = 100
            rmse_u = np.sqrt(np.mean((u_opt[skip:] - u_true[skip:])**2))
            max_err_u = np.max(np.abs(u_opt[skip:] - u_true[skip:]))
            rmse_omega = np.sqrt(np.mean((omega_model[skip:] - omega_true[skip:])**2))

            print(f"    u RMSE = {rmse_u:.4f} V, u max err = {max_err_u:.4f} V")
            print(f"    omega RMSE = {rmse_omega:.6f} rad/s")

            all_results[sigma][lam] = {
                'u_opt': u_opt,
                'omega_model': omega_model,
                'rmse_u': rmse_u,
                'max_err_u': max_err_u,
                'rmse_omega': rmse_omega,
                'omega_meas': omega_meas,
            }

    # --- Plot: 3 rows (noise levels) x 2 cols (u reconstruction, omega fit) ---
    fig, axes = plt.subplots(len(noise_levels), 2, figsize=(14, 4 * len(noise_levels)),
                             sharex=True)
    ms = t_ref * 1000

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for row, sigma in enumerate(noise_levels):
        ax_u = axes[row, 0]
        ax_w = axes[row, 1]

        # True u
        ax_u.plot(ms, u_true, 'k-', lw=2, label='True $u$', zorder=10)
        ax_w.plot(ms, omega_true, 'k-', lw=2, label='True $\\omega$')

        # Noisy omega
        omega_meas = all_results[sigma][lambda_values[sigma][0]]['omega_meas']
        ax_w.plot(ms, omega_meas, '.', color='gray', ms=0.5, alpha=0.3,
                  label=f'Measured ($\\sigma$={sigma})')

        # Best lambda for this noise level (pick the one with lowest u RMSE)
        best_lam = min(lambda_values[sigma],
                       key=lambda l: all_results[sigma][l]['rmse_u'])

        for ci, lam in enumerate(lambda_values[sigma]):
            res = all_results[sigma][lam]
            style = '-' if lam == best_lam else '--'
            alpha_val = 1.0 if lam == best_lam else 0.6
            lw = 1.5 if lam == best_lam else 0.8

            label_u = f'$\\lambda$={lam:.0e} (RMSE={res["rmse_u"]:.2f}V)'
            ax_u.plot(ms, res['u_opt'], style, color=colors[ci],
                      lw=lw, alpha=alpha_val, label=label_u)

            if lam == best_lam:
                ax_w.plot(ms, res['omega_model'], '--', color=colors[ci],
                          lw=1.5, label=f'Model ($\\lambda$={lam:.0e})')

        ax_u.set_ylabel('$u$ [V]')
        ax_u.legend(fontsize=7, loc='best')
        ax_u.grid(True, alpha=0.3)
        ax_u.set_title(f'Input reconstruction ($\\sigma_\\omega$={sigma} rad/s)', fontsize=10)
        ax_u.set_ylim(-2, 16)

        ax_w.set_ylabel('$\\omega$ [rad/s]')
        ax_w.legend(fontsize=7, loc='best')
        ax_w.grid(True, alpha=0.3)
        ax_w.set_title(f'Speed fit ($\\sigma_\\omega$={sigma} rad/s)', fontsize=10)

    axes[-1, 0].set_xlabel('Time [ms]')
    axes[-1, 1].set_xlabel('Time [ms]')

    plt.suptitle('Tikhonov Regularization: Unknown Input Reconstruction of DC Motor',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('/home/rodo/10Paper/results/tikhonov_inverse.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to /home/rodo/10Paper/results/tikhonov_inverse.png")
    plt.close()

    # --- Summary table ---
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'sigma':>6s} {'lambda':>10s} {'u RMSE [V]':>12s} {'u max err [V]':>14s} {'omega RMSE':>12s}")
    print("-" * 70)
    for sigma in noise_levels:
        for lam in lambda_values[sigma]:
            res = all_results[sigma][lam]
            print(f"{sigma:6.1f} {lam:10.0e} {res['rmse_u']:12.4f} "
                  f"{res['max_err_u']:14.4f} {res['rmse_omega']:12.6f}")
    print("=" * 70)


if __name__ == '__main__':
    main()
