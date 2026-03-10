"""
Tikhonov-regularized inverse integral for DC motor input reconstruction.
========================================================================

Given noisy measurements of omega(t) and i(t), reconstruct u(t) using:

  min_u  sum_k [u_k*T - RHS_k]^2  +  lambda * sum_k (u_{k+1} - u_k)^2

where RHS_k = Phi(i_k) - Phi(i_{k-1}) + (T/2)*[R*i_{k-1} + K_e*omega_{k-1} + R*i_k + K_e*omega_k]

This is a LINEAR least-squares problem with closed-form tridiagonal solution:
  (I + lambda * D^T * D) u = b
where b_k = RHS_k / T and D is the first-difference matrix.

Author: Rodolfo H. Rodrigo
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded
import time

import motor_model as mm


# ======================================================================
# Antiderivative of L(i)
# ======================================================================

def Phi(i):
    """Phi(i) = integral of L(i) di = L0 * I_SAT * arctan(i / I_SAT)"""
    return mm.L0 * mm.I_SAT * np.arctan(i / mm.I_SAT)


# ======================================================================
# Unregularized inverse integral (point-by-point)
# ======================================================================

def inverse_integral(omega, i_arr, T):
    """
    Reconstruct u from (omega, i) via integrated electrical equation.

    u_k = [Phi(i_k) - Phi(i_{k-1})] / T + R*(i_{k-1}+i_k)/2 + K_e*(omega_{k-1}+omega_k)/2
    """
    n = len(omega)
    u_hat = np.full(n, np.nan)

    for k in range(1, n):
        phi_diff = Phi(i_arr[k]) - Phi(i_arr[k-1])
        Ri_avg = mm.R * (i_arr[k-1] + i_arr[k]) / 2.0
        Ke_avg = mm.K_e * (omega[k-1] + omega[k]) / 2.0
        u_hat[k] = phi_diff / T + Ri_avg + Ke_avg

    return u_hat


# ======================================================================
# Tikhonov-regularized inverse integral
# ======================================================================

def tikhonov_inverse_integral(omega, i_arr, T, lam):
    """
    Solve the Tikhonov-regularized system:
      (I + lam * D^T * D) u = b

    where b_k = RHS_k / T (the unregularized estimate) for k=1,...,n-1
    and D is the (n-2) x (n-1) first-difference matrix.

    D^T * D is tridiagonal with:
      diagonal:     [1, 2, 2, ..., 2, 1]
      off-diagonal: [-1, -1, ..., -1]

    The full system matrix is tridiagonal -> use scipy.linalg.solve_banded.

    Parameters
    ----------
    omega : ndarray, shape (n,)
    i_arr : ndarray, shape (n,)
    T : float
    lam : float, regularization parameter

    Returns
    -------
    u_hat : ndarray, shape (n,) with u_hat[0] = nan
    """
    n = len(omega)
    m = n - 1  # number of unknowns (u_1, ..., u_{n-1})

    # Build RHS vector b (unregularized estimates)
    b = np.zeros(m)
    for k in range(m):
        # k-th equation corresponds to interval [k, k+1] in original indexing
        kk = k + 1  # original index
        phi_diff = Phi(i_arr[kk]) - Phi(i_arr[kk-1])
        Ri_avg = mm.R * (i_arr[kk-1] + i_arr[kk]) / 2.0
        Ke_avg = mm.K_e * (omega[kk-1] + omega[kk]) / 2.0
        b[k] = (phi_diff + T * Ri_avg + T * Ke_avg) / T  # = phi_diff/T + Ri_avg + Ke_avg

    # Build tridiagonal system: (I + lam * D^T * D)
    # D^T*D diagonal: [1, 2, ..., 2, 1]
    # D^T*D off-diagonal: [-1, -1, ..., -1]
    diag = np.ones(m) + lam * np.where(
        np.arange(m) == 0, 1.0, np.where(np.arange(m) == m-1, 1.0, 2.0)
    )
    off_diag = -lam * np.ones(m - 1)

    # Pack into banded form for solve_banded
    # ab[0, :] = upper diagonal (shifted right by 1)
    # ab[1, :] = main diagonal
    # ab[2, :] = lower diagonal (shifted left by 1)
    ab = np.zeros((3, m))
    ab[0, 1:] = off_diag       # upper diagonal
    ab[1, :] = diag             # main diagonal
    ab[2, :-1] = off_diag       # lower diagonal

    u_reg = solve_banded((1, 1), ab, b)

    # Pack result
    u_hat = np.full(n, np.nan)
    u_hat[1:] = u_reg

    return u_hat


# ======================================================================
# Main
# ======================================================================

def main():
    np.random.seed(42)

    T = 0.0001
    t_final = 0.2
    n = int(t_final / T)

    # --- Ground truth via RK4 ---
    print("Generating ground truth via RK4 (dt=1e-6)...")
    t0 = time.time()
    u_func = lambda t: 12.0 if t > 0.005 else 0.0
    x0 = np.array([0.0, 0.0])
    t_rk4, states_rk4, inputs_rk4 = mm.simulate(
        (0, t_final), x0, u_func, dt_rk4=1e-6
    )
    step = int(T / 1e-6)
    idx = np.arange(0, len(t_rk4), step)[:n]
    t_ref = t_rk4[idx]
    omega_ref = states_rk4[idx, 0]
    i_ref = states_rk4[idx, 1]
    u_ref = inputs_rk4[idx]
    print(f"  Done in {time.time()-t0:.1f}s, n={n} samples\n")

    # --- Parameters ---
    noise_levels = [0.01, 0.05, 0.1, 0.5]
    lambda_values = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
    skip = 100  # skip first 100 samples for error metrics
    s = slice(skip, None)

    # --- Results storage ---
    all_results = {}

    # Header
    print("=" * 100)
    print("TIKHONOV REGULARIZATION FOR INVERSE INTEGRAL INPUT RECONSTRUCTION")
    print("=" * 100)
    print(f"\nMotor: R={mm.R}, K_e={mm.K_e}, L0={mm.L0}, I_SAT={mm.I_SAT}")
    print(f"Sampling: T={T:.0e}, n={n}, skip={skip} for RMSE")
    print(f"Noise added to BOTH omega and i (same sigma)")
    print()

    # --- Sweep noise levels and lambda ---
    print(f"{'sigma':>6s}  {'lambda':>10s}  {'RMSE_unreg':>12s}  {'RMSE_tikh':>12s}  "
          f"{'Improvement':>12s}  {'Time [ms]':>10s}")
    print("-" * 75)

    for sigma in noise_levels:
        np.random.seed(42)
        omega_noisy = omega_ref + np.random.normal(0, sigma, n)
        i_noisy = i_ref + np.random.normal(0, sigma, n)

        # Unregularized
        u_unreg = inverse_integral(omega_noisy, i_noisy, T)
        mask = ~np.isnan(u_unreg[s])
        rmse_unreg = np.sqrt(np.mean((u_unreg[s][mask] - u_ref[s][mask])**2))

        best_rmse = np.inf
        best_lam = None
        best_u = None
        results_sigma = {'rmse_unreg': rmse_unreg, 'u_unreg': u_unreg}

        for lam in lambda_values:
            t0 = time.time()
            u_tikh = tikhonov_inverse_integral(omega_noisy, i_noisy, T, lam)
            dt_ms = (time.time() - t0) * 1000

            mask_t = ~np.isnan(u_tikh[s])
            rmse_tikh = np.sqrt(np.mean((u_tikh[s][mask_t] - u_ref[s][mask_t])**2))

            improvement = rmse_unreg / rmse_tikh if rmse_tikh > 0 else np.inf

            print(f"{sigma:6.2f}  {lam:10.3g}  {rmse_unreg:12.4f}  {rmse_tikh:12.4f}  "
                  f"{improvement:11.1f}x  {dt_ms:10.2f}")

            if rmse_tikh < best_rmse:
                best_rmse = rmse_tikh
                best_lam = lam
                best_u = u_tikh.copy()

        results_sigma['best_lam'] = best_lam
        results_sigma['best_rmse'] = best_rmse
        results_sigma['best_u'] = best_u
        all_results[sigma] = results_sigma
        print()

    # --- Summary table ---
    print("\n" + "=" * 80)
    print("SUMMARY: Best Tikhonov result per noise level")
    print("=" * 80)
    print(f"{'sigma':>6s}  {'RMSE_unreg':>12s}  {'Best lambda':>12s}  {'RMSE_tikh':>12s}  {'Improvement':>12s}")
    print("-" * 60)
    for sigma in noise_levels:
        r = all_results[sigma]
        imp = r['rmse_unreg'] / r['best_rmse'] if r['best_rmse'] > 0 else np.inf
        print(f"{sigma:6.2f}  {r['rmse_unreg']:12.4f}  {r['best_lam']:12.3g}  "
              f"{r['best_rmse']:12.4f}  {imp:11.1f}x")

    print("\nKey insight: Tikhonov smoothing significantly reduces noise in the")
    print("inverse integral reconstruction, especially at high noise levels.")
    print("The computation is FAST (tridiagonal solve, <1ms) compared to")
    print("iterative optimization approaches.")

    # --- Plot ---
    fig, axes = plt.subplots(len(noise_levels), 1, figsize=(12, 3.5 * len(noise_levels)),
                             sharex=True)
    ms = t_ref * 1000

    for row, sigma in enumerate(noise_levels):
        r = all_results[sigma]
        ax = axes[row]

        ax.plot(ms, u_ref, 'k-', lw=2, label='True $u$', zorder=5)
        ax.plot(ms[1:], r['u_unreg'][1:], color='tab:blue', lw=0.3, alpha=0.4,
                label=f'Unreg. (RMSE={r["rmse_unreg"]:.3f}V)')
        ax.plot(ms[1:], r['best_u'][1:], color='tab:red', lw=1.2, alpha=0.9,
                label=f'Tikhonov $\\lambda$={r["best_lam"]:.3g} (RMSE={r["best_rmse"]:.3f}V)')

        ax.set_ylabel('$u$ [V]')
        ax.set_ylim(-5, 20)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'$\\sigma$ = {sigma}  (noise on $\\omega$ and $i$)', fontsize=11)

    axes[-1].set_xlabel('Time [ms]')
    plt.suptitle('Tikhonov Regularization: Inverse Integral Input Reconstruction',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    outpath = '/home/rodo/10Paper/results/tikhonov_inv_integral.png'
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {outpath}")
    plt.close()


if __name__ == '__main__':
    main()
