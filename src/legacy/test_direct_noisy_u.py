"""
Test directo con ruido en u(t): dado u ruidoso, determinar ω(t).

Se usa el regressor directo (HFNN) con u + ruido gaussiano
y se compara ω resultante contra ω de RK4 (con u limpia).

Author: Rodolfo H. Rodrigo
"""

import sys
sys.path.insert(0, '/home/rodo/10Paper/regressor')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from solver_system import solve_system
import motor_model as mm


def run_test(T=0.0001, t_final=0.2, noise_std=0.5, seed=42):
    """Run direct HFNN with noisy u, compare ω against clean RK4."""

    np.random.seed(seed)
    n = int(t_final / T)
    c = mm.C_LOAD

    # --- Ground truth RK4 (u limpia) ---
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

    # --- u con ruido gaussiano ---
    u_noisy = u_ref + np.random.normal(0, noise_std, n)

    # --- Funciones residuo (mismas que test_direct.py) ---
    def g1(x, y, z, w, xp, yp, zp, wp, xpp, ypp, zpp, wpp, t):
        omega, i_val = x, y
        return xp + (mm.b * omega + mm.N_load(omega) - mm.K_t * i_val) / mm.J

    def g2(x, y, z, w, xp, yp, zp, wp, xpp, ypp, zpp, wpp, t):
        omega, i_val = x, y
        i_p = yp
        L_i = mm.L_sat(i_val)
        return L_i * i_p + mm.R * i_val + mm.K_e * omega

    funcs = [g1, g2]

    # --- Jacobiano ---
    def j00(x, y, z, w, xp, yp, zp, wp, xpp, ypp, zpp, wpp, t):
        return (mm.b + 2*c*x) / mm.J + 3/(2*T)

    def j01(x, y, z, w, xp, yp, zp, wp, xpp, ypp, zpp, wpp, t):
        return -mm.K_t / mm.J

    def j10(x, y, z, w, xp, yp, zp, wp, xpp, ypp, zpp, wpp, t):
        return mm.K_e

    def j11(x, y, z, w, xp, yp, zp, wp, xpp, ypp, zpp, wpp, t):
        i_val = y
        i_p = yp
        L_i = mm.L_sat(i_val)
        dL = mm.dL_di(i_val)
        return dL * i_p + mm.R + L_i * 3/(2*T)

    jac_funcs = [[j00, j01], [j10, j11]]

    # --- Hessiano ---
    def h000(x, y, z, w, xp, yp, zp, wp, xpp, ypp, zpp, wpp, t):
        return 2*c / mm.J

    def zeros(*args):
        return 0.0

    def h111(x, y, z, w, xp, yp, zp, wp, xpp, ypp, zpp, wpp, t):
        i_val = y
        i_p = yp
        d2L = mm.d2L_di2(i_val)
        dL = mm.dL_di(i_val)
        return d2L * i_p + 2 * dL * 3/(2*T)

    hess_funcs = [
        [[h000, zeros], [zeros, zeros]],
        [[zeros, zeros], [zeros, h111]]
    ]

    # --- Correr regressor con u ruidosa ---
    exc_g1 = np.zeros(n)
    exc_g2 = u_noisy.copy()

    ic = [[omega_ref[0], omega_ref[1]],
          [i_ref[0], i_ref[1]]]

    omega_hfnn, i_hfnn = solve_system(
        funcs, jac_funcs, hess_funcs, None,
        [exc_g1, exc_g2], ic, T, n
    )

    # --- Errores ---
    skip = int(0.01 / T)
    err_omega = np.max(np.abs(omega_hfnn[skip:] - omega_ref[skip:]))
    err_i = np.max(np.abs(i_hfnn[skip:] - i_ref[skip:]))
    rmse_omega = np.sqrt(np.mean((omega_hfnn[skip:] - omega_ref[skip:])**2))
    rmse_i = np.sqrt(np.mean((i_hfnn[skip:] - i_ref[skip:])**2))

    print(f"  σ_u = {noise_std:.2f} V")
    print(f"  ω: max err = {err_omega:.4e},  RMSE = {rmse_omega:.4e}")
    print(f"  i: max err = {err_i:.4e},  RMSE = {rmse_i:.4e}")

    # --- Plot ---
    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    ms = t_ref * 1000

    axes[0].plot(ms, u_ref, 'k-', lw=1.5, label='True $u$')
    axes[0].plot(ms, u_noisy, 'r.', ms=1, alpha=0.3, label=f'Noisy $u$ ($\\sigma$={noise_std})')
    axes[0].set_ylabel('$u$ [V]')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ms, omega_ref, 'b-', lw=1.5, label='RK4 $\\omega$')
    axes[1].plot(ms, omega_hfnn, 'r--', lw=1, label='HFNN $\\omega$')
    axes[1].set_ylabel('$\\omega$ [rad/s]')
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title(f'$\\omega$ (RMSE={rmse_omega:.4e})')

    axes[2].plot(ms, i_ref, 'b-', lw=1.5, label='RK4 $i$')
    axes[2].plot(ms, i_hfnn, 'r--', lw=1, label='HFNN $i$')
    axes[2].set_ylabel('$i$ [A]')
    axes[2].legend(loc='best')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title(f'$i$ (RMSE={rmse_i:.4e})')

    axes[3].plot(ms, omega_hfnn - omega_ref, 'b-', lw=0.5, label='$\\omega$ error')
    axes[3].plot(ms, i_hfnn - i_ref, 'r-', lw=0.5, label='$i$ error')
    axes[3].set_ylabel('Error')
    axes[3].set_xlabel('Time [ms]')
    axes[3].legend(loc='best')
    axes[3].grid(True, alpha=0.3)

    plt.suptitle(f'Direct HFNN with noisy $u$ ($\\sigma$={noise_std} V, T={T:.0e})', fontsize=13)
    plt.tight_layout()
    plt.savefig('/home/rodo/10Paper/results/test_direct_noisy_u.png', dpi=300, bbox_inches='tight')
    print(f"  Saved to results/test_direct_noisy_u.png")
    plt.close()

    return rmse_omega, rmse_i


if __name__ == '__main__':
    print("=== Direct HFNN con ruido en u(t) ===\n")

    T = 0.0001
    for sigma in [0.1, 0.5, 1.0, 2.0]:
        print(f"\nσ = {sigma}:")
        run_test(T=T, t_final=0.2, noise_std=sigma)
