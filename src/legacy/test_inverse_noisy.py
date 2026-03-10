"""
Test inverso con ruido: dado ω(t) e i(t) con ruido gaussiano, reconstruir u(t).

û_k = L(i_k)·di/dt + R·i_k + K_e·ω_k

Se prueba con 2, 3, y 4 puntos para di/dt.

Author: Rodolfo H. Rodrigo
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import motor_model as mm
from test_inverse import reconstruct_u


def run_test(T=0.0001, t_final=0.2, noise_std_omega=0.1, noise_std_i=0.1, seed=42):
    """Inverse HFNN with noisy ω and i."""

    np.random.seed(seed)

    # Ground truth RK4
    u_func = lambda t: 12.0 if t > 0.005 else 0.0
    x0 = np.array([0.0, 0.0])
    t_rk4, states_rk4, inputs_rk4 = mm.simulate(
        (0, t_final), x0, u_func, dt_rk4=1e-6
    )

    n = int(t_final / T)
    step = int(T / 1e-6)
    idx = np.arange(0, len(t_rk4), step)[:n]

    t_ref = t_rk4[idx]
    omega_ref = states_rk4[idx, 0]
    i_ref = states_rk4[idx, 1]
    u_ref = inputs_rk4[idx]

    # Agregar ruido gaussiano
    omega_noisy = omega_ref + np.random.normal(0, noise_std_omega, n)
    i_noisy = i_ref + np.random.normal(0, noise_std_i, n)

    # Reconstruir u con 2, 3, 4 puntos
    skip = int(0.01 / T)
    results = {}

    for np_ in [2, 3, 4]:
        u_hat = reconstruct_u(omega_noisy, i_noisy, T, n_points=np_)
        err_rmse = np.sqrt(np.mean((u_hat[skip:] - u_ref[skip:])**2))
        err_max = np.max(np.abs(u_hat[skip:] - u_ref[skip:]))
        results[np_] = {'u_hat': u_hat, 'err_rmse': err_rmse, 'err_max': err_max}
        print(f"  {np_}pt: RMSE = {err_rmse:.4f} V,  max err = {err_max:.4f} V")

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    ms = t_ref * 1000

    axes[0].plot(ms, u_ref, 'k-', lw=2, label='True $u$')
    colors = ['b', 'r', 'g']
    styles = [':', '--', '-.']
    for (np_, res), c, s in zip(results.items(), colors, styles):
        axes[0].plot(ms, res['u_hat'], linestyle=s, color=c, lw=0.5, alpha=0.7,
                     label=f'{np_}pt (RMSE={res["err_rmse"]:.2f})')
    axes[0].set_ylabel('$u$ [V]')
    axes[0].legend(loc='best', fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f'$\\hat{{u}}$ from noisy $\\omega$,$i$  '
                      f'($\\sigma_\\omega$={noise_std_omega}, $\\sigma_i$={noise_std_i})')

    # Error
    for (np_, res), c, s in zip(results.items(), colors, styles):
        axes[1].plot(ms, res['u_hat'] - u_ref, color=c, lw=0.5, alpha=0.7,
                     label=f'{np_}pt error')
    axes[1].set_ylabel('$\\hat{u} - u$ [V]')
    axes[1].legend(loc='best', fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # Señales ruidosas
    axes[2].plot(ms, omega_noisy - omega_ref, 'b-', lw=0.3, alpha=0.5, label='$\\omega$ noise')
    axes[2].plot(ms, i_noisy - i_ref, 'r-', lw=0.3, alpha=0.5, label='$i$ noise')
    axes[2].set_ylabel('Noise')
    axes[2].set_xlabel('Time [ms]')
    axes[2].legend(loc='best')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/rodo/10Paper/results/test_inverse_noisy.png', dpi=300, bbox_inches='tight')
    print(f"  Saved to results/test_inverse_noisy.png")
    plt.close()

    return results


if __name__ == '__main__':
    print("=== Inverse HFNN con ruido en ω(t), i(t) ===\n")

    T = 0.0001
    for sigma in [0.01, 0.05, 0.1, 0.5]:
        print(f"\nσ = {sigma}:")
        run_test(T=T, t_final=0.2, noise_std_omega=sigma, noise_std_i=sigma)
