"""
Baseline Comparisons
=====================

Baselines for comparison against the dual HFNN + EKF approach:
1. No filter (direct inversion from noisy measurements)
2. Augmented EKF (random walk on u_k)
3. Simple moving average filter + inversion

All experiments use Gaussian measurement noise only.

Author: Rodolfo H. Rodrigo
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from motor_model import DCMotor, generate_experiment_data
from hfnn_direct import DirectHFNN
from hfnn_inverse import InverseHFNN
from dual_estimation import run_dual_estimation


def no_filter_estimation(noise_std=0.5, n_samples=500, u_type='step'):
    """
    Baseline 1: Direct inversion from raw noisy measurements (no EKF).

    Simply applies current extraction and input reconstruction
    on the noisy ω measurements without any filtering.
    """
    motor = DCMotor()
    T = 0.001

    t, omega_true, omega_meas, i_true, u_true = generate_experiment_data(
        motor, T_sample=T, n_samples=n_samples,
        u_type=u_type, noise_std=noise_std, seed=42
    )

    inverse = InverseHFNN(
        R=motor.R, K_e=motor.K_e, K_t=motor.K_t,
        J=motor.J, b=motor.b,
        L0=motor.L0, i_sat=motor.i_sat, T=T
    )

    i_estimated = np.zeros(n_samples)
    u_estimated = np.zeros(n_samples)
    i_prev = 0.0

    for k in range(1, n_samples):
        i_k = inverse.extract_current(omega_meas[k], omega_meas[k-1])
        i_estimated[k] = i_k
        u_estimated[k] = inverse.reconstruct_input(i_k, i_prev, omega_meas[k])
        i_prev = i_k

    skip = 10
    return {
        't': t,
        'u_true': u_true,
        'u_estimated': u_estimated,
        'i_true': i_true,
        'i_estimated': i_estimated,
        'omega_true': omega_true,
        'omega_est': omega_meas,
        'u_rmse': np.sqrt(np.mean((u_true[skip:] - u_estimated[skip:])**2)),
        'i_rmse': np.sqrt(np.mean((i_true[skip:] - i_estimated[skip:])**2)),
        'omega_rmse': np.sqrt(np.mean((omega_true[skip:] - omega_meas[skip:])**2)),
        'label': 'No filter',
    }


def moving_average_estimation(noise_std=0.5, n_samples=500, u_type='step',
                              window=5):
    """
    Baseline 2: Moving average filter on ω, then direct inversion.
    """
    motor = DCMotor()
    T = 0.001

    t, omega_true, omega_meas, i_true, u_true = generate_experiment_data(
        motor, T_sample=T, n_samples=n_samples,
        u_type=u_type, noise_std=noise_std, seed=42
    )

    # Apply moving average
    omega_ma = np.copy(omega_meas)
    for k in range(window, n_samples):
        omega_ma[k] = np.mean(omega_meas[k-window:k+1])

    inverse = InverseHFNN(
        R=motor.R, K_e=motor.K_e, K_t=motor.K_t,
        J=motor.J, b=motor.b,
        L0=motor.L0, i_sat=motor.i_sat, T=T
    )

    i_estimated = np.zeros(n_samples)
    u_estimated = np.zeros(n_samples)
    i_prev = 0.0

    for k in range(1, n_samples):
        i_k = inverse.extract_current(omega_ma[k], omega_ma[k-1])
        i_estimated[k] = i_k
        u_estimated[k] = inverse.reconstruct_input(i_k, i_prev, omega_ma[k])
        i_prev = i_k

    skip = 10
    return {
        't': t,
        'u_true': u_true,
        'u_estimated': u_estimated,
        'i_true': i_true,
        'i_estimated': i_estimated,
        'omega_true': omega_true,
        'omega_est': omega_ma,
        'u_rmse': np.sqrt(np.mean((u_true[skip:] - u_estimated[skip:])**2)),
        'i_rmse': np.sqrt(np.mean((i_true[skip:] - i_estimated[skip:])**2)),
        'omega_rmse': np.sqrt(np.mean((omega_true[skip:] - omega_ma[skip:])**2)),
        'label': f'MA (w={window})',
    }


def run_all_comparisons(noise_std=0.5, n_samples=500, u_type='step'):
    """
    Run all methods and produce comparison table + plots.
    """
    print(f"Running comparisons: σ={noise_std}, N={n_samples}, input={u_type}")
    print("="*70)

    # Proposed method
    res_hfnn = run_dual_estimation(noise_std=noise_std, n_samples=n_samples,
                                   u_type=u_type)
    res_hfnn['label'] = 'HFNN + EKF'

    # Baselines
    res_nofilter = no_filter_estimation(noise_std=noise_std, n_samples=n_samples,
                                         u_type=u_type)
    res_ma = moving_average_estimation(noise_std=noise_std, n_samples=n_samples,
                                        u_type=u_type)

    all_results = [res_hfnn, res_nofilter, res_ma]

    # Print comparison table
    print(f"\n{'Method':<20} | {'ω RMSE':>10} | {'i RMSE':>10} | {'u RMSE':>10}")
    print("-"*70)
    for res in all_results:
        print(f"{res['label']:<20} | {res['omega_rmse']:10.6f} | "
              f"{res['i_rmse']:10.6f} | {res['u_rmse']:10.6f}")
    print("="*70)

    # Comparison plot: u reconstruction
    t_ms = res_hfnn['t'] * 1000
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(t_ms, res_hfnn['u_true'], 'k-', linewidth=2, label='True $u_k$')
    colors = ['r', 'b', 'g']
    styles = ['--', ':', '-.']
    for res, c, s in zip(all_results, colors, styles):
        axes[0].plot(t_ms, res['u_estimated'], linestyle=s, color=c,
                     linewidth=1, alpha=0.8,
                     label=f'{res["label"]} (RMSE={res["u_rmse"]:.3f})')
    axes[0].set_ylabel('$u$ [V]')
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Input Reconstruction Comparison')

    # Error comparison
    for res, c, s in zip(all_results, colors, styles):
        error = res['u_true'] - res['u_estimated']
        axes[1].plot(t_ms, error, linestyle='-', color=c, linewidth=0.5,
                     alpha=0.7, label=res['label'])
    axes[1].set_ylabel('$u$ error [V]')
    axes[1].set_xlabel('Time [ms]')
    axes[1].legend(loc='upper right', fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Reconstruction Error')

    plt.tight_layout()
    plt.savefig('/home/rodo/10Paper/results/comparison.png', dpi=300,
                bbox_inches='tight')
    print("\nComparison plot saved to results/comparison.png")
    plt.close()

    return all_results


if __name__ == '__main__':
    run_all_comparisons(noise_std=0.5, n_samples=500, u_type='step')
