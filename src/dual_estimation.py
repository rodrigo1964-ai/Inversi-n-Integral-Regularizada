"""
Dual Estimation Pipeline
=========================

Full pipeline:
1. EKF filters state [ω, i] from noisy ω measurements (Gaussian noise)
2. Reconstructs unknown input u from filtered state via electrical equation
3. Uses previous û estimate for next EKF prediction (feedback)

Author: Rodolfo H. Rodrigo
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import motor_model as mm
from ekf import EKF


def run_dual_estimation(noise_std=0.1, n_samples=2000, u_type='step',
                        T_sample=0.0001, use_integral=False):
    """
    Run the complete dual estimation pipeline.

    Parameters
    ----------
    noise_std : float
        Std dev of Gaussian noise on ω [rad/s]
    n_samples : int
        Number of samples
    u_type : str
        'step', 'ramp', 'pulse'
    T_sample : float
        Sampling period [s]
    use_integral : bool
        If True, use integral formulation for u reconstruction.
        If False, use 4-point derivative.

    Returns
    -------
    results : dict
    """
    # Generate ground truth
    t, omega_true, omega_meas, i_true, u_true = mm.generate_experiment_data(
        T_sample=T_sample, n_samples=n_samples,
        u_type=u_type, noise_std=noise_std, seed=42
    )

    # EKF setup
    Q = np.diag([1e-2, 1e-2])    # Process noise covariance
    R_meas = noise_std**2         # Measurement noise variance
    x0 = np.array([0.0, 0.0])    # Initial state
    P0 = np.eye(2) * 1.0         # Initial covariance

    ekf = EKF(T=T_sample, Q=Q, R_meas=R_meas, x0=x0, P0=P0,
              use_integral=use_integral)

    # Storage
    omega_filt = np.zeros(n_samples)
    i_filt = np.zeros(n_samples)
    u_hat = np.zeros(n_samples)

    u_est = 0.0  # Initial input estimate for prediction

    for k in range(1, n_samples):
        x_f, P_f, u_recon = ekf.step(omega_meas[k], u_est=u_est)
        omega_filt[k] = x_f[0]
        i_filt[k] = x_f[1]
        u_hat[k] = u_recon

        # Feed back reconstructed u for next prediction
        u_est = u_recon

    # Metrics (skip initial transient)
    skip = max(50, int(0.01 / T_sample))

    results = {
        't': t,
        'omega_true': omega_true,
        'omega_meas': omega_meas,
        'omega_filt': omega_filt,
        'i_true': i_true,
        'i_filt': i_filt,
        'u_true': u_true,
        'u_hat': u_hat,
        'noise_std': noise_std,
        'omega_rmse': np.sqrt(np.mean((omega_true[skip:] - omega_filt[skip:])**2)),
        'i_rmse': np.sqrt(np.mean((i_true[skip:] - i_filt[skip:])**2)),
        'u_rmse': np.sqrt(np.mean((u_true[skip:] - u_hat[skip:])**2)),
        'omega_meas_rmse': np.sqrt(np.mean((omega_true[skip:] - omega_meas[skip:])**2)),
    }

    return results


def plot_results(results, save_path=None):
    """Publication-quality plots."""
    t = results['t']
    ms = t * 1000

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    # Input reconstruction
    axes[0].plot(ms, results['u_true'], 'k-', lw=1.5, label='True $u_k$')
    axes[0].plot(ms, results['u_hat'], 'r--', lw=1, label='Reconstructed $\\hat{u}_k$')
    axes[0].set_ylabel('$u$ [V]')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f'Input Reconstruction (RMSE = {results["u_rmse"]:.4f} V)')

    # Angular velocity
    axes[1].plot(ms, results['omega_meas'], '.', color='gray', alpha=0.3, ms=1, label='Measured')
    axes[1].plot(ms, results['omega_true'], 'b-', lw=1.5, label='True')
    axes[1].plot(ms, results['omega_filt'], 'r--', lw=1, label='Filtered')
    axes[1].set_ylabel('$\\omega$ [rad/s]')
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title(f'$\\omega$ Filtering (RMSE: meas={results["omega_meas_rmse"]:.4f}, filt={results["omega_rmse"]:.4f})')

    # Current
    axes[2].plot(ms, results['i_true'], 'b-', lw=1.5, label='True $i$')
    axes[2].plot(ms, results['i_filt'], 'r--', lw=1, label='Filtered $\\hat{{i}}$')
    axes[2].set_ylabel('$i$ [A]')
    axes[2].legend(loc='best')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title(f'Current Estimation (RMSE = {results["i_rmse"]:.4f} A)')

    # Errors
    axes[3].plot(ms, results['u_true'] - results['u_hat'], 'r-', lw=0.5, alpha=0.7, label='$u$ error')
    axes[3].plot(ms, results['omega_true'] - results['omega_filt'], 'b-', lw=0.5, alpha=0.7, label='$\\omega$ error')
    axes[3].set_ylabel('Error')
    axes[3].set_xlabel('Time [ms]')
    axes[3].legend(loc='best')
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.close()


def run_noise_comparison():
    """Compare estimation performance across noise levels."""
    noise_levels = [0.01, 0.05, 0.1, 0.5, 1.0]

    print(f"\n{'σ':>6} | {'ω RMSE':>10} | {'i RMSE':>10} | {'u RMSE':>10}")
    print("-" * 50)

    for sigma in noise_levels:
        res = run_dual_estimation(noise_std=sigma, n_samples=2000, u_type='step')
        print(f"{sigma:6.2f} | {res['omega_rmse']:10.4f} | "
              f"{res['i_rmse']:10.4f} | {res['u_rmse']:10.4f}")


if __name__ == '__main__':
    noise_levels = [0.01, 0.05, 0.1, 0.5, 1.0]

    print("=== EKF + Reconstrucción de u: Derivada vs Integral ===\n")
    print(f"{'σ':>6} | {'u RMSE (deriv)':>14} | {'u RMSE (integ)':>14} | {'ω RMSE':>10} | {'i RMSE':>10}")
    print("-" * 65)

    all_res = {}
    for sigma in noise_levels:
        rd = run_dual_estimation(noise_std=sigma, use_integral=False)
        ri = run_dual_estimation(noise_std=sigma, use_integral=True)
        print(f"{sigma:6.2f} | {rd['u_rmse']:14.4f} | {ri['u_rmse']:14.4f} | "
              f"{ri['omega_rmse']:10.4f} | {ri['i_rmse']:10.4f}")
        all_res[sigma] = {'deriv': rd, 'integ': ri}

    # Plot for σ=0.1
    res_d = all_res[0.1]['deriv']
    res_i = all_res[0.1]['integ']
    ms = res_d['t'] * 1000

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(ms, res_d['u_true'], 'k-', lw=2, label='True $u$')
    axes[0].plot(ms, res_d['u_hat'], 'b-', lw=0.5, alpha=0.6,
                 label=f'EKF+Deriv (RMSE={res_d["u_rmse"]:.3f})')
    axes[0].plot(ms, res_i['u_hat'], 'r-', lw=0.8, alpha=0.8,
                 label=f'EKF+Integral (RMSE={res_i["u_rmse"]:.3f})')
    axes[0].set_ylabel('$u$ [V]')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f'Reconstrucción de $u$ ($\\sigma$=0.1)')

    axes[1].plot(ms, res_i['omega_meas'], '.', color='gray', ms=0.5, alpha=0.3, label='Medido')
    axes[1].plot(ms, res_i['omega_true'], 'k-', lw=1.5, label='True')
    axes[1].plot(ms, res_i['omega_filt'], 'r--', lw=1, label='EKF filtrado')
    axes[1].set_ylabel('$\\omega$ [rad/s]')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(ms, res_i['i_true'], 'k-', lw=1.5, label='True $i$')
    axes[2].plot(ms, res_i['i_filt'], 'r--', lw=1, label='EKF $\\hat{i}$')
    axes[2].set_ylabel('$i$ [A]')
    axes[2].set_xlabel('Time [ms]')
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('EKF + Inversa Integral vs Derivada', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/home/rodo/10Paper/results/ekf_deriv_vs_integral.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved to results/ekf_deriv_vs_integral.png")
    plt.close()
