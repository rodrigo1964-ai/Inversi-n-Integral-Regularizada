"""
CaseStudy_1 — Generate Figures and Tables (§VI.C)
===================================================

Outputs:
    fig1_ground_truth.png     — Fig. 1: Ground truth trajectories
    fig2a_direct_clean.png    — Fig. 2(a): Direct regressors vs RK4
    fig2b_inverse_clean.png   — Fig. 2(b): Inverse reconstruction, clean data
    table_III_clean.csv       — Table III: Formulation accuracy

Author: Rodolfo H. Rodrigo — UNSJ — 2026
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import motor_model as mm
from methods import inverse_diff_3pt, inverse_integral
from experiment_clean import generate_ground_truth, run_clean_experiment, rmse

OUT = os.path.dirname(__file__)


def fig1_ground_truth(gt):
    """Fig. 1: Ground truth trajectories (u, ω, i)."""
    ms = gt['t'] * 1000
    sigma = 0.1
    np.random.seed(42)
    omega_noisy = gt['omega'] + np.random.normal(0, sigma, gt['n'])

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(ms, gt['u'], 'k-', lw=2)
    axes[0].set_ylabel('$u$ [V]')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Step input voltage $u(t)$')

    axes[1].plot(ms, gt['omega'], 'b-', lw=1.5, label='True')
    axes[1].plot(ms, omega_noisy, 'r.', alpha=0.15, ms=1, label=f'Measured ($\\sigma$={sigma})')
    axes[1].set_ylabel('$\\omega$ [rad/s]')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Angular velocity $\\omega(t)$')

    axes[2].plot(ms, gt['i'], 'g-', lw=1.5)
    axes[2].set_ylabel('$i$ [A]')
    axes[2].set_xlabel('Time [ms]')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title('Armature current $i(t)$ — transient peak due to $L(i)$')

    plt.tight_layout()
    path = os.path.join(OUT, 'fig1_ground_truth.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def fig2a_direct_clean(gt):
    """Fig. 2(a): Direct regressors (differential and integral) vs RK4."""
    ms = gt['t'] * 1000
    T = gt['T']

    # Direct differential regressor (Euler forward for ω)
    omega_diff = np.full(gt['n'], np.nan)
    omega_int = np.full(gt['n'], np.nan)
    omega_diff[0] = gt['omega'][0]
    omega_int[0] = gt['omega'][0]

    for k in range(gt['n'] - 1):
        # Euler direct: J·(ω_{k+1}-ω_k)/T = K_t·i_k - b·ω_k - N_load(ω_k)
        dw = (mm.K_t * gt['i'][k] - mm.b * gt['omega'][k] - mm.N_load(gt['omega'][k])) / mm.J
        omega_diff[k + 1] = gt['omega'][k] + T * dw

    # Trapezoidal direct: J·(ω_{k+1}-ω_k) + (T/2)·[h(ω_k,i_k)+h(ω_{k+1},i_{k+1})] = 0
    for k in range(gt['n'] - 1):
        h_k = mm.b * gt['omega'][k] + mm.N_load(gt['omega'][k]) - mm.K_t * gt['i'][k]
        h_k1 = mm.b * gt['omega'][k + 1] + mm.N_load(gt['omega'][k + 1]) - mm.K_t * gt['i'][k + 1]
        omega_int[k + 1] = gt['omega'][k] - (T / (2 * mm.J)) * (h_k + h_k1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ms, gt['omega'], 'k-', lw=2, label='RK4 reference')
    ax.plot(ms, omega_diff, 'b--', lw=1, alpha=0.8,
            label=f'Direct differential ({rmse(omega_diff, gt["omega"]):.2e} rad/s)')
    ax.plot(ms, omega_int, 'r:', lw=1.2, alpha=0.8,
            label=f'Direct integral ({rmse(omega_int, gt["omega"]):.2e} rad/s)')
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('$\\omega$ [rad/s]')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title('(a) Direct Regressors vs RK4 — Clean Data')
    plt.tight_layout()
    path = os.path.join(OUT, 'fig2a_direct_clean.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def fig2b_inverse_clean(gt, results):
    """Fig. 2(b): Inverse reconstruction on clean data."""
    ms = gt['t'] * 1000
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ms, gt['u'], 'k-', lw=2, label='True $u(t)$')
    ax.plot(ms, results['u_diff_3pt'], 'b--', lw=1, alpha=0.8,
            label=f'Diff. 3pt (RMSE={results["diff_3pt"]:.1e} V)')
    ax.plot(ms, results['u_integral'], 'r:', lw=1.2, alpha=0.8,
            label=f'Integral (RMSE={results["integral"]:.1e} V)')
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('$u$ [V]')
    ax.set_ylim(-1, 14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title('(b) Inverse Reconstruction — Clean Data')
    plt.tight_layout()
    path = os.path.join(OUT, 'fig2b_inverse_clean.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def save_table_csv(results):
    """Save Table III as CSV."""
    path = os.path.join(OUT, 'table_III_clean.csv')
    with open(path, 'w') as f:
        f.write("Formulation,u_RMSE_V,Order\n")
        rows = [
            ('Inverse Diff. 2pt', results['diff_2pt'], 'O(T)'),
            ('Inverse Diff. 3pt', results['diff_3pt'], 'O(T^2)'),
            ('Inverse Diff. 4pt', results['diff_4pt'], 'O(T^3)'),
            ('Inverse Integral',  results['integral'],  'O(T^2)'),
        ]
        for name, val, order in rows:
            f.write(f"{name},{val:.6e},{order}\n")
    print(f"  Saved: {path}")


if __name__ == '__main__':
    np.random.seed(42)
    gt = generate_ground_truth()
    results = run_clean_experiment(gt)

    fig1_ground_truth(gt)
    fig2a_direct_clean(gt)
    fig2b_inverse_clean(gt, results)
    save_table_csv(results)

    # Print table to stdout
    from experiment_clean import print_table
    print_table(results)
