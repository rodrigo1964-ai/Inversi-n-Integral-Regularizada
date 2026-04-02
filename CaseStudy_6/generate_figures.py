"""
CaseStudy_6 — Generate Figures and Tables (§VI.H)
===================================================

Outputs:
    fig3_noise_scaling.png      — Fig. 3: RMSE vs noise level (log-log)
    table_VIII_comparison.csv   — Table VIII: Comprehensive comparison at σ=0.1

Author: Rodolfo H. Rodrigo — UNSJ — 2026
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiment_comparison import (generate_ground_truth, run_comprehensive_experiment,
                                    print_table_VIII)

OUT = os.path.dirname(__file__)


def fig3_noise_scaling(results):
    """Fig. 3: Input reconstruction RMSE vs measurement noise (log-log)."""
    sigmas = sorted(results.keys())

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.loglog(sigmas, [results[s]['d2'] for s in sigmas], 'bs--', ms=8, lw=1.5,
              label='(1) Inv. Diff. 2pt')
    ax.loglog(sigmas, [results[s]['d3'] for s in sigmas], 'g^--', ms=8, lw=1.5,
              label='(2) Inv. Diff. 3pt')
    ax.loglog(sigmas, [results[s]['d4'] for s in sigmas], 'mv--', ms=8, lw=1.5,
              label='(3) Inv. Diff. 4pt')
    ax.loglog(sigmas, [results[s]['int'] for s in sigmas], 'ro--', ms=8, lw=1.5,
              label='(4) Inv. Integral')

    # EKF methods (may have NaN at high σ)
    ekf_d = [(s, results[s]['ekf_d']) for s in sigmas if not np.isnan(results[s]['ekf_d'])]
    ekf_i = [(s, results[s]['ekf_i']) for s in sigmas if not np.isnan(results[s]['ekf_i'])]
    if ekf_d:
        ax.loglog(*zip(*ekf_d), 'cD--', ms=8, lw=1.5, label='(5) EKF+Deriv')
    if ekf_i:
        ax.loglog(*zip(*ekf_i), 'mP--', ms=8, lw=1.5, label='(6) EKF+Integ')

    ax.loglog(sigmas, [results[s]['tii'] for s in sigmas], 'kH-', ms=12, lw=2.5,
              label='(7) TII')

    ax.set_xlabel('Noise level $\\sigma$ [rad/s, A]', fontsize=12)
    ax.set_ylabel('$u$ RMSE [V]', fontsize=12)
    ax.set_title('Input Reconstruction Error vs Measurement Noise', fontsize=13)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    path = os.path.join(OUT, 'fig3_noise_scaling.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def save_table_csv(results):
    r = results[0.1]
    tii_val = r['tii']
    path = os.path.join(OUT, 'table_VIII_comparison.csv')
    with open(path, 'w') as f:
        f.write("Method,u_RMSE_V,vs_TII\n")
        methods = [
            ("Inverse Diff. 3pt", r['d3']),
            ("Inverse Diff. 2pt", r['d2']),
            ("Inverse Diff. 4pt", r['d4']),
            ("Inverse Integral", r['int']),
            ("EKF + Derivative", r['ekf_d']),
            ("EKF + Integral", r['ekf_i']),
            ("TII", r['tii']),
        ]
        for name, val in methods:
            v = f"{val:.4f}" if not np.isnan(val) else "diverges"
            ratio = f"{val / tii_val:.0f}" if not np.isnan(val) and tii_val > 0 else ""
            f.write(f"{name},{v},{ratio}\n")
    print(f"  Saved: {path}")


if __name__ == '__main__':
    gt = generate_ground_truth()
    results = run_comprehensive_experiment(gt)

    fig3_noise_scaling(results)
    save_table_csv(results)
    print_table_VIII(results)
