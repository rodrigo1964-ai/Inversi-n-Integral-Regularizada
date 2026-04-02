"""
CaseStudy_2 — Generate Figures and Tables (§VI.D)
===================================================

Outputs:
    fig2c_noisy_inverse.png   — Fig. 2(c): Noisy inverse, no regularization
    table_IV_noise.csv        — Table IV: RMSE under noise

Author: Rodolfo H. Rodrigo — UNSJ — 2026
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiment_noise import generate_ground_truth, run_noise_experiment, print_table

OUT = os.path.dirname(__file__)


def fig2c_noisy_inverse(gt, results, sigma=0.1):
    """Fig. 2(c): Inverse reconstruction at σ=0.1, no regularization."""
    ms = gt['t'] * 1000
    r = results[sigma]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ms, gt['u'], 'k-', lw=2, label='True $u(t)$')
    ax.plot(ms, r['u_d3'], 'b-', lw=0.3, alpha=0.4,
            label=f'Diff. 3pt (RMSE={r["d3"]:.2f} V)')
    ax.plot(ms, r['u_int'], 'r-', lw=0.5, alpha=0.5,
            label=f'Integral (RMSE={r["int"]:.2f} V)')
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('$u$ [V]')
    ax.set_ylim(-35, 45)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title(f'(c) Inverse Reconstruction with Noise ($\\sigma$={sigma}) — No Regularization')
    plt.tight_layout()
    path = os.path.join(OUT, 'fig2c_noisy_inverse.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def save_table_csv(results):
    path = os.path.join(OUT, 'table_IV_noise.csv')
    with open(path, 'w') as f:
        f.write("sigma,Diff_2pt_V,Diff_3pt_V,Integral_V,ratio_3pt_int\n")
        for sigma in sorted(results.keys()):
            r = results[sigma]
            f.write(f"{sigma},{r['d2']:.4f},{r['d3']:.4f},{r['int']:.4f},{r['ratio_3pt_int']:.2f}\n")
    print(f"  Saved: {path}")


if __name__ == '__main__':
    gt = generate_ground_truth()
    results = run_noise_experiment(gt)

    fig2c_noisy_inverse(gt, results)
    save_table_csv(results)
    print_table(results)
