"""
CaseStudy_5 — Generate Figures and Tables (§VI.G)
===================================================

Outputs:
    fig2f_ekf_comparison.png  — Fig. 2(f): EKF derivative vs integral at σ=0.1
    table_VII_ekf.csv         — Table VII: EKF comparison

Author: Rodolfo H. Rodrigo — UNSJ — 2026
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiment_ekf import generate_ground_truth, run_ekf_experiment, print_table, rmse

OUT = os.path.dirname(__file__)


def fig2f_ekf_comparison(gt, results, sigma=0.1):
    """Fig. 2(f): EKF-based input reconstruction at σ=0.1."""
    ms = gt['t'] * 1000
    r = results[sigma]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ms, gt['u'], 'k-', lw=2, label='True $u(t)$')

    if not r['diverged_d']:
        ax.plot(ms, r['u_ekf_d'], 'b-', lw=0.5, alpha=0.5,
                label=f'EKF+Deriv (RMSE={r["ekf_d"]:.3f} V)')
    if not r['diverged_i']:
        ax.plot(ms, r['u_ekf_i'], 'r-', lw=0.8, alpha=0.7,
                label=f'EKF+Integ (RMSE={r["ekf_i"]:.3f} V)')

    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('$u$ [V]')
    ax.set_ylim(-5, 20)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title(f'(f) EKF-Based Reconstruction ($\\sigma$={sigma})')
    plt.tight_layout()
    path = os.path.join(OUT, 'fig2f_ekf_comparison.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def save_table_csv(results):
    path = os.path.join(OUT, 'table_VII_ekf.csv')
    with open(path, 'w') as f:
        f.write("sigma,u_Deriv_V,u_Integ_V,Improvement\n")
        for sigma in sorted(results.keys()):
            r = results[sigma]
            d = 'diverges' if r['diverged_d'] else f"{r['ekf_d']:.4f}"
            i = 'diverges' if r['diverged_i'] else f"{r['ekf_i']:.4f}"
            imp = f"{r['improvement']:.1f}" if not np.isnan(r.get('improvement', np.nan)) else ''
            f.write(f"{sigma},{d},{i},{imp}\n")
    print(f"  Saved: {path}")


if __name__ == '__main__':
    gt = generate_ground_truth()
    results = run_ekf_experiment(gt)

    fig2f_ekf_comparison(gt, results)
    save_table_csv(results)
    print_table(results)
