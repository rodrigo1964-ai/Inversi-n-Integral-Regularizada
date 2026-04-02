"""
CaseStudy_3 — Generate Figures and Tables (§VI.E)
===================================================

Outputs:
    fig2d_tii.png              — Fig. 2(d): TII at σ=0.1
    fig2e_tii_improvement.png  — Fig. 2(e): TII improvement across noise levels
    table_V_tii.csv            — Table V: TII performance

Author: Rodolfo H. Rodrigo — UNSJ — 2026
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiment_tii import generate_ground_truth, run_tii_experiment, print_table

OUT = os.path.dirname(__file__)


def fig2d_tii(gt, results, sigma=0.1):
    """Fig. 2(d): TII at σ=0.1."""
    ms = gt['t'] * 1000
    r = results[sigma]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ms, gt['u'], 'k-', lw=2, label='True $u(t)$')
    ax.plot(ms[1:], r['u_unreg'][1:], color='gray', lw=0.3, alpha=0.3,
            label=f'Unreg. (RMSE={r["unreg"]:.2f} V)')
    ax.plot(ms[1:], r['u_tii'][1:], 'r-', lw=1.5,
            label=f'TII $\\lambda$={r["lam"]:.0e} (RMSE={r["tii"]:.4f} V)')
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('$u$ [V]')
    ax.set_ylim(-2, 16)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title(f'(d) Tikhonov Integral Inversion ($\\sigma$={sigma})')
    plt.tight_layout()
    path = os.path.join(OUT, 'fig2d_tii.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def fig2e_tii_improvement(results):
    """Fig. 2(e): TII improvement across noise levels (bar chart, log scale)."""
    sigmas = sorted(results.keys())
    unreg = [results[s]['unreg'] for s in sigmas]
    tii_v = [results[s]['tii'] for s in sigmas]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(sigmas))
    w = 0.35
    ax.bar(x - w / 2, unreg, w, label='Unregularized', color='#d62728', alpha=0.7)
    ax.bar(x + w / 2, tii_v, w, label='TII', color='#2ca02c', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s}' for s in sigmas])
    ax.set_xlabel('Noise $\\sigma$')
    ax.set_ylabel('$u$ RMSE [V]')
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_title('(e) TII Improvement Across Noise Levels')
    for i, (u, t) in enumerate(zip(unreg, tii_v)):
        ax.text(i + w / 2, t * 1.4, f'{u / t:.0f}×', ha='center',
                fontsize=9, fontweight='bold', color='#2ca02c')
    plt.tight_layout()
    path = os.path.join(OUT, 'fig2e_tii_improvement.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def save_table_csv(results):
    path = os.path.join(OUT, 'table_V_tii.csv')
    with open(path, 'w') as f:
        f.write("sigma,Unreg_V,Best_lambda,TII_V,Factor\n")
        for sigma in sorted(results.keys()):
            r = results[sigma]
            f.write(f"{sigma},{r['unreg']:.4f},{r['lam']:.0e},{r['tii']:.4f},{r['factor']:.0f}\n")
    print(f"  Saved: {path}")


if __name__ == '__main__':
    gt = generate_ground_truth()
    results = run_tii_experiment(gt)

    fig2d_tii(gt, results)
    fig2e_tii_improvement(results)
    save_table_csv(results)
    print_table(results)
