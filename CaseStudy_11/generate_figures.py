import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
import matplotlib.pyplot as plt
from experiment_complexity import run_complexity_experiment, print_table

OUT = os.path.dirname(__file__)

def fig_complexity(results):
    ns = [r['n'] for r in results]
    tii_t = [r['tii_ms'] for r in results]
    ekf_t = [r['ekf_ms'] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(ns, tii_t, 'o-', label='TII')
    ax.loglog(ns, ekf_t, 's-', label='EKF + Integral')
    ax.set_xlabel('Number of samples n')
    ax.set_ylabel('Execution time [ms]')
    ax.set_title('Computational Complexity Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'fig_complexity.png'), dpi=300, bbox_inches='tight')
    plt.close()

def save_table_csv(results):
    import csv
    path = os.path.join(OUT, 'table_complexity.csv')
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['n', 'TII_time_ms', 'EKF_time_ms'])
        for r in results:
            w.writerow([r['n'], f"{r['tii_ms']:.2f}", f"{r['ekf_ms']:.2f}"])

if __name__ == '__main__':
    results = run_complexity_experiment()
    fig_complexity(results)
    save_table_csv(results)
    print_table(results)
