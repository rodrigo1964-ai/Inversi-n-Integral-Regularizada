import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
import matplotlib.pyplot as plt
from experiment_mismatch import run_mismatch_experiment, print_table

OUT = os.path.dirname(__file__)

def save_table_csv(results):
    import csv
    deltas = [0.0, 0.1, 0.2, 0.3]
    path = os.path.join(OUT, 'table_mismatch.csv')
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Method'] + [f'delta_{d}' for d in deltas])
        for method in ['EKF + Derivative', 'EKF + Integral', 'TII']:
            w.writerow([method] + [f"{results[method][d]:.3f}" for d in deltas])

def fig_mismatch(results):
    deltas = [0.0, 0.1, 0.2, 0.3]
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in ['EKF + Derivative', 'EKF + Integral', 'TII']:
        vals = [results[method][d] for d in deltas]
        ax.plot(deltas, vals, 'o-', label=method)
    ax.set_xlabel('Parameter mismatch delta')
    ax.set_ylabel('RMSE [V]')
    ax.set_title('Model Mismatch Sensitivity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'fig_mismatch.png'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    results = run_mismatch_experiment()
    fig_mismatch(results)
    save_table_csv(results)
    print_table(results)
