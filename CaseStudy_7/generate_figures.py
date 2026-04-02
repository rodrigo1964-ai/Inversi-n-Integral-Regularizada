import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
import matplotlib.pyplot as plt
from experiment_ablation import generate_ground_truth, run_ablation_experiment, print_table

OUT = os.path.dirname(__file__)

def fig_ablation(gt, results):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    t = gt['t'] * 1000  # ms
    u_true = gt['u']
    for ax, r in zip(axes.flat, results):
        ax.plot(t, u_true, 'k-', lw=0.5, label='True')
        ax.plot(t, r['u_est'], 'r-', lw=0.5, alpha=0.7, label=r['name'])
        ax.set_ylabel('Voltage [V]')
        ax.set_title(f"{r['name']} (RMSE={r['rmse']:.3f} V)")
        ax.legend(fontsize=8)
    axes[1, 0].set_xlabel('Time [ms]')
    axes[1, 1].set_xlabel('Time [ms]')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'fig_ablation.png'), dpi=300, bbox_inches='tight')
    plt.close()

def save_table_csv(results):
    import csv
    path = os.path.join(OUT, 'table_ablation.csv')
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Configuration', 'RMSE_V', 'MAE_V', 'Correlation', 'Stable'])
        for r in results:
            w.writerow([r['name'], f"{r['rmse']:.3f}", f"{r['mae']:.3f}",
                        f"{r['corr']:.4f}", r['stable']])

if __name__ == '__main__':
    gt = generate_ground_truth()
    results = run_ablation_experiment(gt)
    fig_ablation(gt, results)
    save_table_csv(results)
    print_table(results)
