import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
import matplotlib.pyplot as plt
from experiment_nongaussian import generate_ground_truth, run_nongaussian_experiment, print_table

OUT = os.path.dirname(__file__)

def save_table_csv(results):
    import csv
    noise_types = ['Gaussian', 'Laplacian', 'Impulsive']
    path = os.path.join(OUT, 'table_nongaussian.csv')
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Method', 'Gaussian_RMSE', 'Laplacian_RMSE', 'Impulsive_RMSE'])
        for method in ['Diff 3pt', 'EKF + Derivative', 'EKF + Integral', 'TII']:
            w.writerow([method] + [f"{results[method][nt]:.3f}" for nt in noise_types])

def fig_nongaussian(results):
    import matplotlib.pyplot as plt
    noise_types = ['Gaussian', 'Laplacian', 'Impulsive']
    methods = ['Diff 3pt', 'EKF + Derivative', 'EKF + Integral', 'TII']
    x = np.arange(len(noise_types))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, method in enumerate(methods):
        vals = [results[method][nt] for nt in noise_types]
        ax.bar(x + i*width, vals, width, label=method)
    ax.set_xticks(x + 1.5*width)
    ax.set_xticklabels(noise_types)
    ax.set_ylabel('RMSE [V]')
    ax.set_title('Non-Gaussian Noise Robustness')
    ax.legend()
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'fig_nongaussian.png'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    gt = generate_ground_truth()
    results = run_nongaussian_experiment(gt)
    fig_nongaussian(results)
    save_table_csv(results)
    print_table(results)
