import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
import matplotlib.pyplot as plt
from experiment_lambda import generate_ground_truth, run_lambda_experiment, print_table

OUT = os.path.dirname(__file__)

def fig4_lambda_sensitivity(results):
    lambdas = results['lambdas']
    sigmas = results['sigmas']

    fig, ax = plt.subplots(figsize=(8, 5))
    markers = ['o', 's', '^', 'D']
    for i, sigma in enumerate(sigmas):
        ax.loglog(lambdas, results['rmse'][sigma], '-', marker=markers[i],
                  markersize=3, label=f'sigma = {sigma}')
    ax.set_xlabel('Regularization parameter lambda')
    ax.set_ylabel('RMSE [V]')
    ax.set_title('TII Sensitivity to Regularization Parameter')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'fig4_lambda_sensitivity.png'), dpi=300, bbox_inches='tight')
    plt.close()

def save_table_csv(results):
    lambdas = results['lambdas']
    sigmas = results['sigmas']
    path = os.path.join(OUT, 'table_lambda.csv')
    header = 'lambda,' + ','.join(f'sigma_{s:.2f}' for s in sigmas)
    data = np.column_stack([lambdas] + [results['rmse'][s] for s in sigmas])
    np.savetxt(path, data, delimiter=',', header=header, comments='', fmt='%.6f')

if __name__ == '__main__':
    gt = generate_ground_truth()
    results = run_lambda_experiment(gt)
    fig4_lambda_sensitivity(results)
    save_table_csv(results)
    print_table(results)
