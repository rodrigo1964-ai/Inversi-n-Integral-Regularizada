"""
CaseStudy_4 — Generate Figures and Tables (§VI.F)
===================================================

Outputs:
    table_VI_robustness.csv  — Table VI: Direct regressor robustness

(No figures in this experiment — table only.)

Author: Rodolfo H. Rodrigo — UNSJ — 2026
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from experiment_robustness import generate_ground_truth, run_robustness_experiment, print_table

OUT = os.path.dirname(__file__)


def save_table_csv(results):
    path = os.path.join(OUT, 'table_VI_robustness.csv')
    with open(path, 'w') as f:
        f.write("sigma_u_V,omega_RMSE_rad_s,i_RMSE_A\n")
        for sigma_u in sorted(results.keys()):
            r = results[sigma_u]
            f.write(f"{sigma_u},{r['omega_rmse']:.6e},{r['i_rmse']:.6e}\n")
    print(f"  Saved: {path}")


if __name__ == '__main__':
    gt = generate_ground_truth()
    results = run_robustness_experiment(gt)
    save_table_csv(results)
    print_table(results)
