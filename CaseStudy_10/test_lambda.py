import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import numpy as np
from experiment_lambda import generate_ground_truth, run_lambda_experiment

PASS = 0
FAIL = 0

def check(name, condition):
    global PASS, FAIL
    if condition:
        print(f"  PASS {name}")
        PASS += 1
    else:
        print(f"  FAIL {name}")
        FAIL += 1

if __name__ == '__main__':
    gt = generate_ground_truth()
    results = run_lambda_experiment(gt)

    print("\nCaseStudy_10: Lambda Sensitivity Tests")
    print("=" * 50)

    lambdas = results['lambdas']
    sigmas = results['sigmas']

    # Test 1: For each sigma, there exists a plateau where RMSE varies < 2x across at least one decade
    for sigma in sigmas:
        rmse_vals = results['rmse'][sigma]
        found_plateau = False
        log_lambdas = np.log10(lambdas)
        for i in range(len(lambdas)):
            # Find all points within one decade above lambdas[i]
            mask = (log_lambdas >= log_lambdas[i]) & (log_lambdas <= log_lambdas[i] + 1.0)
            if mask.sum() >= 2:
                window = rmse_vals[mask]
                if max(window) < 2 * min(window):
                    found_plateau = True
                    break
        check(f"Plateau exists for sigma={sigma}", found_plateau)

    # Test 2: Optimal lambda increases with sigma
    optimal_lambdas = []
    for sigma in sigmas:
        rmse_vals = results['rmse'][sigma]
        idx = np.nanargmin(rmse_vals)
        optimal_lambdas.append(lambdas[idx])
    increasing = all(optimal_lambdas[i] <= optimal_lambdas[i+1] for i in range(len(optimal_lambdas)-1))
    check("Optimal lambda increases with sigma", increasing)

    print(f"\nResults: {PASS} passed, {FAIL} failed")
    sys.exit(FAIL)
