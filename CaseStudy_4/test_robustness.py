"""
CaseStudy_4 — Validation Tests (§VI.F)
========================================

Verifies:
  1. State RMSE scales approximately with input noise σ_u
  2. At σ_u = 0.1 V (TII quality), errors near clean baseline
  3. ω RMSE < 0.05 rad/s at σ_u = 0.1 V

Author: Rodolfo H. Rodrigo — UNSJ — 2026
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from experiment_robustness import generate_ground_truth, run_robustness_experiment

PASS = 0
FAIL = 0


def check(name, condition):
    global PASS, FAIL
    if condition:
        print(f"  ✓ {name}")
        PASS += 1
    else:
        print(f"  ✗ {name}")
        FAIL += 1


if __name__ == '__main__':
    gt = generate_ground_truth()
    results = run_robustness_experiment(gt)

    print("\nCaseStudy_4: Direct Regressor Robustness Tests")
    print("=" * 50)

    # Scaling: RMSE(2.0)/RMSE(0.1) should be > 5 (sublinear OK due to nonlinear dynamics)
    ratio = results[2.0]['omega_rmse'] / results[0.1]['omega_rmse']
    check(f"ω RMSE scales with σ_u (ratio 2.0/0.1 = {ratio:.1f}, expect > 5)",
          ratio > 5)

    check("ω RMSE < 0.05 rad/s at σ_u = 0.1 V",
          results[0.1]['omega_rmse'] < 0.05)

    check("i RMSE < 0.02 A at σ_u = 0.1 V",
          results[0.1]['i_rmse'] < 0.02)

    # Monotone increase
    sigmas = sorted(results.keys())
    omega_rmses = [results[s]['omega_rmse'] for s in sigmas]
    check("ω RMSE monotonically increases with σ_u",
          all(omega_rmses[i] < omega_rmses[i + 1] for i in range(len(omega_rmses) - 1)))

    print(f"\nResults: {PASS} passed, {FAIL} failed")
    sys.exit(FAIL)
