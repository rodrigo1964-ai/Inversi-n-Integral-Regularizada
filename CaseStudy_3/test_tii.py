"""
CaseStudy_3 — Validation Tests (§VI.E)
========================================

Verifies:
  1. TII RMSE < 1% of input amplitude (12V) at σ=0.1
  2. TII improvement ≥ 85× over unregularized at all noise levels
     (paper reports 90–309×; threshold allows for rounding)
  3. Optimal λ increases with noise (monotone or piecewise)
  4. TII RMSE < 0.2 V across all noise levels

Author: Rodolfo H. Rodrigo — UNSJ — 2026
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from experiment_tii import generate_ground_truth, run_tii_experiment

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
    results = run_tii_experiment(gt)

    print("\nCaseStudy_3: TII Performance Tests")
    print("=" * 50)

    check("TII RMSE < 1% of 12V at σ=0.1",
          results[0.1]['tii'] < 0.12)

    for sigma in [0.01, 0.05, 0.1, 0.5]:
        check(f"σ={sigma}: TII improvement ≥ 85× (got {results[sigma]['factor']:.0f}×)",
              results[sigma]['factor'] >= 85)

    # Optimal λ non-decreasing with σ
    sigmas = sorted(results.keys())
    lambdas = [results[s]['lam'] for s in sigmas]
    check("Optimal λ non-decreasing with σ",
          all(lambdas[i] <= lambdas[i + 1] for i in range(len(lambdas) - 1)))

    check("TII RMSE < 0.2 V at all noise levels",
          all(results[s]['tii'] < 0.2 for s in results))

    print(f"\nResults: {PASS} passed, {FAIL} failed")
    sys.exit(FAIL)
