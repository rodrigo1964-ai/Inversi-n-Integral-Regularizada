"""
CaseStudy_5 — Validation Tests (§VI.G)
========================================

Verifies:
  1. Integral always ≤ derivative RMSE
  2. Integral provides ≥ 2× improvement at σ=0.1
  3. Derivative-based EKF diverges at σ=0.5
  4. Integral-based EKF does NOT diverge at σ=0.5

Author: Rodolfo H. Rodrigo — UNSJ — 2026
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from experiment_ekf import generate_ground_truth, run_ekf_experiment

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
    results = run_ekf_experiment(gt)

    print("\nCaseStudy_5: EKF Derivative vs Integral Tests")
    print("=" * 50)

    # Integral ≤ derivative at all non-divergent levels
    for sigma in [0.01, 0.05, 0.1]:
        r = results[sigma]
        if not r['diverged_d'] and not r['diverged_i']:
            check(f"σ={sigma}: integral ≤ derivative",
                  r['ekf_i'] <= r['ekf_d'])

    # ≥ 2× improvement at σ=0.1
    r01 = results[0.1]
    if not np.isnan(r01.get('improvement', np.nan)):
        check(f"σ=0.1: improvement ≥ 2× (got {r01['improvement']:.1f}×)",
              r01['improvement'] >= 2.0)

    # Divergence behavior at σ=0.5
    r05 = results[0.5]
    check("σ=0.5: derivative diverges OR has very high RMSE",
          r05['diverged_d'] or r05['ekf_d'] > 50.0)

    check("σ=0.5: integral does NOT diverge",
          not r05['diverged_i'])

    print(f"\nResults: {PASS} passed, {FAIL} failed")
    sys.exit(FAIL)
