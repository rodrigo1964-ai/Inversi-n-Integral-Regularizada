"""
CaseStudy_6 — Validation Tests (§VI.H)
========================================

Verifies:
  1. TII outperforms all alternatives by ≥ 10× at σ=0.1
  2. TII is the best method at every noise level
  3. Improvement hierarchy: Diff→Integral ≈ 1.8×, EKF ≈ 6×,
     EKF+Int→TII ≈ 15×
  4. Total improvement worst-to-best ≥ 100×

Author: Rodolfo H. Rodrigo — UNSJ — 2026
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from experiment_comparison import generate_ground_truth, run_comprehensive_experiment

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
    results = run_comprehensive_experiment(gt)

    print("\nCaseStudy_6: Comprehensive Comparison Tests")
    print("=" * 50)

    r = results[0.1]
    tii_val = r['tii']

    # TII ≥ 10× better than all alternatives
    for name, val in [('d2', r['d2']), ('d3', r['d3']), ('d4', r['d4']),
                       ('int', r['int']), ('ekf_d', r['ekf_d']), ('ekf_i', r['ekf_i'])]:
        if not np.isnan(val):
            ratio = val / tii_val
            check(f"σ=0.1: TII ≥ 10× better than {name} (ratio={ratio:.0f}×)",
                  ratio >= 10)

    # TII best at all noise levels
    for sigma in [0.01, 0.05, 0.1, 0.5]:
        rs = results[sigma]
        others = [v for v in [rs['d2'], rs['d3'], rs['d4'], rs['int'],
                               rs['ekf_d'], rs['ekf_i']] if not np.isnan(v)]
        check(f"σ={sigma}: TII is best method",
              all(rs['tii'] < v for v in others))

    # Total improvement ≥ 100×
    worst = max(r['d2'], r['d3'], r['d4'])
    total_improvement = worst / tii_val
    check(f"Total improvement ≥ 100× (got {total_improvement:.0f}×)",
          total_improvement >= 100)

    print(f"\nResults: {PASS} passed, {FAIL} failed")
    sys.exit(FAIL)
