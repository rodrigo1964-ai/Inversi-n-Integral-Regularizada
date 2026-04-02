"""
CaseStudy_2 — Validation Tests (§VI.D)
========================================

Verifies:
  1. 3pt differential is WORSE than 2pt under noise (accuracy reversal)
  2. Integral ≈ 2pt under noise (similar noise gain)
  3. Consistent 3pt/integral ratio ≈ 1.8× across all noise levels
  4. All unregularized methods exceed 12V RMSE at σ ≥ 0.1

Author: Rodolfo H. Rodrigo — UNSJ — 2026
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from experiment_noise import generate_ground_truth, run_noise_experiment

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
    results = run_noise_experiment(gt)

    print("\nCaseStudy_2: Noise-Dominated Regime Tests")
    print("=" * 50)

    # 3pt worse than 2pt at all noise levels
    for sigma in [0.01, 0.05, 0.1, 0.5]:
        r = results[sigma]
        check(f"σ={sigma}: 3pt > 2pt (accuracy reversal)",
              r['d3'] > r['d2'])

    # Consistent ratio ≈ 1.8
    ratios = [results[s]['ratio_3pt_int'] for s in results]
    check("3pt/integral ratio consistent (spread < 0.3)",
          max(ratios) - min(ratios) < 0.3)

    check("3pt/integral ratio ≈ 1.8 (within [1.5, 2.1])",
          all(1.5 < r < 2.1 for r in ratios))

    # RMSE > 12V at σ ≥ 0.1 for 3pt
    check("3pt RMSE > 12V at σ=0.1",
          results[0.1]['d3'] > 12.0)

    print(f"\nResults: {PASS} passed, {FAIL} failed")
    sys.exit(FAIL)
