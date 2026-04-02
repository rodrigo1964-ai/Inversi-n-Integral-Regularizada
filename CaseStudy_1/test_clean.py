"""
CaseStudy_1 — Validation Tests (§VI.C)
========================================

Verifies:
  1. 4pt differential achieves lowest clean-data error (O(T³))
  2. Integral error falls between 3pt and 4pt
  3. All methods < 0.01 V on clean data
  4. Truncation order hierarchy: 2pt > 3pt > integral > 4pt

Author: Rodolfo H. Rodrigo — UNSJ — 2026
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from experiment_clean import generate_ground_truth, run_clean_experiment

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
    np.random.seed(42)
    gt = generate_ground_truth()
    r = run_clean_experiment(gt)

    print("\nCaseStudy_1: Clean Data Accuracy Tests")
    print("=" * 50)

    check("4pt achieves lowest RMSE",
          r['diff_4pt'] < r['diff_3pt'] and r['diff_4pt'] < r['integral'])

    check("Integral between 3pt and 4pt",
          r['diff_4pt'] < r['integral'] < r['diff_3pt'])

    check("All methods < 0.01 V on clean data",
          all(v < 0.01 for v in [r['diff_2pt'], r['diff_3pt'], r['diff_4pt'], r['integral']]))

    check("Truncation hierarchy: 2pt > 3pt > integral > 4pt",
          r['diff_2pt'] > r['diff_3pt'] > r['integral'] > r['diff_4pt'])

    print(f"\nResults: {PASS} passed, {FAIL} failed")
    sys.exit(FAIL)
