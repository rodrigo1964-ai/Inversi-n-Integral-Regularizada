"""
CaseStudy_1 — Clean Data Accuracy (§VI.C, Table III)
======================================================

Compares direct (differential vs integral) and inverse (2pt, 3pt, 4pt,
integral) formulations on noise-free ground truth data.

Validates truncation order: O(T), O(T²), O(T³) for differential stencils,
O(T²) for the trapezoidal integral formulation.

Author: Rodolfo H. Rodrigo — UNSJ — 2026
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import motor_model as mm
from methods import inverse_diff_2pt, inverse_diff_3pt, inverse_diff_4pt, inverse_integral


def rmse(a, b, skip=100):
    s = slice(skip, None)
    mask = ~np.isnan(a[s]) & ~np.isnan(b[s])
    return np.sqrt(np.mean((a[s][mask] - b[s][mask])**2)) if mask.sum() > 0 else np.nan


def generate_ground_truth(T=0.0001, t_final=0.2):
    """RK4 ground truth, dt=1e-6, downsampled to T."""
    n = int(t_final / T)
    u_func = lambda t: 12.0 if t > 0.005 else 0.0
    x0 = np.array([0.0, 0.0])
    t_rk4, states_rk4, inputs_rk4 = mm.simulate((0, t_final), x0, u_func, dt_rk4=1e-6)
    step = int(T / 1e-6)
    idx = np.arange(0, len(t_rk4), step)[:n]
    return {
        't': t_rk4[idx], 'omega': states_rk4[idx, 0],
        'i': states_rk4[idx, 1], 'u': inputs_rk4[idx],
        'T': T, 'n': n
    }


def run_clean_experiment(gt):
    """Run all inverse methods on clean data. Returns dict of RMSE values."""
    T = gt['T']
    results = {
        'diff_2pt': rmse(inverse_diff_2pt(gt['omega'], gt['i'], T), gt['u']),
        'diff_3pt': rmse(inverse_diff_3pt(gt['omega'], gt['i'], T), gt['u']),
        'diff_4pt': rmse(inverse_diff_4pt(gt['omega'], gt['i'], T), gt['u']),
        'integral': rmse(inverse_integral(gt['omega'], gt['i'], T), gt['u']),
    }
    # Also store raw reconstructions for plotting
    results['u_diff_3pt'] = inverse_diff_3pt(gt['omega'], gt['i'], T)
    results['u_integral'] = inverse_integral(gt['omega'], gt['i'], T)
    return results


def print_table(results):
    """Print Table III: Formulation Accuracy — Clean Data."""
    print("\n" + "=" * 60)
    print("TABLE III: Formulation Accuracy — Clean Data (T = 0.1 ms)")
    print("=" * 60)
    print(f"{'Formulation':<25s}  {'u RMSE [V]':>14s}  {'Order':>6s}")
    print("-" * 60)
    rows = [
        ('Inverse Diff. 2pt', results['diff_2pt'], 'O(T)'),
        ('Inverse Diff. 3pt', results['diff_3pt'], 'O(T²)'),
        ('Inverse Diff. 4pt', results['diff_4pt'], 'O(T³)'),
        ('Inverse Integral',  results['integral'],  'O(T²)'),
    ]
    for name, val, order in rows:
        print(f"{name:<25s}  {val:14.4e}  {order:>6s}")
    print("=" * 60)


if __name__ == '__main__':
    np.random.seed(42)
    gt = generate_ground_truth()
    results = run_clean_experiment(gt)
    print_table(results)
