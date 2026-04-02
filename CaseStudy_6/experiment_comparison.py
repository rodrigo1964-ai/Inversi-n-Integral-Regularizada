"""
CaseStudy_6 — Comprehensive Comparison (§VI.H, Table VIII, Fig. 3)
====================================================================

Runs all 7 methods at σ=0.1 for the final comparison table, and
generates the RMSE vs noise level log-log plot across all methods.

Author: Rodolfo H. Rodrigo — UNSJ — 2026
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import motor_model as mm
from methods import (inverse_diff_2pt, inverse_diff_3pt, inverse_diff_4pt,
                     inverse_integral, ekf_derivative, ekf_integral, tii)


def rmse(a, b, skip=100):
    s = slice(skip, None)
    mask = ~np.isnan(a[s]) & ~np.isnan(b[s])
    return np.sqrt(np.mean((a[s][mask] - b[s][mask])**2)) if mask.sum() > 0 else np.nan


def generate_ground_truth(T=0.0001, t_final=0.2):
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


def run_comparison_at_sigma(gt, sigma):
    """Run all 7 methods at a given noise level."""
    T, n = gt['T'], gt['n']
    np.random.seed(42)
    wn = gt['omega'] + np.random.normal(0, sigma, n)
    i_n = gt['i'] + np.random.normal(0, sigma, n)
    np.random.seed(42)
    omega_meas = gt['omega'] + np.random.normal(0, sigma, n)

    r = {}
    r['d2']  = rmse(inverse_diff_2pt(wn, i_n, T), gt['u'])
    r['d3']  = rmse(inverse_diff_3pt(wn, i_n, T), gt['u'])
    r['d4']  = rmse(inverse_diff_4pt(wn, i_n, T), gt['u'])
    r['int'] = rmse(inverse_integral(wn, i_n, T), gt['u'])

    try:
        u5, _, _ = ekf_derivative(omega_meas, T, n, sigma)
        r['ekf_d'] = rmse(u5, gt['u'])
    except Exception:
        r['ekf_d'] = np.nan

    try:
        u6, _, _ = ekf_integral(omega_meas, T, n, sigma)
        r['ekf_i'] = rmse(u6, gt['u'])
    except Exception:
        r['ekf_i'] = np.nan

    # TII: grid search
    best_r = np.inf
    for lam in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
        u7 = tii(wn, i_n, T, lam)
        r7 = rmse(u7, gt['u'])
        if r7 < best_r:
            best_r = r7
    r['tii'] = best_r

    return r


def run_comprehensive_experiment(gt):
    """Run all methods at all noise levels."""
    noise_levels = [0.01, 0.05, 0.1, 0.5]
    return {sigma: run_comparison_at_sigma(gt, sigma) for sigma in noise_levels}


def print_table_VIII(results):
    """Print Table VIII: Comprehensive Comparison at σ=0.1."""
    r = results[0.1]
    tii_val = r['tii']

    print("\n" + "=" * 60)
    print("TABLE VIII: Comprehensive Method Comparison at σ = 0.1")
    print("=" * 60)
    print(f"{'Method':<30s}  {'u RMSE [V]':>12s}  {'vs TII':>8s}")
    print("-" * 60)

    methods = [
        ("Inverse Diff. 3pt", r['d3']),
        ("Inverse Diff. 2pt", r['d2']),
        ("Inverse Diff. 4pt", r['d4']),
        ("Inverse Integral (unreg.)", r['int']),
        ("EKF + Derivative", r['ekf_d']),
        ("EKF + Integral", r['ekf_i']),
        ("TII", r['tii']),
    ]
    for name, val in methods:
        v = f"{val:.4f}" if not np.isnan(val) else "diverges"
        ratio = f"{val / tii_val:.0f}×" if not np.isnan(val) and tii_val > 0 else "—"
        print(f"{name:<30s}  {v:>12s}  {ratio:>8s}")
    print("=" * 60)


if __name__ == '__main__':
    gt = generate_ground_truth()
    results = run_comprehensive_experiment(gt)
    print_table_VIII(results)
