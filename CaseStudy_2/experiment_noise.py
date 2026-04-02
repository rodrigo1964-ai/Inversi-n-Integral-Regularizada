"""
CaseStudy_2 — Noise-Dominated Regime (§VI.D, Table IV)
========================================================

Shows the reversal of accuracy under noise: higher-order stencils
amplify noise more aggressively, making 3pt worse than 2pt.
The integral formulation matches 2pt noise scaling but with
better truncation behavior.

Author: Rodolfo H. Rodrigo — UNSJ — 2026
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import motor_model as mm
from methods import inverse_diff_2pt, inverse_diff_3pt, inverse_integral


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


def run_noise_experiment(gt):
    """Sweep σ ∈ {0.01, 0.05, 0.1, 0.5}, return RMSE dict per noise level."""
    T, n = gt['T'], gt['n']
    noise_levels = [0.01, 0.05, 0.1, 0.5]
    results = {}

    for sigma in noise_levels:
        np.random.seed(42)
        wn = gt['omega'] + np.random.normal(0, sigma, n)
        i_n = gt['i'] + np.random.normal(0, sigma, n)

        r = {}
        r['d2']  = rmse(inverse_diff_2pt(wn, i_n, T), gt['u'])
        r['d3']  = rmse(inverse_diff_3pt(wn, i_n, T), gt['u'])
        r['int'] = rmse(inverse_integral(wn, i_n, T), gt['u'])
        r['ratio_3pt_int'] = r['d3'] / r['int'] if r['int'] > 0 else np.nan

        # Store arrays for figures
        r['u_d3']  = inverse_diff_3pt(wn, i_n, T)
        r['u_int'] = inverse_integral(wn, i_n, T)

        results[sigma] = r

    return results


def print_table(results):
    """Print Table IV: Inverse Input Reconstruction Under Noise."""
    print("\n" + "=" * 70)
    print("TABLE IV: Inverse Input Reconstruction Under Noise (T = 0.1 ms)")
    print("=" * 70)
    print(f"{'σ':>6s}  {'Diff. 2pt [V]':>14s}  {'Diff. 3pt [V]':>14s}  "
          f"{'Integral [V]':>14s}  {'3pt/Int':>8s}")
    print("-" * 70)
    for sigma in sorted(results.keys()):
        r = results[sigma]
        print(f"{sigma:6.2f}  {r['d2']:14.2f}  {r['d3']:14.2f}  "
              f"{r['int']:14.2f}  {r['ratio_3pt_int']:7.1f}×")
    print("=" * 70)


if __name__ == '__main__':
    gt = generate_ground_truth()
    results = run_noise_experiment(gt)
    print_table(results)
