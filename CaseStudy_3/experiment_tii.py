"""
CaseStudy_3 — TII Performance (§VI.E, Table V)
=================================================

Evaluates Tikhonov Integral Inversion across noise levels.
Shows the two-layer suppression: integral (Layer 1) + Tikhonov (Layer 2).

Author: Rodolfo H. Rodrigo — UNSJ — 2026
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import motor_model as mm
from methods import inverse_integral, tii


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


def run_tii_experiment(gt):
    """Sweep σ, grid-search λ, return TII RMSE and optimal λ."""
    T, n = gt['T'], gt['n']
    noise_levels = [0.01, 0.05, 0.1, 0.5]
    lambda_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    results = {}

    for sigma in noise_levels:
        np.random.seed(42)
        wn = gt['omega'] + np.random.normal(0, sigma, n)
        i_n = gt['i'] + np.random.normal(0, sigma, n)

        # Unregularized baseline
        u_unreg = inverse_integral(wn, i_n, T)
        rmse_unreg = rmse(u_unreg, gt['u'])

        # Grid search for best λ
        best_r, best_lam, best_u = np.inf, None, None
        for lam in lambda_values:
            u7 = tii(wn, i_n, T, lam)
            r7 = rmse(u7, gt['u'])
            if r7 < best_r:
                best_r, best_lam, best_u = r7, lam, u7

        factor = rmse_unreg / best_r if best_r > 0 else np.inf
        results[sigma] = {
            'unreg': rmse_unreg, 'lam': best_lam, 'tii': best_r,
            'factor': factor, 'u_unreg': u_unreg, 'u_tii': best_u
        }

    return results


def print_table(results):
    """Print Table V: TII Performance."""
    print("\n" + "=" * 70)
    print("TABLE V: Tikhonov Integral Inversion (TII) Performance")
    print("=" * 70)
    print(f"{'σ':>6s}  {'Unreg. [V]':>12s}  {'Best λ':>8s}  {'TII [V]':>12s}  {'Factor':>8s}")
    print("-" * 70)
    for sigma in sorted(results.keys()):
        r = results[sigma]
        print(f"{sigma:6.2f}  {r['unreg']:12.4f}  {r['lam']:8.0e}  "
              f"{r['tii']:12.4f}  {r['factor']:7.0f}×")
    print("=" * 70)


if __name__ == '__main__':
    gt = generate_ground_truth()
    results = run_tii_experiment(gt)
    print_table(results)
