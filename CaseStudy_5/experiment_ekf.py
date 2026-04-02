"""
CaseStudy_5 — EKF Derivative vs Integral Reconstruction (§VI.G, Table VII)
============================================================================

Compares EKF + 4pt derivative (Method 5) vs EKF + integral (Method 6)
for input reconstruction from filtered states.

Key finding: integral prevents EKF divergence at σ=0.5.

Author: Rodolfo H. Rodrigo — UNSJ — 2026
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import motor_model as mm
from methods import ekf_derivative, ekf_integral


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


def run_ekf_experiment(gt):
    """Compare EKF + derivative vs EKF + integral at each noise level."""
    T, n = gt['T'], gt['n']
    noise_levels = [0.01, 0.05, 0.1, 0.5]
    results = {}

    for sigma in noise_levels:
        np.random.seed(42)
        omega_meas = gt['omega'] + np.random.normal(0, sigma, n)

        r = {}
        # Method 5: EKF + derivative
        try:
            u5, _, _ = ekf_derivative(omega_meas, T, n, sigma)
            r['ekf_d'] = rmse(u5, gt['u'])
            r['u_ekf_d'] = u5
            r['diverged_d'] = False
        except Exception:
            r['ekf_d'] = np.nan
            r['u_ekf_d'] = np.full(n, np.nan)
            r['diverged_d'] = True

        # Method 6: EKF + integral
        try:
            u6, _, _ = ekf_integral(omega_meas, T, n, sigma)
            r['ekf_i'] = rmse(u6, gt['u'])
            r['u_ekf_i'] = u6
            r['diverged_i'] = False
        except Exception:
            r['ekf_i'] = np.nan
            r['u_ekf_i'] = np.full(n, np.nan)
            r['diverged_i'] = True

        # Improvement factor
        if not np.isnan(r['ekf_d']) and not np.isnan(r['ekf_i']) and r['ekf_i'] > 0:
            r['improvement'] = r['ekf_d'] / r['ekf_i']
        else:
            r['improvement'] = np.nan

        results[sigma] = r

    return results


def print_table(results):
    """Print Table VII: EKF + Input Reconstruction."""
    print("\n" + "=" * 70)
    print("TABLE VII: EKF + Input Reconstruction: Derivative vs. Integral")
    print("=" * 70)
    print(f"{'σ_ω':>6s}  {'u Deriv. [V]':>14s}  {'u Integ. [V]':>14s}  {'Improvement':>12s}")
    print("-" * 70)
    for sigma in sorted(results.keys()):
        r = results[sigma]

        def fmt(v, div):
            return f"{'diverges':>14s}" if div else f"{v:14.2f}"

        imp = f"{r['improvement']:.1f}×" if not np.isnan(r.get('improvement', np.nan)) else "—"
        print(f"{sigma:6.2f}  {fmt(r['ekf_d'], r['diverged_d'])}  "
              f"{fmt(r['ekf_i'], r['diverged_i'])}  {imp:>12s}")
    print("=" * 70)


if __name__ == '__main__':
    gt = generate_ground_truth()
    results = run_ekf_experiment(gt)
    print_table(results)
