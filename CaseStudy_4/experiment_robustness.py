"""
CaseStudy_4 — Direct Regressor Robustness (§VI.F, Table VI)
=============================================================

Tests how the direct regressor (state prediction via RK4) degrades
when the input u is corrupted by noise, simulating the feedback loop
where TII's output feeds the next EKF prediction.

Author: Rodolfo H. Rodrigo — UNSJ — 2026
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import motor_model as mm


def rmse(a, b, skip=100):
    s = slice(skip, None)
    return np.sqrt(np.mean((a[s] - b[s])**2))


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


def run_robustness_experiment(gt):
    """Run direct regressor with noisy input u + N(0, σ_u²)."""
    T, n = gt['T'], gt['n']
    sigma_u_levels = [0.1, 0.5, 1.0, 2.0]
    results = {}

    for sigma_u in sigma_u_levels:
        np.random.seed(42)
        u_noisy = gt['u'] + np.random.normal(0, sigma_u, n)

        # Forward-integrate using noisy u
        omega_pred = np.zeros(n)
        i_pred = np.zeros(n)
        omega_pred[0] = gt['omega'][0]
        i_pred[0] = gt['i'][0]

        for k in range(n - 1):
            x_k = np.array([omega_pred[k], i_pred[k]])
            x_next = mm.rk4_step(x_k, u_noisy[k], T)
            omega_pred[k + 1] = x_next[0]
            i_pred[k + 1] = x_next[1]

        results[sigma_u] = {
            'omega_rmse': rmse(omega_pred, gt['omega']),
            'i_rmse': rmse(i_pred, gt['i']),
        }

    return results


def print_table(results):
    """Print Table VI: Direct Regressor Robustness to Input Noise."""
    print("\n" + "=" * 60)
    print("TABLE VI: Direct Regressor Robustness to Input Noise")
    print("=" * 60)
    print(f"{'σ_u [V]':>10s}  {'ω RMSE [rad/s]':>16s}  {'i RMSE [A]':>14s}")
    print("-" * 60)
    for sigma_u in sorted(results.keys()):
        r = results[sigma_u]
        print(f"{sigma_u:10.1f}  {r['omega_rmse']:16.2e}  {r['i_rmse']:14.2e}")
    print("=" * 60)


if __name__ == '__main__':
    gt = generate_ground_truth()
    results = run_robustness_experiment(gt)
    print_table(results)
