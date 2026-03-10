"""
Reproduce All Paper Results — Tables and Figures
==================================================

Runs all 7 methods from methods.py and generates:
  - Table II:  Clean data accuracy (no noise)
  - Table III: Noisy data comparison (σ = 0.01, 0.05, 0.1, 0.5)
  - Table IV:  TII results with optimal λ
  - Table V:   Method comparison summary at σ = 0.1
  - Fig. 3:    Publication figure (6 panels)
  - Fig. 4:    RMSE vs noise level (log-log)

Usage:
    python3 run_experiments.py

Author: Rodolfo H. Rodrigo
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import motor_model as mm
from methods import (inverse_diff_2pt, inverse_diff_3pt, inverse_diff_4pt,
                     inverse_integral, ekf_derivative, ekf_integral, tii, Phi)


# ======================================================================
# Ground truth generation
# ======================================================================

def generate_ground_truth(T=0.0001, t_final=0.2):
    n = int(t_final / T)
    u_func = lambda t: 12.0 if t > 0.005 else 0.0
    x0 = np.array([0.0, 0.0])
    t_rk4, states_rk4, inputs_rk4 = mm.simulate(
        (0, t_final), x0, u_func, dt_rk4=1e-6)
    step = int(T / 1e-6)
    idx = np.arange(0, len(t_rk4), step)[:n]
    return {
        't': t_rk4[idx], 'omega': states_rk4[idx, 0],
        'i': states_rk4[idx, 1], 'u': inputs_rk4[idx],
        'T': T, 'n': n}


def rmse(a, b, skip=100):
    s = slice(skip, None)
    mask = ~np.isnan(a[s]) & ~np.isnan(b[s])
    return np.sqrt(np.mean((a[s][mask] - b[s][mask])**2)) if mask.sum() > 0 else np.nan


# ======================================================================
# TABLE II: Clean Data Accuracy
# ======================================================================

def table_clean(gt):
    T = gt['T']
    rows = [
        ("1. Inverse Diff. 2pt", rmse(inverse_diff_2pt(gt['omega'], gt['i'], T), gt['u'])),
        ("2. Inverse Diff. 3pt", rmse(inverse_diff_3pt(gt['omega'], gt['i'], T), gt['u'])),
        ("3. Inverse Diff. 4pt", rmse(inverse_diff_4pt(gt['omega'], gt['i'], T), gt['u'])),
        ("4. Inverse Integral",  rmse(inverse_integral(gt['omega'], gt['i'], T), gt['u'])),
    ]
    print("\n" + "="*55)
    print("TABLE II: Clean Data Accuracy (T = 0.1 ms)")
    print("="*55)
    print(f"{'Method':<30s}  {'u RMSE [V]':>14s}  {'Order':>6s}")
    print("-"*55)
    orders = ['O(T)', 'O(T²)', 'O(T³)', 'O(T²)']
    for (name, val), order in zip(rows, orders):
        print(f"{name:<30s}  {val:14.4e}  {order:>6s}")
    print("="*55)
    return rows


# ======================================================================
# TABLE III: Noisy Data Comparison
# ======================================================================

def table_noisy(gt):
    T, n = gt['T'], gt['n']
    noise_levels = [0.01, 0.05, 0.1, 0.5]

    print("\n" + "="*100)
    print("TABLE III: Input Reconstruction RMSE [V] — Noisy Measurements")
    print("="*100)
    print(f"{'σ':>6s}  {'(1) 2pt':>10s}  {'(2) 3pt':>10s}  {'(3) 4pt':>10s}  "
          f"{'(4) Integ.':>10s}  {'(5) EKF+D':>10s}  {'(6) EKF+I':>10s}  {'(7) TII':>10s}")
    print("-"*100)

    results = {}
    for sigma in noise_levels:
        np.random.seed(42)
        wn = gt['omega'] + np.random.normal(0, sigma, n)
        i_n = gt['i'] + np.random.normal(0, sigma, n)

        r = {}
        r['d2'] = rmse(inverse_diff_2pt(wn, i_n, T), gt['u'])
        r['d3'] = rmse(inverse_diff_3pt(wn, i_n, T), gt['u'])
        r['d4'] = rmse(inverse_diff_4pt(wn, i_n, T), gt['u'])
        r['int'] = rmse(inverse_integral(wn, i_n, T), gt['u'])

        # EKF methods need omega_meas only
        np.random.seed(42)
        omega_meas = gt['omega'] + np.random.normal(0, sigma, n)
        try:
            u5, _, _ = ekf_derivative(omega_meas, T, n, sigma)
            r['ekf_d'] = rmse(u5, gt['u'])
        except:
            r['ekf_d'] = np.nan
        try:
            u6, _, _ = ekf_integral(omega_meas, T, n, sigma)
            r['ekf_i'] = rmse(u6, gt['u'])
        except:
            r['ekf_i'] = np.nan

        # TII: grid search for best λ
        best_rmse_tii = np.inf
        best_lam = None
        for lam in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
            u7 = tii(wn, i_n, T, lam)
            r7 = rmse(u7, gt['u'])
            if r7 < best_rmse_tii:
                best_rmse_tii = r7
                best_lam = lam
        r['tii'] = best_rmse_tii
        r['tii_lam'] = best_lam

        # Store arrays for plots
        r['u_d3'] = inverse_diff_3pt(wn, i_n, T)
        r['u_int'] = inverse_integral(wn, i_n, T)
        r['u_tii'] = tii(wn, i_n, T, best_lam)

        results[sigma] = r

        def fmt(v):
            return f"{v:10.4f}" if not np.isnan(v) else f"{'diverg.':>10s}"

        print(f"{sigma:6.2f}  {fmt(r['d2'])}  {fmt(r['d3'])}  {fmt(r['d4'])}  "
              f"{fmt(r['int'])}  {fmt(r['ekf_d'])}  {fmt(r['ekf_i'])}  "
              f"{r['tii']:10.4f}")

    print("="*100)
    return results


# ======================================================================
# TABLE IV: TII Detail
# ======================================================================

def table_tii(gt):
    T, n = gt['T'], gt['n']
    noise_levels = [0.01, 0.05, 0.1, 0.5]
    lambda_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

    print("\n" + "="*70)
    print("TABLE IV: TII — Tikhonov Integral Inversion Detail")
    print("="*70)
    print(f"{'σ':>6s}  {'Unreg. [V]':>12s}  {'Best λ':>8s}  {'TII [V]':>12s}  {'Factor':>8s}")
    print("-"*70)

    results = {}
    for sigma in noise_levels:
        np.random.seed(42)
        wn = gt['omega'] + np.random.normal(0, sigma, n)
        i_n = gt['i'] + np.random.normal(0, sigma, n)

        u_unreg = inverse_integral(wn, i_n, T)
        rmse_unreg = rmse(u_unreg, gt['u'])

        best_r, best_l, best_u = np.inf, None, None
        for lam in lambda_values:
            u7 = tii(wn, i_n, T, lam)
            r7 = rmse(u7, gt['u'])
            if r7 < best_r:
                best_r, best_l, best_u = r7, lam, u7

        factor = rmse_unreg / best_r if best_r > 0 else np.inf
        print(f"{sigma:6.2f}  {rmse_unreg:12.4f}  {best_l:8.0e}  {best_r:12.4f}  {factor:7.0f}x")
        results[sigma] = {'unreg': rmse_unreg, 'lam': best_l, 'tii': best_r,
                          'factor': factor, 'u_unreg': u_unreg, 'u_tii': best_u}

    print("="*70)
    return results


# ======================================================================
# TABLE V: Summary comparison at σ = 0.1
# ======================================================================

def table_summary(noisy_res):
    r = noisy_res[0.1]
    print("\n" + "="*60)
    print("TABLE V: Method Comparison at σ = 0.1")
    print("="*60)
    print(f"{'Method':<35s}  {'u RMSE [V]':>12s}  {'vs TII':>8s}")
    print("-"*60)

    methods = [
        ("(1) Inverse Diff. 2pt", r['d2']),
        ("(2) Inverse Diff. 3pt", r['d3']),
        ("(3) Inverse Diff. 4pt", r['d4']),
        ("(4) Inverse Integral", r['int']),
        ("(5) EKF + Derivative", r['ekf_d']),
        ("(6) EKF + Integral", r['ekf_i']),
        ("(7) TII", r['tii']),
    ]
    tii_val = r['tii']
    for name, val in methods:
        v = f"{val:.4f}" if not np.isnan(val) else "diverges"
        ratio = f"{val/tii_val:.0f}x" if not np.isnan(val) and tii_val > 0 else "—"
        print(f"{name:<35s}  {v:>12s}  {ratio:>8s}")
    print("="*60)


# ======================================================================
# Fig. 3: Publication Figure (6 panels)
# ======================================================================

def figure_publication(gt, noisy_res, tii_res):
    ms = gt['t'] * 1000
    T, n = gt['T'], gt['n']
    sigma = 0.1

    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.28)

    # (a) Clean: inverse methods
    ax = fig.add_subplot(gs[0, 0])
    u_d3 = inverse_diff_3pt(gt['omega'], gt['i'], T)
    u_int = inverse_integral(gt['omega'], gt['i'], T)
    ax.plot(ms, gt['u'], 'k-', lw=2, label='True $u(t)$')
    ax.plot(ms, u_d3, 'b--', lw=1, alpha=0.8,
            label=f'(2) Diff. 3pt ({rmse(u_d3, gt["u"]):.1e} V)')
    ax.plot(ms, u_int, 'r:', lw=1.2, alpha=0.8,
            label=f'(4) Integral ({rmse(u_int, gt["u"]):.1e} V)')
    ax.set_ylabel('$u$ [V]'); ax.set_xlabel('Time [ms]')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_title('(a) Clean Data — Inverse Reconstruction', fontsize=10)
    ax.set_ylim(-1, 14)

    # (b) Noisy: diff vs integral (no regularization)
    ax = fig.add_subplot(gs[0, 1])
    r = noisy_res[sigma]
    ax.plot(ms, gt['u'], 'k-', lw=2, label='True $u(t)$')
    ax.plot(ms, r['u_d3'], 'b-', lw=0.3, alpha=0.4,
            label=f'(2) Diff. 3pt ({r["d3"]:.2f} V)')
    ax.plot(ms, r['u_int'], 'r-', lw=0.5, alpha=0.5,
            label=f'(4) Integral ({r["int"]:.2f} V)')
    ax.set_ylabel('$u$ [V]'); ax.set_xlabel('Time [ms]')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_title(f'(b) Noisy Data ($\\sigma$={sigma}) — No Regularization', fontsize=10)
    ax.set_ylim(-30, 40)

    # (c) TII at σ=0.1
    ax = fig.add_subplot(gs[1, 0])
    rt = tii_res[sigma]
    ax.plot(ms, gt['u'], 'k-', lw=2, label='True $u(t)$')
    ax.plot(ms[1:], rt['u_unreg'][1:], color='gray', lw=0.3, alpha=0.3,
            label=f'Unreg. ({rt["unreg"]:.2f} V)')
    ax.plot(ms[1:], rt['u_tii'][1:], 'r-', lw=1.5,
            label=f'(7) TII $\\lambda$={rt["lam"]:.0e} ({rt["tii"]:.4f} V)')
    ax.set_ylabel('$u$ [V]'); ax.set_xlabel('Time [ms]')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_title(f'(c) TII — Tikhonov Integral Inversion ($\\sigma$={sigma})', fontsize=10)
    ax.set_ylim(-2, 16)

    # (d) TII across noise levels — bar chart
    ax = fig.add_subplot(gs[1, 1])
    sigmas = sorted(tii_res.keys())
    unreg = [tii_res[s]['unreg'] for s in sigmas]
    tii_v = [tii_res[s]['tii'] for s in sigmas]
    x = np.arange(len(sigmas))
    w = 0.35
    ax.bar(x - w/2, unreg, w, label='(4) Unregularized', color='#d62728', alpha=0.7)
    ax.bar(x + w/2, tii_v, w, label='(7) TII', color='#2ca02c', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s}' for s in sigmas])
    ax.set_xlabel('Noise $\\sigma$'); ax.set_ylabel('$u$ RMSE [V]')
    ax.set_yscale('log'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')
    ax.set_title('(d) TII Improvement Across Noise Levels', fontsize=10)
    for i, (u, t) in enumerate(zip(unreg, tii_v)):
        ax.text(i + w/2, t * 1.4, f'{u/t:.0f}x', ha='center', fontsize=8,
                fontweight='bold', color='#2ca02c')

    # (e) EKF comparison at σ=0.1
    ax = fig.add_subplot(gs[2, 0])
    np.random.seed(42)
    omega_meas = gt['omega'] + np.random.normal(0, sigma, n)
    u5, _, _ = ekf_derivative(omega_meas, T, n, sigma)
    u6, _, _ = ekf_integral(omega_meas, T, n, sigma)
    ax.plot(ms, gt['u'], 'k-', lw=2, label='True $u(t)$')
    ax.plot(ms, u5, 'b-', lw=0.5, alpha=0.5,
            label=f'(5) EKF+Deriv ({rmse(u5, gt["u"]):.3f} V)')
    ax.plot(ms, u6, 'r-', lw=0.8, alpha=0.7,
            label=f'(6) EKF+Integ ({rmse(u6, gt["u"]):.3f} V)')
    ax.set_ylabel('$u$ [V]'); ax.set_xlabel('Time [ms]')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_title(f'(e) EKF-Based Reconstruction ($\\sigma$={sigma})', fontsize=10)
    ax.set_ylim(-5, 20)

    # (f) RMSE vs σ — all methods
    ax = fig.add_subplot(gs[2, 1])
    sigmas_n = sorted(noisy_res.keys())
    ax.loglog(sigmas_n, [noisy_res[s]['d2'] for s in sigmas_n], 'bs--', ms=6, label='(1) 2pt')
    ax.loglog(sigmas_n, [noisy_res[s]['d3'] for s in sigmas_n], 'g^--', ms=6, label='(2) 3pt')
    ax.loglog(sigmas_n, [noisy_res[s]['d4'] for s in sigmas_n], 'mv--', ms=6, label='(3) 4pt')
    ax.loglog(sigmas_n, [noisy_res[s]['int'] for s in sigmas_n], 'ro--', ms=6, label='(4) Integral')
    ekf_d = [noisy_res[s]['ekf_d'] for s in sigmas_n]
    ekf_i = [noisy_res[s]['ekf_i'] for s in sigmas_n]
    valid_d = [(s, v) for s, v in zip(sigmas_n, ekf_d) if not np.isnan(v)]
    valid_i = [(s, v) for s, v in zip(sigmas_n, ekf_i) if not np.isnan(v)]
    if valid_d:
        ax.loglog(*zip(*valid_d), 'cD--', ms=6, label='(5) EKF+D')
    if valid_i:
        ax.loglog(*zip(*valid_i), 'mP--', ms=6, label='(6) EKF+I')
    ax.loglog(sigmas_n, [tii_res[s]['tii'] for s in sigmas_n], 'kH-', ms=10, lw=2.5,
              label='(7) TII')
    ax.set_xlabel('Noise $\\sigma$'); ax.set_ylabel('$u$ RMSE [V]')
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3, which='both')
    ax.set_title('(f) RMSE Scaling with Noise Level', fontsize=10)

    plt.suptitle('Regularized Integral Inversion for Unknown Input Reconstruction\n'
                 'DC Motor with Nonlinear Magnetic Saturation $L(i)$',
                 fontsize=13, fontweight='bold')
    plt.savefig('/home/rodo/10Paper/results/fig_publication.png', dpi=300, bbox_inches='tight')
    print("Saved: results/fig_publication.png")
    plt.close()


# ======================================================================
# Fig. 4: RMSE vs σ (standalone)
# ======================================================================

def figure_noise_scaling(noisy_res, tii_res):
    sigmas = sorted(noisy_res.keys())

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.loglog(sigmas, [noisy_res[s]['d2'] for s in sigmas], 'bs--', ms=8, lw=1.5, label='(1) Inv. Diff. 2pt')
    ax.loglog(sigmas, [noisy_res[s]['d3'] for s in sigmas], 'g^--', ms=8, lw=1.5, label='(2) Inv. Diff. 3pt')
    ax.loglog(sigmas, [noisy_res[s]['d4'] for s in sigmas], 'mv--', ms=8, lw=1.5, label='(3) Inv. Diff. 4pt')
    ax.loglog(sigmas, [noisy_res[s]['int'] for s in sigmas], 'ro--', ms=8, lw=1.5, label='(4) Inv. Integral')
    ax.loglog(sigmas, [tii_res[s]['tii'] for s in sigmas], 'kD-', ms=10, lw=2.5, label='(7) TII')

    ax.set_xlabel('Noise level $\\sigma$ [rad/s, A]', fontsize=12)
    ax.set_ylabel('$u$ RMSE [V]', fontsize=12)
    ax.set_title('Input Reconstruction Error vs Measurement Noise', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig('/home/rodo/10Paper/results/fig_noise_scaling.png', dpi=300, bbox_inches='tight')
    print("Saved: results/fig_noise_scaling.png")
    plt.close()


# ======================================================================
# Main
# ======================================================================

if __name__ == '__main__':
    np.random.seed(42)

    print("Generating RK4 ground truth (dt=1e-6)...")
    gt = generate_ground_truth()

    res_clean = table_clean(gt)
    res_noisy = table_noisy(gt)
    res_tii = table_tii(gt)
    table_summary(res_noisy)

    print("\nGenerating figures...")
    figure_publication(gt, res_noisy, res_tii)
    figure_noise_scaling(res_noisy, res_tii)

    print("\nAll tables and figures generated.")
