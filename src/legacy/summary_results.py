"""
Summary of All Experiments — Regularized Integral Inversion (TII)
=================================================================

Generates publication-quality tables and figures with English titles.

Author: Rodolfo H. Rodrigo
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import motor_model as mm
from test_integral import (direct_differential, direct_integral,
                           inverse_differential, inverse_integral, Phi)
from tikhonov_inv_integral import tikhonov_inverse_integral


# ======================================================================
# Common setup
# ======================================================================

def generate_ground_truth(T=0.0001, t_final=0.2):
    """Generate RK4 ground truth."""
    n = int(t_final / T)
    u_func = lambda t: 12.0 if t > 0.005 else 0.0
    x0 = np.array([0.0, 0.0])
    t_rk4, states_rk4, inputs_rk4 = mm.simulate(
        (0, t_final), x0, u_func, dt_rk4=1e-6
    )
    step = int(T / 1e-6)
    idx = np.arange(0, len(t_rk4), step)[:n]
    return {
        't': t_rk4[idx], 'omega': states_rk4[idx, 0],
        'i': states_rk4[idx, 1], 'u': inputs_rk4[idx],
        'T': T, 'n': n
    }


def rmse(a, b, skip=100):
    """RMSE skipping first `skip` samples."""
    s = slice(skip, None)
    mask = ~np.isnan(a[s]) & ~np.isnan(b[s])
    if mask.sum() == 0:
        return np.nan
    return np.sqrt(np.mean((a[s][mask] - b[s][mask])**2))


# ======================================================================
# Experiment 1: Clean data — 4 formulations vs RK4
# ======================================================================

def experiment_1(gt):
    """Table I: Four Formulations Accuracy (No Noise)."""
    T, n = gt['T'], gt['n']

    omega_dd, i_dd = direct_differential(
        gt['u'], gt['omega'][0], gt['omega'][1],
        gt['i'][0], gt['i'][1], T, n)
    omega_di, i_di = direct_integral(
        gt['u'], gt['omega'][0], None,
        gt['i'][0], None, T, n)
    u_d2 = inverse_differential(gt['omega'], gt['i'], T, n_points=2)
    u_d3 = inverse_differential(gt['omega'], gt['i'], T, n_points=3)
    u_d4 = inverse_differential(gt['omega'], gt['i'], T, n_points=4)
    u_int = inverse_integral(gt['omega'], gt['i'], T)

    results = {
        'Direct Differential': {
            'omega_rmse': rmse(omega_dd, gt['omega']),
            'i_rmse': rmse(i_dd, gt['i']),
            'u_rmse': None},
        'Direct Integral': {
            'omega_rmse': rmse(omega_di, gt['omega']),
            'i_rmse': rmse(i_di, gt['i']),
            'u_rmse': None},
        'Inverse Diff. 2pt': {
            'omega_rmse': None, 'i_rmse': None,
            'u_rmse': rmse(u_d2, gt['u'])},
        'Inverse Diff. 3pt': {
            'omega_rmse': None, 'i_rmse': None,
            'u_rmse': rmse(u_d3, gt['u'])},
        'Inverse Diff. 4pt': {
            'omega_rmse': None, 'i_rmse': None,
            'u_rmse': rmse(u_d4, gt['u'])},
        'Inverse Integral': {
            'omega_rmse': None, 'i_rmse': None,
            'u_rmse': rmse(u_int, gt['u'])},
    }
    return results


# ======================================================================
# Experiment 2: Noise robustness — Inverse formulations
# ======================================================================

def experiment_2(gt):
    """Table II: Inverse Reconstruction with Noisy Measurements."""
    T, n = gt['T'], gt['n']
    noise_levels = [0.01, 0.05, 0.1, 0.5]
    results = {}

    for sigma in noise_levels:
        np.random.seed(42)
        omega_n = gt['omega'] + np.random.normal(0, sigma, n)
        i_n = gt['i'] + np.random.normal(0, sigma, n)

        u_d2 = inverse_differential(omega_n, i_n, T, n_points=2)
        u_d3 = inverse_differential(omega_n, i_n, T, n_points=3)
        u_int = inverse_integral(omega_n, i_n, T)

        results[sigma] = {
            'rmse_d2': rmse(u_d2, gt['u']),
            'rmse_d3': rmse(u_d3, gt['u']),
            'rmse_int': rmse(u_int, gt['u']),
            'u_d2': u_d2, 'u_d3': u_d3, 'u_int': u_int,
        }
    return results


# ======================================================================
# Experiment 3: Tikhonov Regularized Integral Inversion (TII)
# ======================================================================

def experiment_3(gt):
    """Table III: TII — Tikhonov Integral Inversion."""
    T, n = gt['T'], gt['n']
    noise_levels = [0.01, 0.05, 0.1, 0.5]
    lambda_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    results = {}

    for sigma in noise_levels:
        np.random.seed(42)
        omega_n = gt['omega'] + np.random.normal(0, sigma, n)
        i_n = gt['i'] + np.random.normal(0, sigma, n)

        # Unregularized
        u_unreg = inverse_integral(omega_n, i_n, T)
        rmse_unreg = rmse(u_unreg, gt['u'])

        best_rmse = np.inf
        best_lam = None
        best_u = None

        for lam in lambda_values:
            u_tik = tikhonov_inverse_integral(omega_n, i_n, T, lam)
            r = rmse(u_tik, gt['u'])
            if r < best_rmse:
                best_rmse = r
                best_lam = lam
                best_u = u_tik

        results[sigma] = {
            'rmse_unreg': rmse_unreg,
            'best_lambda': best_lam,
            'rmse_tikh': best_rmse,
            'improvement': rmse_unreg / best_rmse if best_rmse > 0 else np.inf,
            'u_unreg': u_unreg, 'u_tikh': best_u,
        }
    return results


# ======================================================================
# Experiment 4: Direct regressor with noisy input
# ======================================================================

def experiment_4(gt):
    """Table IV: Direct Regressor Robustness (Noisy Input u)."""
    T, n = gt['T'], gt['n']
    noise_levels = [0.1, 0.5, 1.0, 2.0]
    results = {}

    for sigma in noise_levels:
        np.random.seed(42)
        u_noisy = gt['u'] + np.random.normal(0, sigma, n)

        omega_dd, i_dd = direct_differential(
            u_noisy, gt['omega'][0], gt['omega'][1],
            gt['i'][0], gt['i'][1], T, n)

        results[sigma] = {
            'omega_rmse': rmse(omega_dd, gt['omega']),
            'i_rmse': rmse(i_dd, gt['i']),
        }
    return results


# ======================================================================
# Experiment 5: EKF + integral vs derivative reconstruction
# ======================================================================

def experiment_5(gt):
    """Table V: EKF State Filtering + Input Reconstruction."""
    from ekf import EKF

    T, n = gt['T'], gt['n']
    noise_levels = [0.01, 0.05, 0.1, 0.5]
    results = {}

    for sigma in noise_levels:
        np.random.seed(42)
        omega_meas = gt['omega'] + np.random.normal(0, sigma, n)

        row = {}
        for method, use_int in [('derivative', False), ('integral', True)]:
            Q = np.diag([1e-2, 1e-2])
            ekf = EKF(T=T, Q=Q, R_meas=sigma**2,
                       x0=np.array([0.0, 0.0]),
                       P0=np.eye(2) * 1.0, use_integral=use_int)

            omega_f = np.zeros(n)
            i_f = np.zeros(n)
            u_hat = np.zeros(n)
            u_est = 0.0

            for k in range(1, n):
                x_f, _, u_r = ekf.step(omega_meas[k], u_est=u_est)
                omega_f[k] = x_f[0]
                i_f[k] = x_f[1]
                u_hat[k] = u_r
                u_est = u_r

            row[method] = {
                'omega_rmse': rmse(omega_f, gt['omega']),
                'i_rmse': rmse(i_f, gt['i']),
                'u_rmse': rmse(u_hat, gt['u']),
                'omega_f': omega_f, 'i_f': i_f, 'u_hat': u_hat,
            }

        row['omega_meas'] = omega_meas
        results[sigma] = row
    return results


# ======================================================================
# Print Tables
# ======================================================================

def print_table_1(res):
    print("\n" + "="*75)
    print("TABLE I: Four Formulations Accuracy — Clean Data (T=0.1ms)")
    print("="*75)
    print(f"{'Formulation':<25s}  {'ω RMSE [rad/s]':>16s}  {'i RMSE [A]':>12s}  {'u RMSE [V]':>12s}")
    print("-"*75)
    for name, r in res.items():
        w = f"{r['omega_rmse']:.4e}" if r['omega_rmse'] is not None else "—"
        i = f"{r['i_rmse']:.4e}" if r['i_rmse'] is not None else "—"
        u = f"{r['u_rmse']:.4e}" if r['u_rmse'] is not None else "—"
        print(f"{name:<25s}  {w:>16s}  {i:>12s}  {u:>12s}")
    print("="*75)


def print_table_2(res):
    print("\n" + "="*75)
    print("TABLE II: Inverse Input Reconstruction — Noisy ω, i (T=0.1ms)")
    print("="*75)
    print(f"{'σ':>6s}  {'Diff. 2pt [V]':>14s}  {'Diff. 3pt [V]':>14s}  {'Integral [V]':>14s}  {'3pt/Int':>8s}")
    print("-"*75)
    for sigma, r in res.items():
        ratio = r['rmse_d3'] / r['rmse_int'] if r['rmse_int'] > 0 else np.inf
        print(f"{sigma:6.2f}  {r['rmse_d2']:14.4f}  {r['rmse_d3']:14.4f}  "
              f"{r['rmse_int']:14.4f}  {ratio:8.1f}x")
    print("="*75)


def print_table_3(res):
    print("\n" + "="*75)
    print("TABLE III: Tikhonov Integral Inversion (TII)")
    print("="*75)
    print(f"{'σ':>6s}  {'Unreg. [V]':>12s}  {'Best λ':>8s}  {'TII [V]':>12s}  {'Improvement':>12s}")
    print("-"*75)
    for sigma, r in res.items():
        print(f"{sigma:6.2f}  {r['rmse_unreg']:12.4f}  {r['best_lambda']:8.0e}  "
              f"{r['rmse_tikh']:12.4f}  {r['improvement']:11.1f}x")
    print("="*75)


def print_table_4(res):
    print("\n" + "="*75)
    print("TABLE IV: Direct Regressor Robustness — Noisy Input u")
    print("="*75)
    print(f"{'σ_u [V]':>8s}  {'ω RMSE [rad/s]':>16s}  {'i RMSE [A]':>12s}")
    print("-"*45)
    for sigma, r in res.items():
        print(f"{sigma:8.1f}  {r['omega_rmse']:16.4e}  {r['i_rmse']:12.4e}")
    print("="*45)


def print_table_5(res):
    print("\n" + "="*80)
    print("TABLE V: EKF + Input Reconstruction — Derivative vs Integral")
    print("="*80)
    print(f"{'σ_ω':>6s}  {'u RMSE deriv':>14s}  {'u RMSE integ':>14s}  "
          f"{'ω RMSE':>10s}  {'i RMSE':>10s}  {'Improv.':>8s}")
    print("-"*80)
    for sigma, r in res.items():
        ud = r['derivative']['u_rmse']
        ui = r['integral']['u_rmse']
        wr = r['integral']['omega_rmse']
        ir = r['integral']['i_rmse']
        imp = ud / ui if ui > 0 and not np.isnan(ud) else np.nan
        ud_s = f"{ud:14.4f}" if not np.isnan(ud) else f"{'diverges':>14s}"
        ui_s = f"{ui:14.4f}" if not np.isnan(ui) else f"{'diverges':>14s}"
        imp_s = f"{imp:7.1f}x" if not np.isnan(imp) else "—"
        print(f"{sigma:6.2f}  {ud_s}  {ui_s}  {wr:10.4f}  {ir:10.4f}  {imp_s:>8s}")
    print("="*80)


def print_comparison_table(res2, res3, res5):
    print("\n" + "="*80)
    print("TABLE VI: Method Comparison — u Reconstruction RMSE [V] at σ=0.1")
    print("="*80)
    methods = [
        ("Inverse Differential (2pt)", res2[0.1]['rmse_d2']),
        ("Inverse Differential (3pt)", res2[0.1]['rmse_d3']),
        ("Inverse Integral (unreg.)", res2[0.1]['rmse_int']),
        ("EKF + Derivative", res5[0.1]['derivative']['u_rmse']),
        ("EKF + Integral", res5[0.1]['integral']['u_rmse']),
        ("TII (Tikhonov + Integral)", res3[0.1]['rmse_tikh']),
    ]
    print(f"{'Method':<35s}  {'u RMSE [V]':>12s}  {'vs TII':>8s}")
    print("-"*60)
    tii = res3[0.1]['rmse_tikh']
    for name, val in methods:
        v = f"{val:.4f}" if not np.isnan(val) else "diverges"
        ratio = f"{val/tii:.0f}x" if not np.isnan(val) and tii > 0 else "—"
        print(f"{name:<35s}  {v:>12s}  {ratio:>8s}")
    print("="*60)


# ======================================================================
# Publication Figure
# ======================================================================

def make_publication_figure(gt, res2, res3, res5):
    """Fig. 1: Combined publication figure."""
    T, n = gt['T'], gt['n']
    ms = gt['t'] * 1000

    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

    sigma = 0.1

    # --- (a) Clean: 4 formulations ---
    ax = fig.add_subplot(gs[0, 0])
    omega_dd, i_dd = direct_differential(
        gt['u'], gt['omega'][0], gt['omega'][1],
        gt['i'][0], gt['i'][1], T, n)
    omega_di, i_di = direct_integral(
        gt['u'], gt['omega'][0], None, gt['i'][0], None, T, n)
    ax.plot(ms, gt['omega'], 'k-', lw=2, label='RK4 reference')
    ax.plot(ms, omega_dd, 'b--', lw=1, alpha=0.8, label='Direct differential')
    ax.plot(ms, omega_di, 'r:', lw=1.2, alpha=0.8, label='Direct integral')
    ax.set_ylabel('$\\omega$ [rad/s]')
    ax.set_xlabel('Time [ms]')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title('(a) Direct Regressors vs RK4 — Clean Data', fontsize=10)

    # --- (b) Clean: Inverse u ---
    ax = fig.add_subplot(gs[0, 1])
    u_d3 = inverse_differential(gt['omega'], gt['i'], T, n_points=3)
    u_int = inverse_integral(gt['omega'], gt['i'], T)
    ax.plot(ms, gt['u'], 'k-', lw=2, label='True $u(t)$')
    ax.plot(ms, u_d3, 'b--', lw=1, alpha=0.8,
            label=f'Diff. 3pt (RMSE={rmse(u_d3, gt["u"]):.1e})')
    ax.plot(ms, u_int, 'r:', lw=1.2, alpha=0.8,
            label=f'Integral (RMSE={rmse(u_int, gt["u"]):.1e})')
    ax.set_ylabel('$u$ [V]')
    ax.set_xlabel('Time [ms]')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title('(b) Inverse Reconstruction — Clean Data', fontsize=10)
    ax.set_ylim(-1, 14)

    # --- (c) Noisy inverse: diff vs integral ---
    ax = fig.add_subplot(gs[1, 0])
    r2 = res2[sigma]
    ax.plot(ms, gt['u'], 'k-', lw=2, label='True $u(t)$')
    ax.plot(ms, r2['u_d3'], 'b-', lw=0.3, alpha=0.4,
            label=f'Diff. 3pt (RMSE={r2["rmse_d3"]:.2f})')
    ax.plot(ms, r2['u_int'], 'r-', lw=0.5, alpha=0.5,
            label=f'Integral (RMSE={r2["rmse_int"]:.2f})')
    ax.set_ylabel('$u$ [V]')
    ax.set_xlabel('Time [ms]')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title(f'(c) Inverse with Noise ($\\sigma$={sigma}) — No Regularization',
                 fontsize=10)
    ax.set_ylim(-30, 40)

    # --- (d) TII: Tikhonov + Integral ---
    ax = fig.add_subplot(gs[1, 1])
    r3 = res3[sigma]
    ax.plot(ms, gt['u'], 'k-', lw=2, label='True $u(t)$')
    ax.plot(ms[1:], r3['u_unreg'][1:], color='gray', lw=0.3, alpha=0.3,
            label=f'Unreg. (RMSE={r3["rmse_unreg"]:.2f})')
    ax.plot(ms[1:], r3['u_tikh'][1:], 'r-', lw=1.5,
            label=f'TII $\\lambda$={r3["best_lambda"]:.0e} (RMSE={r3["rmse_tikh"]:.4f})')
    ax.set_ylabel('$u$ [V]')
    ax.set_xlabel('Time [ms]')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title(f'(d) Tikhonov Integral Inversion ($\\sigma$={sigma})', fontsize=10)
    ax.set_ylim(-2, 16)

    # --- (e) TII across noise levels ---
    ax = fig.add_subplot(gs[2, 0])
    sigmas = sorted(res3.keys())
    unreg = [res3[s]['rmse_unreg'] for s in sigmas]
    tikh = [res3[s]['rmse_tikh'] for s in sigmas]
    x = np.arange(len(sigmas))
    w = 0.35
    bars1 = ax.bar(x - w/2, unreg, w, label='Unregularized', color='#d62728', alpha=0.7)
    bars2 = ax.bar(x + w/2, tikh, w, label='TII', color='#2ca02c', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s}' for s in sigmas])
    ax.set_xlabel('Noise level $\\sigma$')
    ax.set_ylabel('$u$ RMSE [V]')
    ax.set_yscale('log')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_title('(e) TII Improvement Across Noise Levels', fontsize=10)
    # Add improvement labels
    for i, (u, t) in enumerate(zip(unreg, tikh)):
        if t > 0:
            ax.text(i + w/2, t * 1.3, f'{u/t:.0f}x', ha='center', fontsize=8,
                    fontweight='bold', color='#2ca02c')

    # --- (f) EKF comparison ---
    ax = fig.add_subplot(gs[2, 1])
    if 0.1 in res5:
        r5 = res5[0.1]
        ax.plot(ms, gt['u'], 'k-', lw=2, label='True $u(t)$')
        ud = r5['derivative']['u_rmse']
        ui = r5['integral']['u_rmse']
        if not np.isnan(ud):
            ax.plot(ms, r5['derivative']['u_hat'], 'b-', lw=0.5, alpha=0.5,
                    label=f'EKF+Deriv (RMSE={ud:.3f})')
        ax.plot(ms, r5['integral']['u_hat'], 'r-', lw=0.8, alpha=0.7,
                label=f'EKF+Integ (RMSE={ui:.3f})')
        ax.set_ylabel('$u$ [V]')
        ax.set_xlabel('Time [ms]')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title(f'(f) EKF + Input Reconstruction ($\\sigma$={sigma})', fontsize=10)
        ax.set_ylim(-5, 20)

    plt.suptitle('Regularized Integral Inversion for Unknown Input Reconstruction\n'
                 'DC Motor with Nonlinear Magnetic Saturation',
                 fontsize=14, fontweight='bold')
    plt.savefig('/home/rodo/10Paper/results/summary_publication.png',
                dpi=300, bbox_inches='tight')
    print("\nSaved: results/summary_publication.png")
    plt.close()


# ======================================================================
# Noise scaling figure
# ======================================================================

def make_noise_scaling_figure(res2, res3):
    """Fig. 2: RMSE vs noise level for all methods."""
    sigmas = sorted(res2.keys())

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    d2 = [res2[s]['rmse_d2'] for s in sigmas]
    d3 = [res2[s]['rmse_d3'] for s in sigmas]
    integ = [res2[s]['rmse_int'] for s in sigmas]
    tii = [res3[s]['rmse_tikh'] for s in sigmas]

    ax.loglog(sigmas, d2, 'bs--', lw=1.5, ms=8, label='Inverse Diff. 2pt')
    ax.loglog(sigmas, d3, 'g^--', lw=1.5, ms=8, label='Inverse Diff. 3pt')
    ax.loglog(sigmas, integ, 'ro--', lw=1.5, ms=8, label='Inverse Integral')
    ax.loglog(sigmas, tii, 'kD-', lw=2.5, ms=10, label='TII (Tikhonov + Integral)')

    ax.set_xlabel('Noise level $\\sigma$ [rad/s, A]', fontsize=12)
    ax.set_ylabel('$u$ RMSE [V]', fontsize=12)
    ax.set_title('Input Reconstruction Error vs Measurement Noise', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('/home/rodo/10Paper/results/noise_scaling.png',
                dpi=300, bbox_inches='tight')
    print("Saved: results/noise_scaling.png")
    plt.close()


# ======================================================================
# Main
# ======================================================================

if __name__ == '__main__':
    np.random.seed(42)

    print("Generating ground truth...")
    gt = generate_ground_truth(T=0.0001, t_final=0.2)

    print("Running Experiment 1: Clean data...")
    res1 = experiment_1(gt)
    print_table_1(res1)

    print("Running Experiment 2: Noisy inverse...")
    res2 = experiment_2(gt)
    print_table_2(res2)

    print("Running Experiment 3: TII...")
    res3 = experiment_3(gt)
    print_table_3(res3)

    print("Running Experiment 4: Direct with noisy u...")
    res4 = experiment_4(gt)
    print_table_4(res4)

    print("Running Experiment 5: EKF...")
    res5 = experiment_5(gt)
    print_table_5(res5)

    print_comparison_table(res2, res3, res5)

    print("\nGenerating figures...")
    make_publication_figure(gt, res2, res3, res5)
    make_noise_scaling_figure(res2, res3)

    print("\nDone.")
