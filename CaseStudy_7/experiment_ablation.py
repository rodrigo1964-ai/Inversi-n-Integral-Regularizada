import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
import motor_model as mm
from methods import inverse_diff_3pt, inverse_integral, ekf_integral, tii

def rmse(a, b, skip=100):
    s = slice(skip, None)
    mask = ~np.isnan(a[s]) & ~np.isnan(b[s])
    return np.sqrt(np.mean((a[s][mask] - b[s][mask])**2)) if mask.sum() > 0 else np.nan

def mae(a, b, skip=100):
    s = slice(skip, None)
    mask = ~np.isnan(a[s]) & ~np.isnan(b[s])
    return np.mean(np.abs(a[s][mask] - b[s][mask])) if mask.sum() > 0 else np.nan

def corr(a, b, skip=100):
    s = slice(skip, None)
    mask = ~np.isnan(a[s]) & ~np.isnan(b[s])
    if mask.sum() < 2:
        return np.nan
    return np.corrcoef(a[s][mask], b[s][mask])[0, 1]

def generate_ground_truth(T=0.0001, t_final=0.2):
    n = int(t_final / T)
    u_func = lambda t: 6.0 + 6.0 * np.sin(2 * np.pi * 25 * t)
    x0 = np.array([0.0, 0.0])
    t_rk4, states_rk4, inputs_rk4 = mm.simulate((0, t_final), x0, u_func, dt_rk4=1e-6)
    step = int(T / 1e-6)
    idx = np.arange(0, len(t_rk4), step)[:n]
    return {
        't': t_rk4[idx], 'omega': states_rk4[idx, 0],
        'i': states_rk4[idx, 1], 'u': inputs_rk4[idx],
        'T': T, 'n': n
    }

def run_ablation_experiment(gt):
    sigma = 0.1
    np.random.seed(42)
    wn = gt['omega'] + np.random.normal(0, sigma, gt['n'])
    i_n = gt['i'] + np.random.normal(0, sigma, gt['n'])
    T = gt['T']
    n = gt['n']
    u_true = gt['u']

    # Config 1: Differential only
    u_diff = inverse_diff_3pt(wn, i_n, T)
    # Config 2: Integral only
    u_int = inverse_integral(wn, i_n, T)
    # Config 3: Integral + EKF
    u_ekf, _, _ = ekf_integral(wn, T, n, sigma)
    # Config 4: TII (full)
    u_tii = tii(wn, i_n, T, lam=50)

    configs = [
        ('Differential only', u_diff),
        ('Integral only', u_int),
        ('Integral + EKF', u_ekf),
        ('TII (full)', u_tii),
    ]

    results = []
    for name, u_est in configs:
        r = rmse(u_est, u_true)
        m = mae(u_est, u_true)
        c = corr(u_est, u_true)
        nan_frac = np.sum(np.isnan(u_est)) / len(u_est)
        stable = 'Yes' if nan_frac < 0.5 else 'No'
        results.append({
            'name': name, 'rmse': r, 'mae': m, 'corr': c,
            'stable': stable, 'u_est': u_est
        })

    return results

def print_table(results):
    print(f"{'Configuration':<20} {'RMSE_V':>8} {'MAE_V':>8} {'Corr':>8} {'Stable':>6}")
    print('-' * 54)
    for r in results:
        print(f"{r['name']:<20} {r['rmse']:8.3f} {r['mae']:8.3f} {r['corr']:8.4f} {r['stable']:>6}")

if __name__ == '__main__':
    gt = generate_ground_truth()
    results = run_ablation_experiment(gt)
    print_table(results)
