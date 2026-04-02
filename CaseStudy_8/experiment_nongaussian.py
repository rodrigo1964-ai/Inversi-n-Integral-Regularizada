import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
import motor_model as mm
from methods import inverse_diff_3pt, ekf_derivative, ekf_integral, tii

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

def generate_noise(noise_type, sigma, n, rng):
    if noise_type == 'Gaussian':
        return rng.normal(0, sigma, n)
    elif noise_type == 'Laplacian':
        return rng.laplace(0, sigma / np.sqrt(2), n)
    elif noise_type == 'Impulsive':
        base = rng.normal(0, sigma, n)
        mask = rng.random(n) < 0.05
        base[mask] = rng.uniform(-5*sigma, 5*sigma, mask.sum())
        return base
    raise ValueError(f"Unknown noise type: {noise_type}")

def run_nongaussian_experiment(gt):
    sigma = 0.1
    T = gt['T']
    n = gt['n']
    u_true = gt['u']
    noise_types = ['Gaussian', 'Laplacian', 'Impulsive']
    method_names = ['Diff 3pt', 'EKF + Derivative', 'EKF + Integral', 'TII']

    results = {m: {} for m in method_names}

    for noise_type in noise_types:
        np.random.seed(42)
        rng = np.random.RandomState(42)

        # Noise for omega and i (for standalone methods)
        noise_w = generate_noise(noise_type, sigma, n, rng)
        noise_i = generate_noise(noise_type, sigma, n, rng)
        wn = gt['omega'] + noise_w
        i_n = gt['i'] + noise_i

        # For EKF methods: noise on omega only
        np.random.seed(42)
        rng_ekf = np.random.RandomState(42)
        noise_w_ekf = generate_noise(noise_type, sigma, n, rng_ekf)
        wn_ekf = gt['omega'] + noise_w_ekf

        # Diff 3pt
        u_diff = inverse_diff_3pt(wn, i_n, T)
        results['Diff 3pt'][noise_type] = rmse(u_diff, u_true)

        # EKF + Derivative
        u_ekfd, _, _ = ekf_derivative(wn_ekf, T, n, sigma)
        results['EKF + Derivative'][noise_type] = rmse(u_ekfd, u_true)

        # EKF + Integral
        u_ekfi, _, _ = ekf_integral(wn_ekf, T, n, sigma)
        results['EKF + Integral'][noise_type] = rmse(u_ekfi, u_true)

        # TII
        u_tii = tii(wn, i_n, T, lam=100)
        results['TII'][noise_type] = rmse(u_tii, u_true)

    return results

def print_table(results):
    noise_types = ['Gaussian', 'Laplacian', 'Impulsive']
    print(f"{'Method':<20} {'Gaussian':>10} {'Laplacian':>10} {'Impulsive':>10}")
    print('-' * 52)
    for method in ['Diff 3pt', 'EKF + Derivative', 'EKF + Integral', 'TII']:
        vals = [f"{results[method][nt]:.3f}" for nt in noise_types]
        print(f"{method:<20} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10}")

if __name__ == '__main__':
    gt = generate_ground_truth()
    results = run_nongaussian_experiment(gt)
    print_table(results)
