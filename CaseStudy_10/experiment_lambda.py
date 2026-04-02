import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
import motor_model as mm
from methods import tii

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

def run_lambda_experiment(gt):
    lambdas = np.logspace(-3, 5, 50)
    sigmas = [0.01, 0.05, 0.1, 0.5]
    T = gt['T']
    n = gt['n']
    u_true = gt['u']

    results = {sigma: np.zeros(len(lambdas)) for sigma in sigmas}

    for sigma in sigmas:
        np.random.seed(42)
        wn = gt['omega'] + np.random.normal(0, sigma, n)
        i_n = gt['i'] + np.random.normal(0, sigma, n)

        for j, lam in enumerate(lambdas):
            u_tii = tii(wn, i_n, T, lam=lam)
            results[sigma][j] = rmse(u_tii, u_true)

    return {'lambdas': lambdas, 'sigmas': sigmas, 'rmse': results}

def print_table(results):
    lambdas = results['lambdas']
    sigmas = results['sigmas']
    header = f"{'lambda':>12}" + "".join(f"{'s='+str(s):>12}" for s in sigmas)
    print(header)
    print('-' * (12 + 12*len(sigmas)))
    for j, lam in enumerate(lambdas):
        vals = "".join(f"{results['rmse'][s][j]:12.4f}" for s in sigmas)
        print(f"{lam:12.4f}{vals}")

if __name__ == '__main__':
    gt = generate_ground_truth()
    results = run_lambda_experiment(gt)
    print_table(results)
