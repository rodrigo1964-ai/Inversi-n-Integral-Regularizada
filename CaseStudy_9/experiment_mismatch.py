import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
import motor_model as mm
from methods import ekf_derivative, ekf_integral, tii

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

def run_mismatch_experiment():
    sigma = 0.1
    deltas = [0.0, 0.1, 0.2, 0.3]
    method_names = ['EKF + Derivative', 'EKF + Integral', 'TII']
    results = {m: {} for m in method_names}

    original_L0 = mm.L0
    original_C = mm.C_LOAD

    for delta in deltas:
        # Generate ground truth with perturbed parameters
        mm.L0 = original_L0 * (1 + delta)
        mm.C_LOAD = original_C * (1 + delta)
        gt = generate_ground_truth()
        mm.L0 = original_L0
        mm.C_LOAD = original_C

        u_true = gt['u']
        T = gt['T']
        n = gt['n']

        np.random.seed(42)
        wn = gt['omega'] + np.random.normal(0, sigma, n)
        i_n = gt['i'] + np.random.normal(0, sigma, n)

        # EKF methods use nominal params internally
        u_ekfd, _, _ = ekf_derivative(wn, T, n, sigma)
        results['EKF + Derivative'][delta] = rmse(u_ekfd, u_true)

        u_ekfi, _, _ = ekf_integral(wn, T, n, sigma)
        results['EKF + Integral'][delta] = rmse(u_ekfi, u_true)

        u_tii = tii(wn, i_n, T, lam=100)
        results['TII'][delta] = rmse(u_tii, u_true)

    return results

def print_table(results):
    deltas = [0.0, 0.1, 0.2, 0.3]
    header = f"{'Method':<20}" + "".join(f"{'d='+str(d):>10}" for d in deltas)
    print(header)
    print('-' * (20 + 10*len(deltas)))
    for method in ['EKF + Derivative', 'EKF + Integral', 'TII']:
        vals = "".join(f"{results[method][d]:10.3f}" for d in deltas)
        print(f"{method:<20}{vals}")

if __name__ == '__main__':
    results = run_mismatch_experiment()
    print_table(results)
