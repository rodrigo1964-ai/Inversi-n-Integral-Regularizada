import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
import time
import motor_model as mm
from methods import tii, ekf_integral

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

def run_complexity_experiment():
    sigma = 0.1
    T = 0.0001
    n_values = [500, 1000, 2000, 5000, 10000]
    n_reps = 5
    results = []

    for n_val in n_values:
        t_final = n_val * T
        gt = generate_ground_truth(T=T, t_final=t_final)

        np.random.seed(42)
        wn = gt['omega'] + np.random.normal(0, sigma, gt['n'])
        i_n = gt['i'] + np.random.normal(0, sigma, gt['n'])

        # Time TII
        tii_times = []
        for _ in range(n_reps):
            t0 = time.perf_counter()
            tii(wn, i_n, T, lam=100)
            tii_times.append((time.perf_counter() - t0) * 1000)

        # Time EKF
        ekf_times = []
        for _ in range(n_reps):
            t0 = time.perf_counter()
            ekf_integral(wn, T, gt['n'], sigma)
            ekf_times.append((time.perf_counter() - t0) * 1000)

        results.append({
            'n': n_val,
            'tii_ms': np.mean(tii_times),
            'ekf_ms': np.mean(ekf_times),
        })

    return results

def print_table(results):
    print(f"{'n':>8} {'TII [ms]':>10} {'EKF [ms]':>10}")
    print('-' * 30)
    for r in results:
        print(f"{r['n']:8d} {r['tii_ms']:10.2f} {r['ekf_ms']:10.2f}")

if __name__ == '__main__':
    results = run_complexity_experiment()
    print_table(results)
