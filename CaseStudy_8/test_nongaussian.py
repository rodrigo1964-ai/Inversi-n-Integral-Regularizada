import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from experiment_nongaussian import generate_ground_truth, run_nongaussian_experiment

PASS = 0
FAIL = 0

def check(name, condition):
    global PASS, FAIL
    if condition:
        print(f"  PASS {name}")
        PASS += 1
    else:
        print(f"  FAIL {name}")
        FAIL += 1

if __name__ == '__main__':
    gt = generate_ground_truth()
    results = run_nongaussian_experiment(gt)

    print("\nCaseStudy_8: Non-Gaussian Noise Tests")
    print("=" * 50)

    noise_types = ['Gaussian', 'Laplacian', 'Impulsive']

    # TII RMSE < 0.5 V for ALL noise types
    for nt in noise_types:
        check(f"TII RMSE < 0.5 for {nt}", results['TII'][nt] < 0.5)

    # TII is best method for ALL noise types
    methods = ['Diff 3pt', 'EKF + Derivative', 'EKF + Integral', 'TII']
    for nt in noise_types:
        tii_rmse = results['TII'][nt]
        best = all(tii_rmse <= results[m][nt] for m in methods)
        check(f"TII is best for {nt}", best)

    # Diff 3pt degrades most under impulsive
    diffs = {m: results[m]['Impulsive'] - results[m]['Gaussian'] for m in methods}
    diff_worst = diffs['Diff 3pt'] >= max(diffs.values())
    check("Diff 3pt degrades most under impulsive", diff_worst)

    print(f"\nResults: {PASS} passed, {FAIL} failed")
    sys.exit(FAIL)
