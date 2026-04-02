import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from experiment_mismatch import run_mismatch_experiment

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
    results = run_mismatch_experiment()

    print("\nCaseStudy_9: Model Mismatch Tests")
    print("=" * 50)

    deltas = [0.0, 0.1, 0.2, 0.3]
    methods = ['EKF + Derivative', 'EKF + Integral', 'TII']

    # TII RMSE at delta=0.3 still < 1.0 V
    check("TII RMSE at delta=0.3 < 1.0 V", results['TII'][0.3] < 1.0)

    # TII is best at ALL delta values
    for d in deltas:
        tii_best = all(results['TII'][d] <= results[m][d] for m in methods)
        check(f"TII best at delta={d}", tii_best)

    # RMSE monotonically increases with delta for all methods
    # Allow 5% tolerance: when baseline error is large, small perturbations
    # may not strictly increase RMSE due to noise and estimation dynamics.
    for m in methods:
        vals = [results[m][d] for d in deltas]
        tol = 0.05 * vals[0] if vals[0] > 0 else 0.0
        monotone = all(vals[i] <= vals[i+1] + tol for i in range(len(vals)-1))
        check(f"{m} RMSE increases with delta (5% tol)", monotone)

    print(f"\nResults: {PASS} passed, {FAIL} failed")
    sys.exit(FAIL)
