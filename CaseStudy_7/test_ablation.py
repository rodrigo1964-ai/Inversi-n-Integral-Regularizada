import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from experiment_ablation import generate_ground_truth, run_ablation_experiment

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
    results = run_ablation_experiment(gt)

    print("\nCaseStudy_7: Ablation Study Tests")
    print("=" * 50)

    rmses = [r['rmse'] for r in results]
    check("RMSE strictly decreasing", all(rmses[i] > rmses[i+1] for i in range(len(rmses)-1)))
    check("TII correlation > 0.99", results[-1]['corr'] > 0.99)
    check("All configurations stable", all(r['stable'] == 'Yes' for r in results))

    print(f"\nResults: {PASS} passed, {FAIL} failed")
    sys.exit(FAIL)
