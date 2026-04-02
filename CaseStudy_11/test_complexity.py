import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from experiment_complexity import run_complexity_experiment

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
    results = run_complexity_experiment()

    print("\nCaseStudy_11: Computational Complexity Tests")
    print("=" * 50)

    # Find n=2000 entry
    r2000 = next(r for r in results if r['n'] == 2000)
    check("TII time < 10 ms for n=2000", r2000['tii_ms'] < 10)

    # TII scales approximately linearly
    r1000 = next(r for r in results if r['n'] == 1000)
    r10000 = next(r for r in results if r['n'] == 10000)
    ratio = r10000['tii_ms'] / r1000['tii_ms']
    check(f"TII scales ~linearly (ratio={ratio:.1f}, need <15)", ratio < 15)

    # TII faster than EKF at all n
    for r in results:
        check(f"TII faster than EKF at n={r['n']}", r['tii_ms'] < r['ekf_ms'])

    print(f"\nResults: {PASS} passed, {FAIL} failed")
    sys.exit(FAIL)
