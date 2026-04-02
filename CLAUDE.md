# CLAUDE.md — Contract for CaseStudy 7–11 Implementation

## Objective

Implement five new experiments (CaseStudy 7–11) for the paper "From Ill-Posed Differentiation to Well-Posed Integral Inversion" targeting Control Engineering Practice. These experiments were requested by the co-author (Daniel Patiño) and are **required for submission**. The paper (`Submission_CEP/paper_CEP.tex`) already has the tables with placeholder `---` entries that must be filled with real computed values.

## Context

- **Repository:** `/home/rodo/10Paper/`
- **Existing code:** `src/motor_model.py`, `src/methods.py`, `src/ekf.py` — DO NOT MODIFY these.
- **Existing CaseStudies 1–6:** Working, tested. Use them as pattern templates.
- **Paper tables to fill:** Tables 9 (ablation), 10 (non-Gaussian), 11 (mismatch), 12 (complexity), and Fig. 4 (λ sensitivity).

## Architecture Rules

1. **Each CaseStudy is a self-contained directory** with exactly three files:
   - `experiment_*.py` — runs the experiment, prints results to stdout
   - `generate_figures.py` — calls experiment, saves PNG (300 dpi) and CSV
   - `test_*.py` — validates key numerical assertions, exits 0/1

2. **Import pattern** (copy exactly from CaseStudy_3):
   ```python
   import sys, os
   sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
   import numpy as np
   import motor_model as mm
   from methods import inverse_diff_3pt, inverse_integral, ekf_derivative, ekf_integral, tii
   ```

3. **Ground truth generation** — reuse the same function in every CaseStudy:
   ```python
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
   ```

4. **RMSE function** (same everywhere):
   ```python
   def rmse(a, b, skip=100):
       s = slice(skip, None)
       mask = ~np.isnan(a[s]) & ~np.isnan(b[s])
       return np.sqrt(np.mean((a[s][mask] - b[s][mask])**2)) if mask.sum() > 0 else np.nan
   ```

5. **MAE function** (new, needed for ablation):
   ```python
   def mae(a, b, skip=100):
       s = slice(skip, None)
       mask = ~np.isnan(a[s]) & ~np.isnan(b[s])
       return np.mean(np.abs(a[s][mask] - b[s][mask])) if mask.sum() > 0 else np.nan
   ```

6. **Correlation function** (new):
   ```python
   def corr(a, b, skip=100):
       s = slice(skip, None)
       mask = ~np.isnan(a[s]) & ~np.isnan(b[s])
       if mask.sum() < 2:
           return np.nan
       return np.corrcoef(a[s][mask], b[s][mask])[0, 1]
   ```

7. **Random seed:** Always `np.random.seed(42)` before generating noise.

8. **Motor parameters:** Use `mm.R`, `mm.K_e`, `mm.K_t`, `mm.J`, `mm.b`, `mm.L0`, `mm.I_SAT`, `mm.C_LOAD` — never hardcode values.

9. **Figures:** matplotlib, `plt.savefig(fname, dpi=300, bbox_inches='tight')`, LaTeX labels via `plt.rc('text', usetex=False)` (keep it simple, no LaTeX requirement).

10. **CSV output:** `np.savetxt` or `csv.writer`, one header row + data rows.

## CaseStudy 7: Ablation Study

**Paper section:** §6.9, Table 9  
**Directory:** `CaseStudy_7/`  
**Files:** `experiment_ablation.py`, `generate_figures.py`, `test_ablation.py`

**What to do:**
At σ=0.1, run four configurations progressively:
1. `inverse_diff_3pt(wn, i_n, T)` — Differential only
2. `inverse_integral(wn, i_n, T)` — Integral only (no regularization)
3. `ekf_integral(wn, T, n, sigma)` — Integral + EKF (causal)
4. `tii(wn, i_n, T, lam=100)` — TII (full framework)

**Metrics per configuration:** RMSE, MAE, correlation ρ, stability (always "Yes" unless diverges = NaN > 50%).

**Output table (CSV):**
```
Configuration,RMSE_V,MAE_V,Correlation,Stable
Differential only,16.89,XX.XX,X.XXX,Yes
Integral only,9.33,XX.XX,X.XXX,Yes
Integral + EKF,1.57,XX.XX,X.XXX,Yes
TII (full),0.102,XX.XX,X.XXX,Yes
```

**Test assertions:**
- RMSE strictly decreasing across the 4 rows
- TII correlation > 0.99
- All configurations stable

## CaseStudy 8: Non-Gaussian Noise Robustness

**Paper section:** §6.10, Table 10  
**Directory:** `CaseStudy_8/`  
**Files:** `experiment_nongaussian.py`, `generate_figures.py`, `test_nongaussian.py`

**What to do:**
At σ=0.1, generate three noise types applied to BOTH ω and i:

1. **Gaussian** (baseline): `np.random.normal(0, sigma, n)`
2. **Laplacian** (heavy-tailed): `np.random.laplace(0, sigma/np.sqrt(2), n)` — scaled so Var = σ²
3. **Impulsive** (outliers): with probability 0.05, replace sample with `np.random.uniform(-5*sigma, 5*sigma)`; otherwise Gaussian.

Run four methods on each noise type: Diff 3pt, EKF+Derivative, EKF+Integral, TII (λ=100).

**IMPORTANT for EKF methods:** `ekf_derivative` and `ekf_integral` take `omega_meas` only (not `i_n`). The EKF only observes ω. So for non-Gaussian noise, apply it to `omega_meas`. For standalone methods (diff, integral, tii) that take both ω and i as input, apply noise to both.

**Output table (CSV):**
```
Method,Gaussian_RMSE,Laplacian_RMSE,Impulsive_RMSE
Diff 3pt,16.89,XX.XX,XX.XX
EKF + Derivative,3.38,XX.XX,XX.XX
EKF + Integral,1.57,XX.XX,XX.XX
TII,0.102,XX.XX,XX.XX
```

**Test assertions:**
- TII RMSE < 0.5 V for ALL noise types
- TII is best method for ALL noise types
- Diff 3pt degrades most under impulsive noise

## CaseStudy 9: Model Mismatch Analysis

**Paper section:** §6.11, Table 11  
**Directory:** `CaseStudy_9/`  
**Files:** `experiment_mismatch.py`, `generate_figures.py`, `test_mismatch.py`

**What to do:**
At σ=0.1, generate ground truth with **perturbed** parameters, but run estimation with **nominal** parameters.

Perturbation: multiply L0 and C_LOAD each by (1+δ) for δ ∈ {0, 0.1, 0.2, 0.3}.

**Implementation approach:**
- For ground truth generation: temporarily override `mm.L0` and `mm.C_LOAD`, simulate, then restore.
  ```python
  original_L0 = mm.L0
  original_C = mm.C_LOAD
  mm.L0 = original_L0 * (1 + delta)
  mm.C_LOAD = original_C * (1 + delta)
  gt = generate_ground_truth()
  mm.L0 = original_L0
  mm.C_LOAD = original_C
  ```
- For estimation: methods use the nominal parameters (unmodified `mm.*`).
- Run: EKF+Derivative, EKF+Integral, TII (λ=100) at each δ.

**Output table (CSV):**
```
Method,delta_0.0,delta_0.1,delta_0.2,delta_0.3
EKF + Derivative,3.38,XX.XX,XX.XX,XX.XX
EKF + Integral,1.57,XX.XX,XX.XX,XX.XX
TII,0.102,XX.XX,XX.XX,XX.XX
```

**Test assertions:**
- TII RMSE at δ=0.3 still < 1.0 V ("graceful degradation")
- TII is best method at ALL δ values
- RMSE monotonically increases with δ for all methods

## CaseStudy 10: Sensitivity to λ

**Paper section:** §6.12, Fig. 4  
**Directory:** `CaseStudy_10/`  
**Files:** `experiment_lambda.py`, `generate_figures.py`, `test_lambda.py`

**What to do:**
Sweep λ over `np.logspace(-3, 5, 50)` (50 points from 10⁻³ to 10⁵).
For each λ, compute TII RMSE at σ ∈ {0.01, 0.05, 0.1, 0.5}.

**Figure (CRITICAL — this is Fig. 4 in the paper):**
- X-axis: λ (log scale)
- Y-axis: RMSE [V] (log scale)
- 4 curves, one per σ level, with legend
- Save as `fig4_lambda_sensitivity.png`

**Output CSV:**
```
lambda,sigma_0.01,sigma_0.05,sigma_0.10,sigma_0.50
0.001,XX.XX,XX.XX,XX.XX,XX.XX
...
```

**Test assertions:**
- For each σ, there exists a plateau where RMSE varies < 2× across at least one decade of λ
- Optimal λ increases with σ

## CaseStudy 11: Computational Complexity

**Paper section:** §6.13, Table 12  
**Directory:** `CaseStudy_11/`  
**Files:** `experiment_complexity.py`, `generate_figures.py`, `test_complexity.py`

**What to do:**
Measure wall-clock time for TII and augmented EKF at varying n.
Use `n_values = [500, 1000, 2000, 5000, 10000]`.

For TII: call `tii(wn, i_n, T, lam=100)` and time it with `time.perf_counter()`.
For augmented EKF: call `ekf_integral(wn, T, n, sigma)` and time it.
Average over 5 repetitions each.

σ=0.1 for all runs.

**IMPORTANT:** For different n values, adjust `t_final = n * T` and regenerate ground truth accordingly.

**Output table (CSV):**
```
n,TII_time_ms,EKF_time_ms
500,XX.XX,XX.XX
1000,XX.XX,XX.XX
2000,XX.XX,XX.XX
5000,XX.XX,XX.XX
10000,XX.XX,XX.XX
```

**Test assertions:**
- TII time < 5 ms for n=2000
- TII time scales approximately linearly (time[10000]/time[1000] < 15, allowing overhead)
- TII faster than EKF at all n

## After Implementation

Once all 5 CaseStudies pass their tests, update the paper tables:
1. Read each CSV output
2. Fill the `---` entries in `Submission_CEP/paper_CEP.tex`
3. Copy `CaseStudy_10/fig4_lambda_sensitivity.png` to `Submission_CEP/`
4. Uncomment the `\includegraphics` for Fig. 4 in the paper
5. Recompile: `cd Submission_CEP && pdflatex paper_CEP && pdflatex paper_CEP`

## Execution Order

```bash
cd /home/rodo/10Paper

# Implement and test each CaseStudy in order
cd CaseStudy_7  && python3 experiment_ablation.py     && python3 test_ablation.py
cd ../CaseStudy_8  && python3 experiment_nongaussian.py  && python3 test_nongaussian.py
cd ../CaseStudy_9  && python3 experiment_mismatch.py     && python3 test_mismatch.py
cd ../CaseStudy_10 && python3 experiment_lambda.py       && python3 test_lambda.py
cd ../CaseStudy_11 && python3 experiment_complexity.py   && python3 test_complexity.py

# Generate all figures and CSVs
for d in CaseStudy_{7,8,9,10,11}; do
    cd /home/rodo/10Paper/$d && python3 generate_figures.py
done
```

## Do NOT

- Do NOT modify `src/motor_model.py`, `src/methods.py`, or `src/ekf.py`
- Do NOT modify CaseStudy 1–6
- Do NOT change motor parameters (R=1.0, K_e=0.1, etc.)
- Do NOT use deep learning frameworks
- Do NOT use random seeds other than 42
- Do NOT create figures with LaTeX rendering (use plain matplotlib text)
