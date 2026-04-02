# Tikhonov Integral Inversion (TII)

**Paper 10** — Tikhonov Integral Inversion for Simultaneous State and Input
Estimation in Nonlinear Systems Using Dual Homotopy-Based Regressors.

**Target:** *IEEE Transactions on Systems, Man, and Cybernetics: Systems*

**Authors:** Rodolfo H. Rodrigo, Gustavo Schweickardt, Daniel H. Patiño

---

## Reproducibility Guide

### Requirements

```bash
Python >= 3.10
pip install numpy scipy matplotlib
```

All experiments use `np.random.seed(42)` for reproducibility.
Results were generated with NumPy 1.26, SciPy 1.12, Matplotlib 3.8
on Ubuntu 24.04 / Python 3.12.

### Reproduce everything at once

```bash
chmod +x run_all.sh
./run_all.sh              # Tests + all figures (~5 min)
./run_all.sh --tests      # Only validation tests (~3 min)
./run_all.sh --figures    # Only figure generation (~4 min)
```

### Reproduce individual figures and tables

| Paper element | Command | Output file |
|:---|:---|:---|
| **Fig. 1** — Ground truth trajectories | `cd CaseStudy_1 && python3 generate_figures.py` | `CaseStudy_1/fig1_ground_truth.png` |
| **Fig. 2(a)** — Direct regressors, clean | (same as above) | `CaseStudy_1/fig2a_direct_clean.png` |
| **Fig. 2(b)** — Inverse reconstruction, clean | (same as above) | `CaseStudy_1/fig2b_inverse_clean.png` |
| **Fig. 2(c)** — Noisy inverse, no regularization | `cd CaseStudy_2 && python3 generate_figures.py` | `CaseStudy_2/fig2c_noisy_inverse.png` |
| **Fig. 2(d)** — TII at σ=0.1 | `cd CaseStudy_3 && python3 generate_figures.py` | `CaseStudy_3/fig2d_tii.png` |
| **Fig. 2(e)** — TII improvement across noise levels | (same as above) | `CaseStudy_3/fig2e_tii_improvement.png` |
| **Fig. 2(f)** — EKF derivative vs integral | `cd CaseStudy_5 && python3 generate_figures.py` | `CaseStudy_5/fig2f_ekf_comparison.png` |
| **Fig. 3** — RMSE vs noise (log-log) | `cd CaseStudy_6 && python3 generate_figures.py` | `CaseStudy_6/fig3_noise_scaling.png` |
| **Table III** — Clean data accuracy | `cd CaseStudy_1 && python3 generate_figures.py` | `CaseStudy_1/table_III_clean.csv` |
| **Table IV** — Noise-dominated regime | `cd CaseStudy_2 && python3 generate_figures.py` | `CaseStudy_2/table_IV_noise.csv` |
| **Table V** — TII performance | `cd CaseStudy_3 && python3 generate_figures.py` | `CaseStudy_3/table_V_tii.csv` |
| **Table VI** — Direct regressor robustness | `cd CaseStudy_4 && python3 generate_figures.py` | `CaseStudy_4/table_VI_robustness.csv` |
| **Table VII** — EKF deriv. vs integral | `cd CaseStudy_5 && python3 generate_figures.py` | `CaseStudy_5/table_VII_ekf.csv` |
| **Table VIII** — Comprehensive comparison | `cd CaseStudy_6 && python3 generate_figures.py` | `CaseStudy_6/table_VIII_comparison.csv` |

**Note:** Tables I (comparison of approaches) and II (motor parameters) are
descriptive and not auto-generated.

---

## Repository Structure

```
10Paper/
│
├── CaseStudy_1/                    # §VI.C: Clean Data Accuracy
│   ├── experiment_clean.py         #   All inverse methods on noise-free data
│   ├── generate_figures.py         #   → Fig 1, Fig 2(a-b), Table III
│   └── test_clean.py              #   Validates truncation order hierarchy
│
├── CaseStudy_2/                    # §VI.D: Noise-Dominated Regime
│   ├── experiment_noise.py         #   Accuracy reversal under noise
│   ├── generate_figures.py         #   → Fig 2(c), Table IV
│   └── test_noise.py              #   Validates 3pt/integral ratio ≈ 1.8×
│
├── CaseStudy_3/                    # §VI.E: TII Performance
│   ├── experiment_tii.py           #   Grid search λ, two-layer suppression
│   ├── generate_figures.py         #   → Fig 2(d-e), Table V
│   └── test_tii.py                #   Validates 90–309× improvement
│
├── CaseStudy_4/                    # §VI.F: Direct Regressor Robustness
│   ├── experiment_robustness.py    #   State RMSE vs input noise σ_u
│   ├── generate_figures.py         #   → Table VI
│   └── test_robustness.py         #   Validates linear scaling
│
├── CaseStudy_5/                    # §VI.G: EKF Derivative vs Integral
│   ├── experiment_ekf.py           #   Methods 5-6 across noise levels
│   ├── generate_figures.py         #   → Fig 2(f), Table VII
│   └── test_ekf.py                #   Validates divergence prevention
│
├── CaseStudy_6/                    # §VI.H: Comprehensive Comparison
│   ├── experiment_comparison.py    #   All 7 methods, all noise levels
│   ├── generate_figures.py         #   → Fig 3, Table VIII
│   └── test_comparison.py         #   Validates 166× total improvement
│
├── src/                            # Shared library (imported by all CaseStudy scripts)
│   ├── motor_model.py              #   DC motor with L(i) saturation, RK4 ground truth
│   ├── methods.py                  #   All 7 methods: diff 2/3/4pt, integral, EKF×2, TII
│   ├── ekf.py                      #   Extended Kalman Filter (continuous-discrete)
│   └── run_experiments.py          #   Legacy monolithic script (kept for reference)
│
├── docs/                           # Manuscript (IEEE format)
├── results/                        # Legacy figures (from monolithic run)
├── regressor/                      # HFNN regressor from Paper 1 (reference)
│
├── run_all.sh                      # ./run_all.sh [--tests | --figures]
├── requirements.txt
├── CLAUDE.md                       # Project contract
├── LICENSE
└── README.md                       # This file
```

---

## Experimental Summary

| Case Study | Paper § | Experiment | Key Result |
|:---:|:---|:---|:---|
| 1 | §VI.C | Clean data accuracy | 4pt: 6.98×10⁻⁷ V, Integral: 1.33×10⁻⁵ V |
| 2 | §VI.D | Noise-dominated regime | 3pt/Integral ratio ≈ 1.8× (consistent) |
| 3 | §VI.E | TII performance | 90–309× improvement over unregularized |
| 4 | §VI.F | Direct regressor robustness | Linear scaling, viable at σ_u = 0.1 V |
| 5 | §VI.G | EKF deriv. vs integral | 2.2× at σ=0.1, prevents divergence at σ=0.5 |
| 6 | §VI.H | Comprehensive comparison | TII: 0.102 V vs 16.89 V (3pt) → 166× |

## How it works

Each `CaseStudy_N/` is self-contained:

1. `experiment_*.py` defines the experiment, generates ground truth, runs methods,
   and prints tables to stdout.
2. `generate_figures.py` calls the experiment pipeline and saves publication-quality
   figures (PNG, 300 dpi) and tables (CSV) in the same directory.
3. `test_*.py` validates key numerical assertions without generating figures.

The core library in `src/` provides:
- `motor_model.py`: DC motor with nonlinear L(i) saturation, RK4 simulation,
  analytical Jacobians.
- `methods.py`: All 7 methods consolidated: 3 differential stencils, integral,
  EKF+derivative, EKF+integral, TII (Tikhonov + tridiagonal solver).
- `ekf.py`: Continuous-discrete EKF with Joseph-form covariance update,
  supporting both derivative and integral input reconstruction.

## Motor Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| R | 1.0 Ω | Armature resistance |
| K_e, K_t | 0.1 V·s/rad, 0.1 N·m/A | Motor constants |
| J | 0.001 kg·m² | Moment of inertia |
| b | 0.01 N·m·s/rad | Viscous friction |
| L₀ | 0.01 H | Unsaturated inductance |
| I_sat | 10.0 A | Saturation current |
| L(i) | L₀/(1+(i/I_sat)²) | Saturation model |

## Key Result (σ = 0.1)

| Method | RMSE [V] | vs TII |
|--------|----------|--------|
| (2) Diff. 3pt | 16.89 | 166× |
| (1) Diff. 2pt | 9.41 | 93× |
| (4) Integral | 9.33 | 92× |
| (5) EKF + Deriv | 3.38 | 33× |
| (6) EKF + Integ | 1.57 | 15× |
| **(7) TII** | **0.102** | **1×** |

## Citation

```bibtex
@article{rodrigo2026tii,
  title={Tikhonov Integral Inversion for Simultaneous State and Input
         Estimation in Nonlinear Systems Using Dual Homotopy-Based Regressors},
  author={Rodrigo, Rodolfo H. and Schweickardt, Gustavo
          and Pati{\~n}o, Daniel H.},
  journal={IEEE Trans. Syst., Man, Cybern., Syst.},
  year={2026},
  note={Submitted}
}
```

## License

GPL-3.0 — See [LICENSE](LICENSE).
