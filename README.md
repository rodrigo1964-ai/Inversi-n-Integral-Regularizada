# Tikhonov Integral Inversion (TII)

[![DOI](https://img.shields.io/badge/DOI-10.6084%2Fm9.figshare.31920189-blue)](https://doi.org/10.6084/m9.figshare.31920189)

**Paper 10** — From Ill-Posed Differentiation to Well-Posed Integral Inversion:
A Tikhonov-Based Framework for Nonlinear Input Reconstruction.

**Target:** *Control Engineering Practice* (Elsevier / IFAC)

**Authors:** Rodolfo H. Rodrigo ([ORCID: 0000-0002-8787-0038](https://orcid.org/0000-0002-8787-0038)), Gustavo Schweickardt, H. Daniel Patiño

**Repository:** https://doi.org/10.6084/m9.figshare.31920189

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
| **Table 3** — Clean data accuracy | `cd CaseStudy_1 && python3 generate_figures.py` | `CaseStudy_1/table_3_clean.csv` |
| **Table 4** — Noise-dominated regime | `cd CaseStudy_2 && python3 generate_figures.py` | `CaseStudy_2/table_4_noise.csv` |
| **Table 5** — TII performance | `cd CaseStudy_3 && python3 generate_figures.py` | `CaseStudy_3/table_5_tii.csv` |
| **Table 6** — Direct regressor robustness | `cd CaseStudy_4 && python3 generate_figures.py` | `CaseStudy_4/table_6_robustness.csv` |
| **Table 7** — EKF deriv. vs integral | `cd CaseStudy_5 && python3 generate_figures.py` | `CaseStudy_5/table_7_ekf.csv` |
| **Table 8** — Comprehensive comparison | `cd CaseStudy_6 && python3 generate_figures.py` | `CaseStudy_6/table_8_comparison.csv` |
| **Table 9** — Ablation study | `cd CaseStudy_7 && python3 generate_figures.py` | `CaseStudy_7/table_9_ablation.csv` |
| **Table 10** — Non-Gaussian noise robustness | `cd CaseStudy_8 && python3 generate_figures.py` | `CaseStudy_8/table_10_nongaussian.csv` |
| **Table 11** — Model mismatch analysis | `cd CaseStudy_9 && python3 generate_figures.py` | `CaseStudy_9/table_11_mismatch.csv` |
| **Fig. 4** — λ sensitivity | `cd CaseStudy_10 && python3 generate_figures.py` | `CaseStudy_10/fig4_lambda_sensitivity.png` |
| **Table 12** — Computational complexity | `cd CaseStudy_11 && python3 generate_figures.py` | `CaseStudy_11/table_12_complexity.csv` |

**Note:** Tables 1 (comparison of approaches) and 2 (motor parameters) are
descriptive and not auto-generated.

---

## Repository Structure

```
10Paper/
│
├── CaseStudy_1/                    # §6.3: Clean Data Accuracy
│   ├── experiment_clean.py         #   All inverse methods on noise-free data
│   ├── generate_figures.py         #   → Fig 1, Fig 2(a-b), Table 3
│   └── test_clean.py              #   Validates truncation order hierarchy
│
├── CaseStudy_2/                    # §6.4: Noise-Dominated Regime
│   ├── experiment_noise.py         #   Accuracy reversal under noise
│   ├── generate_figures.py         #   → Fig 2(c), Table 4
│   └── test_noise.py              #   Validates 3pt/integral ratio ≈ 1.8×
│
├── CaseStudy_3/                    # §6.5: TII Performance
│   ├── experiment_tii.py           #   Grid search λ, two-layer suppression
│   ├── generate_figures.py         #   → Fig 2(d-e), Table 5
│   └── test_tii.py                #   Validates 90–309× improvement
│
├── CaseStudy_4/                    # §6.6: Direct Regressor Robustness
│   ├── experiment_robustness.py    #   State RMSE vs input noise σ_u
│   ├── generate_figures.py         #   → Table 6
│   └── test_robustness.py         #   Validates linear scaling
│
├── CaseStudy_5/                    # §6.7: EKF Derivative vs Integral
│   ├── experiment_ekf.py           #   Methods 5-6 across noise levels
│   ├── generate_figures.py         #   → Fig 2(f), Table 7
│   └── test_ekf.py                #   Validates divergence prevention
│
├── CaseStudy_6/                    # §6.8: Comprehensive Comparison
│   ├── experiment_comparison.py    #   All 7 methods, all noise levels
│   ├── generate_figures.py         #   → Fig 3, Table 8
│   └── test_comparison.py         #   Validates 166× total improvement
│
├── CaseStudy_7/                    # §6.9: Ablation Study
│   ├── experiment_ablation.py      #   Progressive component addition
│   ├── generate_figures.py         #   → Table 9
│   └── test_ablation.py           #   Validates each layer contribution
│
├── CaseStudy_8/                    # §6.10: Non-Gaussian Noise Robustness
│   ├── experiment_nongaussian.py   #   Laplacian + impulsive noise
│   ├── generate_figures.py         #   → Table 10
│   └── test_nongaussian.py        #   Validates structural robustness
│
├── CaseStudy_9/                    # §6.11: Model Mismatch Analysis
│   ├── experiment_mismatch.py      #   Parametric perturbations δ ∈ {0.1,0.2,0.3}
│   ├── generate_figures.py         #   → Table 11
│   └── test_mismatch.py           #   Validates graceful degradation
│
├── CaseStudy_10/                   # §6.12: Sensitivity to λ
│   ├── experiment_lambda.py        #   RMSE(λ) sweep
│   ├── generate_figures.py         #   → Fig 4
│   └── test_lambda.py             #   Validates broad plateau
│
├── CaseStudy_11/                   # §6.13: Computational Complexity
│   ├── experiment_complexity.py    #   Wall-time benchmarks
│   ├── generate_figures.py         #   → Table 12
│   └── test_complexity.py         #   Validates O(n) scaling
│
├── src/                            # Shared library
│   ├── motor_model.py              #   DC motor with L(i) saturation, RK4
│   ├── methods.py                  #   All methods: diff 2/3/4pt, integral, EKF×2, TII
│   ├── ekf.py                      #   Extended Kalman Filter (continuous-discrete)
│   └── run_experiments.py          #   Legacy monolithic script (reference)
│
├── Submission_CEP/                 # Manuscript for Control Engineering Practice
│   ├── paper_CEP.tex               #   elsarticle preprint format
│   ├── elsarticle.cls
│   ├── elsarticle-num.bst
│   ├── motor_ground_truth.png
│   ├── summary_publication.png
│   ├── noise_scaling.png
│   └── abstract.txt
│
├── Submission_IEEE/                # Original IEEE TSMC:S submission (archived)
│
├── regressor/                      # HFNN regressor from Paper 1 (reference)
├── results/                        # Legacy figures (from monolithic run)
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
| 1 | §6.3 | Clean data accuracy | 4pt: 6.98×10⁻⁷ V, Integral: 1.33×10⁻⁵ V |
| 2 | §6.4 | Noise-dominated regime | 3pt/Integral ratio ≈ 1.8× (consistent) |
| 3 | §6.5 | TII performance | 90–309× improvement over unregularized |
| 4 | §6.6 | Direct regressor robustness | Linear scaling, viable at σ_u = 0.1 V |
| 5 | §6.7 | EKF deriv. vs integral | 2.2× at σ=0.1, prevents divergence at σ=0.5 |
| 6 | §6.8 | Comprehensive comparison | TII: 0.102 V vs 16.89 V (3pt) → 166× |
| 7 | §6.9 | Ablation study | Each component contributes systematically |
| 8 | §6.10 | Non-Gaussian noise | TII stable under Laplacian and impulsive noise |
| 9 | §6.11 | Model mismatch | Graceful degradation with δ up to 30% |
| 10 | §6.12 | λ sensitivity | Robust over wide range, low tuning sensitivity |
| 11 | §6.13 | Computational complexity | O(n) — Thomas algorithm, <1 ms for n=2000 |

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

### Paper

```bibtex
@article{rodrigo2026tii,
  title={From Ill-Posed Differentiation to Well-Posed Integral Inversion:
         A Tikhonov-Based Framework for Nonlinear Input Reconstruction},
  author={Rodrigo, Rodolfo H. and Schweickardt, Gustavo
          and Pati{\~n}o, H. Daniel},
  journal={Control Engineering Practice},
  year={2026},
  note={Submitted}
}
```

### Code Repository

```bibtex
@software{rodrigo2026tii_code,
  author = {Rodrigo, Rodolfo H.},
  title = {Tikhonov Integral Inversion -- Implementation and Reproducibility Code},
  year = {2026},
  doi = {10.6084/m9.figshare.31920189},
  url = {https://doi.org/10.6084/m9.figshare.31920189}
}
```

## License

GPL-3.0 — See [LICENSE](LICENSE).
