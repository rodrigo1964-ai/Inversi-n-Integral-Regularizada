# Regularized Integral Inversion (TII)

**Unknown input reconstruction for nonlinear dynamic systems via Tikhonov-regularized integral inversion.**

DC motor with nonlinear magnetic saturation L(i) as case study.

---

## Methods

Seven methods are compared for reconstructing the unknown input voltage u(t) from noisy measurements of angular velocity ω(t) and current i(t):

| # | Method | File | Function | Description |
|---|--------|------|----------|-------------|
| 1 | Inverse Diff. 2pt | `src/methods.py` | `inverse_diff_2pt()` | Forward difference, O(T) |
| 2 | Inverse Diff. 3pt | `src/methods.py` | `inverse_diff_3pt()` | 3-point backward, O(T²) |
| 3 | Inverse Diff. 4pt | `src/methods.py` | `inverse_diff_4pt()` | 4-point backward, O(T³) |
| 4 | Inverse Integral | `src/methods.py` | `inverse_integral()` | Trapezoidal, no di/dt |
| 5 | EKF + Derivative | `src/methods.py` | `ekf_derivative()` | Kalman filter + method 3 |
| 6 | EKF + Integral | `src/methods.py` | `ekf_integral()` | Kalman filter + method 4 |
| 7 | **TII** | `src/methods.py` | `tii()` | **Tikhonov + Integral** |

## Reproduce all results

```bash
cd src
python3 run_experiments.py
```

Generates Tables II–V and Figures 3–4 from the paper. Output in `results/`.

## File structure

```
src/
├── motor_model.py        # DC motor model, RK4 ground truth, parameters
├── ekf.py                # Extended Kalman Filter (methods 5, 6)
├── methods.py            # All 7 methods in one file
├── run_experiments.py    # Reproduce all tables and figures
└── dual_estimation.py    # EKF pipeline with feedback

results/                  # Generated figures (PNG, 300 dpi)
docs/                     # LaTeX paper (IEEE format)
regressor/                # HFNN regressor from Paper 1 (reference)
legacy/                   # Earlier test scripts (not needed to reproduce)
```

## Motor parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| R | 1.0 Ω | Armature resistance |
| K_e | 0.1 V·s/rad | Back-EMF constant |
| K_t | 0.1 N·m/A | Torque constant |
| J | 0.001 kg·m² | Moment of inertia |
| b | 0.01 N·m·s/rad | Viscous friction |
| L₀ | 0.01 H | Unsaturated inductance |
| I_sat | 10.0 A | Saturation current |
| L(i) | L₀/(1+(i/I_sat)²) | Saturation model |

## Experiment settings

- **Sampling**: T = 0.1 ms, t_final = 0.2 s, n = 2000 samples
- **Ground truth**: RK4, dt = 1 μs
- **Input**: Step 0→12 V at t = 5 ms
- **Noise**: Gaussian, σ ∈ {0.01, 0.05, 0.1, 0.5} on both ω and i
- **EKF**: Q = diag(10⁻², 10⁻²), R = σ²
- **TII**: λ by grid search over {10⁻³, ..., 10³}
- **RMSE**: Skips first 100 samples (initialization transient)
- **Seed**: np.random.seed(42)

## Key result

At σ = 0.1, input reconstruction RMSE [V]:

| Method | RMSE | vs TII |
|--------|------|--------|
| (3) Diff. 4pt | 25.38 | 250x |
| (2) Diff. 3pt | 16.89 | 166x |
| (1) Diff. 2pt | 9.26 | 91x |
| (4) Integral | 9.33 | 92x |
| (5) EKF + Deriv | 3.38 | 33x |
| (6) EKF + Integ | 1.57 | 15x |
| **(7) TII** | **0.10** | **1x** |

TII solves a tridiagonal linear system in O(n) — computation time < 5 ms.

## Dependencies

Python 3.10+, NumPy, SciPy, Matplotlib.

## Author

Rodolfo H. Rodrigo — Departamento Electromecánica, Facultad de Ingeniería, UNSJ
