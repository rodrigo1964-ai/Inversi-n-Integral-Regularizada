# CLAUDE.md — Project Contract for Dual HFNN + Kalman Paper

## Project Overview

This project develops a **second paper** extending the HFNN (Homotopy-Based Functional Neural Network) framework to simultaneous state and input estimation using nonlinear Kalman filtering. The first paper (reference: `rodrigo2025hfnn`) introduced the HFNN for grey-box identification of nonlinear dynamic systems. This paper generalizes the core idea: **HAM resolves any variable of an implicit equation F=0, not just the state**.

## Core Thesis

Given an implicit nonlinear discrete equation F(y_k, y_{k-1}, ..., u_k) = 0 arising from LMM discretization of a nonlinear ODE, the Homotopy Analysis Method produces explicit algebraic regressors for **any** variable of the equation. This yields a symmetric pair:
- **Direct HFNN:** resolves for y_k (state prediction)
- **Inverse HFNN:** resolves for u_k (input reconstruction)

Both have identical structure (Newton + Halley corrections), identical convergence guarantees, and analytical derivatives via embedded Gaussian RBF. Combined with EKF, this provides simultaneous filtered state and input estimation without state augmentation or random-walk hypotheses.

## Case Study

DC motor with nonlinear magnetic saturation L(i):
- **Mechanical:** J·dω/dt + b·ω + N_load(ω) = K_t·i → Direct HFNN predicts ω
- **Electrical:** u = R·i + L(i)·di/dt + K_e·ω → Inverse HFNN reconstructs i (implicit because L(i_k)·i_k is nonlinear)

The motor is a **benchmark**, not the contribution. Target journal: IEEE Trans. Systems, Man, and Cybernetics: Systems.

## Directory Structure

```
/home/rodo/10Paper/
├── CLAUDE.md                          # This file
├── docs/
│   └── dual_hfnn_kalman.tex           # Main paper (IEEE format, IEEEtran.cls)
├── src/                               # Python simulations (to be created)
│   ├── motor_model.py                 # DC motor with L(i) saturation (RK4 ground truth)
│   ├── hfnn_direct.py                 # Direct HFNN regressor (mechanical subsystem)
│   ├── hfnn_inverse.py                # Inverse HFNN regressor (electrical subsystem)
│   ├── ekf.py                         # EKF implementation with analytical Jacobians
│   ├── dual_estimation.py             # Full pipeline: direct + inverse + EKF
│   ├── train_lm.py                    # Levenberg-Marquardt training for both regressors
│   ├── rbf.py                         # Gaussian RBF with analytical N, N', N''
│   └── comparisons.py                 # Baselines: augmented EKF, UKF, no-filter
├── results/                           # Generated figures and tables (to be created)
├── IEEE-Transactions-LaTeX2e-templates-and-instructions/
│   ├── IEEEtran.cls                   # IEEE LaTeX class file
│   └── ...
```

## Key Equations (Reference)

### Inverse HFNN for current (electrical equation)

Implicit equation:
```
g_i(i_k) = R·i_k + L(i_k)·(i_k - i_{k-1})/T + K_e·ω_k - u_k = 0
```

Derivatives:
```
g_i'(i_k) = R + L'(i_k)·(i_k - i_{k-1})/T + L(i_k)/T
g_i''(i_k) = L''(i_k)·(i_k - i_{k-1})/T + 2·L'(i_k)/T
```

Regressor:
```
î_k = i_{k-1} + ζ₁ + ζ₂
ζ₁ = -g_i(i_{k-1}) / g_i'(i_{k-1})
ζ₂ = -(1/2)·g_i(i_{k-1})²·g_i''(i_{k-1}) / g_i'(i_{k-1})³
```

### EKF State

```
x_k = [ω_k, ω_{k-1}]ᵀ
Transition: x_{k+1} = [HFNN_direct(ω_k, ω_{k-1}, i_k; w), ω_k]ᵀ
Measurement: z_k = [1, 0]·x_k + v_k
```

### Filtered Input

```
û_{k-1} = G(x̂_{k|k}, u_{k-1}, i_{k-2})    (inverse HFNN on filtered state)
P_u = J_G · P_{k|k} · J_Gᵀ                  (uncertainty propagation)
```

## Motor Parameters (Baseline)

```python
R = 1.0        # Ohm, armature resistance
K_e = 0.01     # V·s/rad, back-EMF constant
K_t = 0.01     # N·m/A, torque constant
J = 0.01       # kg·m², moment of inertia
b = 0.1        # N·m·s/rad, viscous friction
T = 0.001      # s, sampling period

# Saturation model for L(i):
# L(i) = L0 / (1 + (i/i_sat)^2)  or similar monotone decreasing
L0 = 0.5       # H, unsaturated inductance
i_sat = 5.0    # A, saturation current
```

## Coding Conventions

- **Language:** Python 3.10+
- **Dependencies:** numpy, scipy, matplotlib
- **No deep learning frameworks** — all computations are explicit (RBF, LM, EKF are hand-coded)
- **Style:** Functions over classes where possible. Clear docstrings. Variables match paper notation.
- **Naming:** Use `omega` for ω, `i_k` for i_k, `L_sat` for L(i), etc.
- **Plots:** Publication-quality (matplotlib with LaTeX labels, 300 dpi PNG or PDF)
- **Random seed:** Fix np.random.seed(42) for reproducibility

## Tasks (Priority Order)

### Phase 1: Ground Truth Generation
1. Implement DC motor model (electrical + mechanical) with nonlinear L(i)
2. Generate true trajectories via RK4 with known parameters
3. Add measurement noise to ω (Gaussian, varying σ)

### Phase 2: Direct HFNN (Mechanical)
4. Implement Gaussian RBF with analytical derivatives (N, N', N'')
5. Implement direct HFNN regressor for ω_{k+1}
6. Train via Levenberg-Marquardt on 30-50 samples
7. Validate prediction accuracy (MSE, comparison with RK4)

### Phase 3: Inverse HFNN (Electrical)
8. Implement inverse HFNN regressor for i_k from implicit electrical eq.
9. Train L(i) RBF approximation
10. Validate current reconstruction accuracy

### Phase 4: EKF Integration
11. Implement EKF with analytical Jacobians from HFNN
12. Run dual estimation: filtered ω + reconstructed i from filtered state
13. Compare filtered vs unfiltered input estimation

### Phase 5: Comparisons and Paper Completion
14. Implement baselines: augmented EKF (random walk), UKF, direct inversion
15. Generate all figures and tables for Section 6 (Results)
16. Update docs/dual_hfnn_kalman.tex with results
17. Final review and formatting

## Critical Constraints

- **The inverse HFNN must use HAM, not algebraic inversion.** The electrical equation is genuinely implicit in i_k because L(i_k) is nonlinear. Verify that direct algebraic solution is impossible.
- **Both regressors share the same theoretical framework.** Do not use different methods for direct vs inverse.
- **Analytical Jacobians only.** No numerical differentiation in the EKF. The whole point is that RBF derivatives are exact.
- **Few samples.** Target 30-50 training samples to match the efficiency claim from Paper 1.
- **The motor is a benchmark, not the contribution.** Do not over-engineer the motor model. Keep it simple enough to demonstrate the concept cleanly.

## LaTeX Compilation

```bash
cd /home/rodo/10Paper/docs
cp ../IEEE-Transactions-LaTeX2e-templates-and-instructions/IEEEtran.cls .
pdflatex dual_hfnn_kalman.tex
pdflatex dual_hfnn_kalman.tex
```

(No BibTeX needed — references are inline via thebibliography.)

## Future Direction: Dual Kalman for Simultaneous State and Parameter Estimation

This paper opens a powerful next step. Since the direct HFNN regressor approximates the plant as:

```
ŷ_{k+1} = F_θ(y_k, u_k)
```

where θ are the trainable parameters (RBF weights, Padé coefficients), the architecture is naturally compatible with **Dual Kalman Filtering**: one filter estimates the state, another simultaneously estimates the parameters θ.

The structure would be:

1. **State filter (EKF 1):** estimates x_k using F_θ as transition model, with θ fixed at current estimate.
2. **Parameter filter (EKF 2):** estimates θ_k treating the state as known (from EKF 1), with dynamics θ_{k+1} = θ_k + w_θ.
3. **Inverse HFNN:** reconstructs filtered input from filtered state, as in this paper.

This is Dual Extended Kalman Filtering (DEKF), used in modern adaptive control. The key advantage here is that:

- The HFNN provides **analytical Jacobians** ∂F/∂x and ∂F/∂θ from the RBF structure — no numerical differentiation.
- The inverse regressor adapts automatically as θ changes, because both regressors derive from the same equation.
- The parameter adaptation inherits the data efficiency of the HFNN: few samples, fast convergence.

This would yield a **fully adaptive dual estimation architecture**: state, input, and parameters estimated simultaneously, all from the same implicit equation resolved by HAM for different variables, with Kalman providing optimal fusion at every level.

This is the natural third paper in the sequence:
1. Paper 1 (submitted): HFNN for grey-box identification.
2. Paper 2 (this project): Dual regressors + EKF for state and input estimation.
3. Paper 3 (future): Dual Kalman for adaptive state + parameter + input estimation.

## Embedded Implementation: IMUs on ESP32

The architecture has a property that most neural-network-based estimators lack: **it is a closed-form formula, not an iterative algorithm**. This makes it directly implementable on resource-constrained embedded hardware.

Consider an IMU (accelerometer + gyroscope) on an ESP32 microcontroller. The estimation pipeline per sample is:

1. **HFNN direct regressor:** one evaluation of the RBF (M Gaussian kernels), two rational expressions (ζ₁, ζ₂). Cost: ~M multiplications + M exponentials + a few divisions.
2. **HFNN inverse regressor:** same cost structure.
3. **EKF:** 2×2 or 3×3 matrices. Prediction, update, gain computation — all O(n²) with n ≤ 3.

No matrix inversions larger than 3×3. No iterations. No backpropagation. No ODE solver in the loop. The entire pipeline is deterministic in execution time — critical for real-time embedded systems.

For an IMU application:
- The nonlinear ODE comes from rigid body kinematics (gyroscope bias drift, accelerometer nonlinearity).
- The direct HFNN estimates orientation/attitude.
- The inverse HFNN reconstructs the effective angular rate or acceleration (the "true" input that the sensor actually experienced, stripped of bias and noise).
- The EKF fuses everything with quantified uncertainty.

This is competitive with the Madgwick/Mahony filters commonly used on ESP32 IMUs, but with two advantages: (a) it handles nonlinear sensor characteristics that those linear-assumption filters cannot, and (b) it provides formal convergence guarantees from the HAM framework.

The ESP32 has:
- 240 MHz dual-core processor
- Hardware floating-point unit
- Sufficient for RBF with M ≤ 20 centres at 1 kHz sampling

This positions the HFNN dual architecture not just as a theoretical tool but as a practical embedded estimation engine. A potential fourth paper or application note: real-time HFNN+EKF on ESP32 for IMU sensor fusion.

## Author

Rodolfo H. Rodrigo — Departamento Electromecánica, Facultad de Ingeniería, UNSJ
