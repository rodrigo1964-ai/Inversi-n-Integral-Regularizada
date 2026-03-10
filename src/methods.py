"""
Seven Methods for Unknown Input Reconstruction — DC Motor Case Study
=====================================================================

This file consolidates all seven methods compared in the paper.
Each method is clearly numbered to match the paper's Table II/III.

Methods:
    1. inverse_diff_2pt     — 2-point forward difference
    2. inverse_diff_3pt     — 3-point backward difference, O(T²)
    3. inverse_diff_4pt     — 4-point backward difference, O(T³)
    4. inverse_integral     — Trapezoidal integral (unregularized)
    5. ekf_derivative       — EKF state filter + 4pt derivative reconstruction
    6. ekf_integral         — EKF state filter + integral reconstruction
    7. tii                  — Tikhonov Integral Inversion (regularized)

All methods reconstruct u(t) from noisy measurements of ω(t) and i(t),
using the electrical equation:  L(i)·di/dt + R·i + K_e·ω = u

Ground truth: RK4 simulation with dt=1e-6 s.
Sampling: T = 0.1 ms, n = 2000 samples, t_final = 0.2 s.
Noise: Gaussian, σ applied to both ω and i.

Author: Rodolfo H. Rodrigo
"""

import numpy as np
from scipy.linalg import solve_banded
import motor_model as mm
from ekf import EKF


# ======================================================================
# Antiderivative of L(i) = L0 / (1 + (i/i_sat)^2)
# ======================================================================

def Phi(i):
    """Phi(i) = ∫L(i)di = L0·I_SAT·arctan(i/I_SAT)"""
    return mm.L0 * mm.I_SAT * np.arctan(i / mm.I_SAT)


# ======================================================================
# Method 1: Inverse Differential — 2-point forward difference
# ======================================================================

def inverse_diff_2pt(omega, i_arr, T):
    """
    Method 1: u_k = L(i_k)·(i_{k+1} - i_k)/T + R·i_k + K_e·ω_k

    Simplest finite difference. O(T) truncation error.
    Noise amplification factor: σ/T.
    """
    n = len(omega)
    u_hat = np.full(n, np.nan)
    for k in range(n - 1):
        di_dt = (i_arr[k+1] - i_arr[k]) / T
        u_hat[k] = mm.L_sat(i_arr[k]) * di_dt + mm.R * i_arr[k] + mm.K_e * omega[k]
    return u_hat


# ======================================================================
# Method 2: Inverse Differential — 3-point backward difference
# ======================================================================

def inverse_diff_3pt(omega, i_arr, T):
    """
    Method 2: di/dt ≈ (3i_k - 4i_{k-1} + i_{k-2}) / (2T)

    O(T²) truncation error.
    Noise amplification: √(9+16+1)/(2T) = √26/(2T) ≈ 2.55σ/T.
    """
    n = len(omega)
    u_hat = np.full(n, np.nan)
    for k in range(2, n):
        di_dt = (3*i_arr[k] - 4*i_arr[k-1] + i_arr[k-2]) / (2*T)
        u_hat[k] = mm.L_sat(i_arr[k]) * di_dt + mm.R * i_arr[k] + mm.K_e * omega[k]
    return u_hat


# ======================================================================
# Method 3: Inverse Differential — 4-point backward difference
# ======================================================================

def inverse_diff_4pt(omega, i_arr, T):
    """
    Method 3: di/dt ≈ (11i_k - 18i_{k-1} + 9i_{k-2} - 2i_{k-3}) / (6T)

    O(T³) truncation error.
    Noise amplification: √(121+324+81+4)/(6T) = √530/(6T) ≈ 3.83σ/T.
    """
    n = len(omega)
    u_hat = np.full(n, np.nan)
    for k in range(3, n):
        di_dt = (11*i_arr[k] - 18*i_arr[k-1] + 9*i_arr[k-2] - 2*i_arr[k-3]) / (6*T)
        u_hat[k] = mm.L_sat(i_arr[k]) * di_dt + mm.R * i_arr[k] + mm.K_e * omega[k]
    return u_hat


# ======================================================================
# Method 4: Inverse Integral — Trapezoidal (unregularized)
# ======================================================================

def inverse_integral(omega, i_arr, T):
    """
    Method 4: Integrate electrical eq. over [t_{k-1}, t_k]:

        u_k·T = [Phi(i_k) - Phi(i_{k-1})]
               + (T/2)·[R·i_{k-1} + K_e·ω_{k-1} + R·i_k + K_e·ω_k]

    Avoids explicit di/dt computation.
    Phi(i) = L0·I_SAT·arctan(i/I_SAT) is the antiderivative of L(i).
    R·i and K_e·ω terms are trapezoidal-averaged (smoothing).
    Noise amplification: L·√2·σ/T (same order as 2pt, smaller coefficient).
    """
    n = len(omega)
    u_hat = np.full(n, np.nan)
    for k in range(1, n):
        phi_diff = Phi(i_arr[k]) - Phi(i_arr[k-1])
        Ri_avg = mm.R * (i_arr[k-1] + i_arr[k]) / 2.0
        Ke_avg = mm.K_e * (omega[k-1] + omega[k]) / 2.0
        u_hat[k] = phi_diff / T + Ri_avg + Ke_avg
    return u_hat


# ======================================================================
# Method 5: EKF + Derivative reconstruction
# ======================================================================

def ekf_derivative(omega_meas, T, n, sigma):
    """
    Method 5: Extended Kalman Filter + 4-point backward derivative.

    1. EKF filters state [ω, i] from noisy ω measurements.
    2. Reconstructs u via 4pt derivative of filtered i.
    3. Feeds back û for next EKF prediction.

    Parameters: Q = diag(1e-2, 1e-2), R_meas = σ².
    """
    Q = np.diag([1e-2, 1e-2])
    ekf = EKF(T=T, Q=Q, R_meas=sigma**2,
              x0=np.array([0.0, 0.0]),
              P0=np.eye(2) * 1.0, use_integral=False)

    omega_f = np.zeros(n)
    i_f = np.zeros(n)
    u_hat = np.zeros(n)
    u_est = 0.0

    for k in range(1, n):
        x_f, _, u_r = ekf.step(omega_meas[k], u_est=u_est)
        omega_f[k] = x_f[0]
        i_f[k] = x_f[1]
        u_hat[k] = u_r
        u_est = u_r

    return u_hat, omega_f, i_f


# ======================================================================
# Method 6: EKF + Integral reconstruction
# ======================================================================

def ekf_integral(omega_meas, T, n, sigma):
    """
    Method 6: Extended Kalman Filter + integral reconstruction.

    Same EKF as Method 5, but uses Phi-based integral formula
    instead of finite differences for u reconstruction.
    """
    Q = np.diag([1e-2, 1e-2])
    ekf = EKF(T=T, Q=Q, R_meas=sigma**2,
              x0=np.array([0.0, 0.0]),
              P0=np.eye(2) * 1.0, use_integral=True)

    omega_f = np.zeros(n)
    i_f = np.zeros(n)
    u_hat = np.zeros(n)
    u_est = 0.0

    for k in range(1, n):
        x_f, _, u_r = ekf.step(omega_meas[k], u_est=u_est)
        omega_f[k] = x_f[0]
        i_f[k] = x_f[1]
        u_hat[k] = u_r
        u_est = u_r

    return u_hat, omega_f, i_f


# ======================================================================
# Method 7: TII — Tikhonov Integral Inversion
# ======================================================================

def tii(omega, i_arr, T, lam):
    """
    Method 7: Tikhonov Integral Inversion.

    Solves the regularized linear system:
        (I + λ·D^T·D) u = b

    where b_k = inverse_integral point-by-point estimate,
    and D is the first-difference matrix (roughness penalty).

    D^T·D is tridiagonal → solved in O(n) via banded solver.

    Parameters
    ----------
    omega, i_arr : ndarray, shape (n,)
        Noisy measurements
    T : float
        Sampling period
    lam : float
        Regularization parameter (λ)

    Returns
    -------
    u_tii : ndarray, shape (n,)
        Regularized input reconstruction
    """
    n = len(omega)

    # Step 1: unregularized estimate (Method 4)
    b = inverse_integral(omega, i_arr, T)
    b[0] = 0.0  # replace NaN

    # Step 2: build tridiagonal system (I + λ·D^T·D)
    # D^T·D diagonal:  [1, 2, 2, ..., 2, 1]
    # D^T·D off-diag:  [-1, -1, ..., -1]
    diag_main = np.ones(n) + lam * 2.0
    diag_main[0] = 1.0 + lam
    diag_main[-1] = 1.0 + lam
    diag_off = -lam * np.ones(n - 1)

    # Banded format for scipy: ab[0] = upper diagonal, ab[1] = main, ab[2] = lower
    ab = np.zeros((3, n))
    ab[0, 1:] = diag_off      # upper diagonal
    ab[1, :] = diag_main       # main diagonal
    ab[2, :-1] = diag_off      # lower diagonal

    u_tii = solve_banded((1, 1), ab, b)
    return u_tii
