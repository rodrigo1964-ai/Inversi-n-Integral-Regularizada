"""
Integral Formulation for Unknown Input Reconstruction
======================================================

Reconstructs u(t) from noisy measurements of omega(t) and i(t)
by integrating the electrical equation, avoiding numerical differentiation.

Electrical equation:
    L(i) di/dt + R i + K_e omega = u

Integrate from t_{k-1} to t_k:
    u_k T ≈ [Phi(i_k) - Phi(i_{k-1})] + R (i_k + i_{k-1})/2 T + K_e (omega_k + omega_{k-1})/2 T

where Phi(i) = integral of L(i) di = L0 i_sat arctan(i / i_sat)
is the antiderivative of L(i) = L0 / (1 + (i/i_sat)^2).

No derivatives of noisy signals are needed -- only:
  - evaluation of arctan (smooth)
  - trapezoidal averages (smooth)
  - subtraction of consecutive cumulative sums (algebraic, no noise amplification)

Author: Rodolfo H. Rodrigo
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from motor_model import (R, K_e, K_t, J, b, L0, I_SAT, L_sat,
                          simulate, rk4_step)


# ---------------------------------------------------------------------------
# Antiderivative of L(i) = L0 / (1 + (i/i_sat)^2)
# ---------------------------------------------------------------------------

def Phi(i):
    """
    Antiderivative of L(i) w.r.t. i:
        Phi(i) = integral L(i) di = L0 * i_sat * arctan(i / i_sat)
    """
    return L0 * I_SAT * np.arctan(i / I_SAT)


# ---------------------------------------------------------------------------
# Integral-based input reconstruction (step-by-step)
# ---------------------------------------------------------------------------

def reconstruct_u_integral_step(i_meas, omega_meas, T):
    """
    Step-by-step integral reconstruction of u_k.

    From integrating the electrical eq over [t_{k-1}, t_k]:
        u_k T ≈ [Phi(i_k) - Phi(i_{k-1})]
                + R (i_k + i_{k-1})/2 T
                + K_e (omega_k + omega_{k-1})/2 T

    Parameters
    ----------
    i_meas : ndarray, shape (N,)
        Current measurements (possibly noisy)
    omega_meas : ndarray, shape (N,)
        Angular velocity measurements (possibly noisy)
    T : float
        Sampling period

    Returns
    -------
    u_rec : ndarray, shape (N,)
        Reconstructed voltage. u_rec[0] = NaN (no previous sample).
    """
    N = len(i_meas)
    u_rec = np.full(N, np.nan)

    for k in range(1, N):
        # Flux linkage difference (antiderivative evaluation, no derivative)
        delta_phi = Phi(i_meas[k]) - Phi(i_meas[k-1])

        # Trapezoidal averages for R*i and K_e*omega
        Ri_avg = R * (i_meas[k] + i_meas[k-1]) / 2.0
        Ke_omega_avg = K_e * (omega_meas[k] + omega_meas[k-1]) / 2.0

        u_rec[k] = delta_phi / T + Ri_avg + Ke_omega_avg

    return u_rec


# ---------------------------------------------------------------------------
# Cumulative integral reconstruction
# ---------------------------------------------------------------------------

def reconstruct_u_cumulative(i_meas, omega_meas, T):
    """
    Cumulative integral reconstruction.

    Define cumulative sums:
        S_u(k) = sum_{j=0}^{k} u_j T
        S_Ri(k) = sum_{j=0}^{k} R i_j T
        S_Ke(k) = sum_{j=0}^{k} K_e omega_j T

    Then:
        S_u(k) = Phi(i_k) - Phi(i_0) + S_Ri(k) + S_Ke(k)

    And u_k = [S_u(k) - S_u(k-1)] / T  (subtraction, not differentiation).

    This is algebraically equivalent to step-by-step but provides a different
    perspective: noise enters through Phi and cumulative sums, both smooth.

    Returns
    -------
    u_rec : ndarray, shape (N,)
    """
    N = len(i_meas)

    # Cumulative flux linkage
    phi_vals = Phi(i_meas)  # vectorized

    # Cumulative trapezoidal sums for R*i and K_e*omega
    cum_Ri = np.cumsum(R * i_meas) * T
    cum_Ke = np.cumsum(K_e * omega_meas) * T

    # Total cumulative integral of u
    S_u = (phi_vals - phi_vals[0]) + cum_Ri + cum_Ke

    # Recover u_k by differencing consecutive cumulative sums
    u_rec = np.full(N, np.nan)
    u_rec[1:] = np.diff(S_u) / T

    return u_rec


# ---------------------------------------------------------------------------
# Derivative-based reconstruction (for comparison)
# ---------------------------------------------------------------------------

def reconstruct_u_derivative_2pt(i_meas, omega_meas, T):
    """
    Point-by-point derivative approach (2-point forward difference).

    u_k = L(i_k) (i_{k+1} - i_k) / T + R i_k + K_e omega_k

    This amplifies noise by factor 1/T.
    """
    N = len(i_meas)
    u_rec = np.full(N, np.nan)

    for k in range(N - 1):
        di_dt = (i_meas[k+1] - i_meas[k]) / T
        u_rec[k] = L_sat(i_meas[k]) * di_dt + R * i_meas[k] + K_e * omega_meas[k]

    return u_rec


def reconstruct_u_derivative_3pt(i_meas, omega_meas, T):
    """
    3-point central difference approach.

    di/dt ≈ (i_{k+1} - i_{k-1}) / (2T)

    Slightly less noisy than 2-point, but still amplifies noise by 1/(2T).
    """
    N = len(i_meas)
    u_rec = np.full(N, np.nan)

    for k in range(1, N - 1):
        di_dt = (i_meas[k+1] - i_meas[k-1]) / (2.0 * T)
        u_rec[k] = L_sat(i_meas[k]) * di_dt + R * i_meas[k] + K_e * omega_meas[k]

    return u_rec


# ---------------------------------------------------------------------------
# RMSE utility
# ---------------------------------------------------------------------------

def rmse(u_true, u_rec):
    """RMSE over valid (non-NaN) entries."""
    mask = ~np.isnan(u_rec) & ~np.isnan(u_true)
    if mask.sum() == 0:
        return np.nan
    return np.sqrt(np.mean((u_true[mask] - u_rec[mask])**2))


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    np.random.seed(42)

    # ----- Generate ground truth -----
    T = 0.0001          # sampling period [s]
    t_final = 0.2       # [s]
    dt_rk4 = 1e-6       # RK4 fine step

    u_func = lambda t: 12.0 if t > 0.005 else 0.0
    x0 = np.array([0.0, 0.0])

    t_fine, states_fine, inputs_fine = simulate((0, t_final), x0, u_func, dt_rk4=dt_rk4)

    # Downsample to T
    step = int(T / dt_rk4)
    idx = np.arange(0, len(t_fine), step)
    t_s = t_fine[idx]
    omega_true = states_fine[idx, 0]
    i_true = states_fine[idx, 1]
    u_true = inputs_fine[idx]
    N = len(t_s)

    print(f"Samples: {N}, T={T*1e3:.2f} ms, t_final={t_final*1e3:.0f} ms")
    print(f"omega range: [{omega_true.min():.3f}, {omega_true.max():.3f}] rad/s")
    print(f"i range: [{i_true.min():.3f}, {i_true.max():.3f}] A")

    # ----- Test 1: Clean data -----
    print("\n=== CLEAN DATA ===")
    u_int_step = reconstruct_u_integral_step(i_true, omega_true, T)
    u_int_cum  = reconstruct_u_cumulative(i_true, omega_true, T)
    u_deriv_2  = reconstruct_u_derivative_2pt(i_true, omega_true, T)
    u_deriv_3  = reconstruct_u_derivative_3pt(i_true, omega_true, T)

    print(f"  Integral (step):     RMSE = {rmse(u_true, u_int_step):.6f} V")
    print(f"  Integral (cumul):    RMSE = {rmse(u_true, u_int_cum):.6f} V")
    print(f"  Derivative (2pt):    RMSE = {rmse(u_true, u_deriv_2):.6f} V")
    print(f"  Derivative (3pt):    RMSE = {rmse(u_true, u_deriv_3):.6f} V")

    # ----- Test 2: Varying noise levels -----
    noise_levels = [0.01, 0.05, 0.1, 0.5]
    results = {}

    for sigma in noise_levels:
        np.random.seed(42)
        omega_noisy = omega_true + np.random.normal(0, sigma, N)
        i_noisy = i_true + np.random.normal(0, sigma, N)

        u_is = reconstruct_u_integral_step(i_noisy, omega_noisy, T)
        u_ic = reconstruct_u_cumulative(i_noisy, omega_noisy, T)
        u_d2 = reconstruct_u_derivative_2pt(i_noisy, omega_noisy, T)
        u_d3 = reconstruct_u_derivative_3pt(i_noisy, omega_noisy, T)

        results[sigma] = {
            'integral_step': u_is,
            'integral_cum': u_ic,
            'deriv_2pt': u_d2,
            'deriv_3pt': u_d3,
            'rmse_is': rmse(u_true, u_is),
            'rmse_ic': rmse(u_true, u_ic),
            'rmse_d2': rmse(u_true, u_d2),
            'rmse_d3': rmse(u_true, u_d3),
        }

    print("\n=== NOISY DATA ===")
    print(f"{'sigma':>8s}  {'Integral(step)':>16s}  {'Integral(cum)':>16s}  {'Deriv(2pt)':>16s}  {'Deriv(3pt)':>16s}")
    print("-" * 80)
    for sigma in noise_levels:
        r = results[sigma]
        print(f"{sigma:8.2f}  {r['rmse_is']:16.4f}  {r['rmse_ic']:16.4f}  {r['rmse_d2']:16.4f}  {r['rmse_d3']:16.4f}")

    # ----- Plot -----
    fig, axes = plt.subplots(len(noise_levels) + 1, 1, figsize=(12, 3.5 * (len(noise_levels) + 1)),
                              sharex=True)

    ms = t_s * 1000  # time in ms

    # Row 0: clean data
    ax = axes[0]
    ax.plot(ms, u_true, 'k-', linewidth=1.5, label='True $u(t)$')
    ax.plot(ms, u_int_step, 'b-', linewidth=1.0, alpha=0.8, label='Integral (step)')
    ax.plot(ms, u_deriv_2, 'r-', linewidth=0.5, alpha=0.5, label='Derivative (2pt)')
    ax.set_ylabel('$u$ [V]')
    ax.set_title('Clean data', fontsize=11)
    ax.legend(loc='right', fontsize=8)
    ax.set_ylim(-2, 16)
    ax.grid(True, alpha=0.3)

    # Rows 1..4: noisy data
    for idx_s, sigma in enumerate(noise_levels):
        ax = axes[idx_s + 1]
        r = results[sigma]

        ax.plot(ms, u_true, 'k-', linewidth=1.5, label='True $u(t)$')
        ax.plot(ms, r['integral_step'], 'b-', linewidth=0.8, alpha=0.7,
                label=f'Integral (RMSE={r["rmse_is"]:.2f})')
        ax.plot(ms, r['deriv_2pt'], 'r-', linewidth=0.3, alpha=0.3,
                label=f'Deriv 2pt (RMSE={r["rmse_d2"]:.2f})')
        ax.plot(ms, r['deriv_3pt'], 'g-', linewidth=0.3, alpha=0.4,
                label=f'Deriv 3pt (RMSE={r["rmse_d3"]:.2f})')

        ax.set_ylabel('$u$ [V]')
        ax.set_title(f'Noise $\\sigma = {sigma}$', fontsize=11)
        ax.legend(loc='right', fontsize=7)
        ax.grid(True, alpha=0.3)

        # Set reasonable y-limits (clip extreme derivative values)
        y_lo = min(-5, np.nanmin(r['integral_step']))
        y_hi = max(20, np.nanmax(r['integral_step']))
        ax.set_ylim(max(y_lo, -50), min(y_hi, 60))

    axes[-1].set_xlabel('Time [ms]')
    plt.tight_layout()
    plt.savefig('/home/rodo/10Paper/results/integral_inverse.png', dpi=300,
                bbox_inches='tight')
    print("\nSaved figure to /home/rodo/10Paper/results/integral_inverse.png")

    # ----- Summary table -----
    print("\n=== NOISE AMPLIFICATION RATIO (Derivative RMSE / Integral RMSE) ===")
    for sigma in noise_levels:
        r = results[sigma]
        ratio_2pt = r['rmse_d2'] / r['rmse_is'] if r['rmse_is'] > 0 else np.inf
        ratio_3pt = r['rmse_d3'] / r['rmse_is'] if r['rmse_is'] > 0 else np.inf
        print(f"  sigma={sigma:.2f}:  2pt/integral = {ratio_2pt:.1f}x,  3pt/integral = {ratio_3pt:.1f}x")
