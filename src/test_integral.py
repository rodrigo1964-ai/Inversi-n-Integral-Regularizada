"""
Test de las 4 formulaciones: directa/inversa × diferencial/integral.

Sistema motor DC 2x2:
  Mecánica:  J·ω' = K_t·i - b·ω - N_load(ω)
  Eléctrica: L(i)·i' = u - R·i - K_e·ω

Formulación diferencial (3pt backward):
  Reemplaza y' → (3y_k - 4y_{k-1} + y_{k-2})/(2T)

Formulación integral (Simpson 3pt):
  Integra de t_{k-2} a t_k, usa cuadratura Simpson 1/3:
  ∫f dt ≈ (T/3)·[f_{k-2} + 4f_{k-1} + f_k]

HAM se aplica a ambas. Referencia: RK4 sin ruido.

Author: Rodolfo H. Rodrigo
"""

import sys
sys.path.insert(0, '/home/rodo/10Paper/regressor')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import motor_model as mm


# ======================================================================
# Antiderivada de L(i) para la formulación integral
# ======================================================================

def Phi(i):
    """∫L(i)di = L0·i_sat·arctan(i/i_sat)"""
    return mm.L0 * mm.I_SAT * np.arctan(i / mm.I_SAT)


# ======================================================================
# 1. DIRECTA DIFERENCIAL (ya probada en test_direct.py, reimplementada acá)
# ======================================================================

def direct_differential(u, omega_0, omega_1, i_0, i_1, T, n):
    """Regressor directo diferencial: u → [ω, i] con 3pt backward + HAM."""
    omega = np.zeros(n)
    i_arr = np.zeros(n)
    omega[0], omega[1] = omega_0, omega_1
    i_arr[0], i_arr[1] = i_0, i_1

    c = mm.C_LOAD

    for k in range(2, n):
        # Estimación inicial
        wk = omega[k-1]
        ik = i_arr[k-1]

        # --- z1 (Newton) ---
        wp = (3*wk - 4*omega[k-1] + omega[k-2]) / (2*T)
        ip = (3*ik - 4*i_arr[k-1] + i_arr[k-2]) / (2*T)

        g1 = wp + (mm.b*wk + c*wk**2 - mm.K_t*ik) / mm.J
        g2 = mm.L_sat(ik)*ip + mm.R*ik + mm.K_e*wk - u[k]

        j00 = (mm.b + 2*c*wk)/mm.J + 3/(2*T)
        j01 = -mm.K_t/mm.J
        j10 = mm.K_e
        j11 = mm.dL_di(ik)*ip + mm.R + mm.L_sat(ik)*3/(2*T)

        det = j00*j11 - j01*j10
        dw = (j11*g1 - j01*g2) / det
        di = (-j10*g1 + j00*g2) / det

        wk -= dw
        ik -= di

        # --- z2 (Halley) ---
        wp = (3*wk - 4*omega[k-1] + omega[k-2]) / (2*T)
        ip = (3*ik - 4*i_arr[k-1] + i_arr[k-2]) / (2*T)

        g1 = wp + (mm.b*wk + c*wk**2 - mm.K_t*ik) / mm.J
        g2 = mm.L_sat(ik)*ip + mm.R*ik + mm.K_e*wk - u[k]

        j00 = (mm.b + 2*c*wk)/mm.J + 3/(2*T)
        j01 = -mm.K_t/mm.J
        j10 = mm.K_e
        j11 = mm.dL_di(ik)*ip + mm.R + mm.L_sat(ik)*3/(2*T)

        # Hessiano (solo diagonales dominantes)
        h1_00 = 2*c/mm.J
        h2_11 = mm.d2L_di2(ik)*ip + 2*mm.dL_di(ik)*3/(2*T)

        det = j00*j11 - j01*j10
        G = np.array([g1, g2])
        dz1 = np.array([dw, di])  # delta del paso anterior

        # Producto Hessiano H[dz1, dz1] (simplificado, solo diagonal)
        H1 = h1_00 * dw**2
        H2 = h2_11 * di**2

        dz2_1 = -0.5 * (j11*H1 - j01*H2) / det
        dz2_2 = -0.5 * (-j10*H1 + j00*H2) / det

        wk += dz2_1
        ik += dz2_2

        omega[k] = wk
        i_arr[k] = ik

    return omega, i_arr


# ======================================================================
# 2. DIRECTA INTEGRAL (Simpson 3pt + HAM)
# ======================================================================

def direct_integral(u, omega_0, omega_1, i_0, i_1, T, n):
    """Regressor directo integral: u → [ω, i] con trapezoidal + HAM.

    Integra de t_{k-1} a t_k con regla trapezoidal (A-estable):
      J·(ω_k - ω_{k-1}) + (T/2)·[h1_{k-1} + h1_k] = 0
      Phi(i_k) - Phi(i_{k-1}) + (T/2)·[h2_{k-1} + h2_k] - (T/2)·[u_{k-1}+u_k] = 0

    Jacobiano dominado por J y L(i), no por 1/T.
    Solo requiere 1 condición inicial (no 2).
    """
    omega = np.zeros(n)
    i_arr = np.zeros(n)
    omega[0] = omega_0
    i_arr[0] = i_0

    c = mm.C_LOAD
    T2 = T / 2.0

    for k in range(1, n):
        wk = omega[k-1]
        ik = i_arr[k-1]

        w_prev = omega[k-1]
        i_prev = i_arr[k-1]

        # Términos en k-1 (conocidos)
        h1_prev = mm.b*w_prev + c*w_prev**2 - mm.K_t*i_prev
        h2_prev = mm.R*i_prev + mm.K_e*w_prev

        u_sum = u[k-1] + u[k]

        # 2 iteraciones Newton
        for _it in range(2):
            h1_k = mm.b*wk + c*wk**2 - mm.K_t*ik
            h2_k = mm.R*ik + mm.K_e*wk

            g1 = mm.J*(wk - w_prev) + T2*(h1_prev + h1_k)
            g2 = (Phi(ik) - Phi(i_prev)) + T2*(h2_prev + h2_k) - T2*u_sum

            j00 = mm.J + T2*(mm.b + 2*c*wk)
            j01 = -T2*mm.K_t
            j10 = T2*mm.K_e
            j11 = mm.L_sat(ik) + T2*mm.R

            det = j00*j11 - j01*j10
            dw = (j11*g1 - j01*g2) / det
            di = (-j10*g1 + j00*g2) / det

            wk -= dw
            ik -= di

        omega[k] = wk
        i_arr[k] = ik

    return omega, i_arr


# ======================================================================
# 3. INVERSA DIFERENCIAL (3pt backward)
# ======================================================================

def inverse_differential(omega, i_arr, T, n_points=3):
    """Reconstruye u desde (ω, i) con derivada discreta de i."""
    n = len(omega)
    u_hat = np.full(n, np.nan)

    for k in range(n_points - 1, n):
        ik = i_arr[k]
        L_ik = mm.L_sat(ik)

        if n_points == 2:
            di_dt = (i_arr[k] - i_arr[k-1]) / T
        elif n_points == 3:
            di_dt = (3*i_arr[k] - 4*i_arr[k-1] + i_arr[k-2]) / (2*T)
        elif n_points == 4:
            di_dt = (11*i_arr[k] - 18*i_arr[k-1] + 9*i_arr[k-2] - 2*i_arr[k-3]) / (6*T)

        u_hat[k] = L_ik * di_dt + mm.R * ik + mm.K_e * omega[k]

    return u_hat


# ======================================================================
# 4. INVERSA INTEGRAL (Simpson 3pt)
# ======================================================================

def inverse_integral(omega, i_arr, T):
    """
    Reconstruye u desde (ω, i) con formulación integral paso a paso.

    Integra la ec. eléctrica de t_{k-1} a t_k:
      ∫L(i)di/dt dt = Phi(i_k) - Phi(i_{k-1})     (exacto)
      ∫Ri dt ≈ R·T·(i_{k-1}+i_k)/2                (trapezoidal, suaviza)
      ∫K_e·ω dt ≈ K_e·T·(ω_{k-1}+ω_k)/2           (trapezoidal, suaviza)
      ∫u dt ≈ u_k·T                                (u constante en intervalo)

    Resultado (sin recursión):
      u_k = [Phi(i_k) - Phi(i_{k-1})]/T + R·(i_{k-1}+i_k)/2 + K_e·(ω_{k-1}+ω_k)/2

    No amplifica ruido como la derivada: Phi evalúa arctan (suave)
    y los términos resistivos/back-EMF se promedian.
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
# Test principal
# ======================================================================

def run_test(T=0.0001, t_final=0.2):
    """Compara las 4 formulaciones contra RK4 sin ruido."""

    n = int(t_final / T)

    # Ground truth RK4
    u_func = lambda t: 12.0 if t > 0.005 else 0.0
    x0 = np.array([0.0, 0.0])
    t_rk4, states_rk4, inputs_rk4 = mm.simulate(
        (0, t_final), x0, u_func, dt_rk4=1e-6
    )

    step = int(T / 1e-6)
    idx = np.arange(0, len(t_rk4), step)[:n]
    t_ref = t_rk4[idx]
    omega_ref = states_rk4[idx, 0]
    i_ref = states_rk4[idx, 1]
    u_ref = inputs_rk4[idx]

    print(f"T = {T:.0e},  n = {n}")

    # --- 1. Directa diferencial ---
    omega_dd, i_dd = direct_differential(
        u_ref, omega_ref[0], omega_ref[1], i_ref[0], i_ref[1], T, n
    )

    # --- 2. Directa integral (trapezoidal, solo 1 IC) ---
    omega_di, i_di = direct_integral(
        u_ref, omega_ref[0], None, i_ref[0], None, T, n
    )

    # --- 3. Inversa diferencial (3pt) ---
    u_invd = inverse_differential(omega_ref, i_ref, T, n_points=3)

    # --- 4. Inversa integral ---
    u_invi = inverse_integral(omega_ref, i_ref, T)

    # Errores (skip transient)
    skip = int(0.01 / T)
    s = slice(skip, None)

    print(f"\n{'Formulación':<25s}  {'ω RMSE':>12s}  {'i RMSE':>12s}  {'u RMSE':>12s}")
    print("-" * 65)

    ew_dd = np.sqrt(np.mean((omega_dd[s] - omega_ref[s])**2))
    ei_dd = np.sqrt(np.mean((i_dd[s] - i_ref[s])**2))
    print(f"{'Directa diferencial':<25s}  {ew_dd:12.4e}  {ei_dd:12.4e}  {'—':>12s}")

    ew_di = np.sqrt(np.mean((omega_di[s] - omega_ref[s])**2))
    ei_di = np.sqrt(np.mean((i_di[s] - i_ref[s])**2))
    print(f"{'Directa integral':<25s}  {ew_di:12.4e}  {ei_di:12.4e}  {'—':>12s}")

    mask_d = ~np.isnan(u_invd[s])
    eu_invd = np.sqrt(np.mean((u_invd[s][mask_d] - u_ref[s][mask_d])**2))
    print(f"{'Inversa diferencial 3pt':<25s}  {'—':>12s}  {'—':>12s}  {eu_invd:12.4e}")

    mask_i = ~np.isnan(u_invi[s])
    eu_invi = np.sqrt(np.mean((u_invi[s][mask_i] - u_ref[s][mask_i])**2))
    print(f"{'Inversa integral':<25s}  {'—':>12s}  {'—':>12s}  {eu_invi:12.4e}")

    # --- Plot ---
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    ms = t_ref * 1000

    # Directas: ω
    axes[0, 0].plot(ms, omega_ref, 'k-', lw=1.5, label='RK4')
    axes[0, 0].plot(ms, omega_dd, 'b--', lw=1, label='Dif')
    axes[0, 0].plot(ms, omega_di, 'r:', lw=1, label='Int')
    axes[0, 0].set_ylabel('$\\omega$ [rad/s]')
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_title('Directas: $\\omega$')

    # Directas: i
    axes[0, 1].plot(ms, i_ref, 'k-', lw=1.5, label='RK4')
    axes[0, 1].plot(ms, i_dd, 'b--', lw=1, label='Dif')
    axes[0, 1].plot(ms, i_di, 'r:', lw=1, label='Int')
    axes[0, 1].set_ylabel('$i$ [A]')
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_title('Directas: $i$')

    # Errores directas
    axes[1, 0].plot(ms, omega_dd - omega_ref, 'b-', lw=0.5, label='Dif')
    axes[1, 0].plot(ms, omega_di - omega_ref, 'r-', lw=0.5, label='Int')
    axes[1, 0].set_ylabel('$\\omega$ error')
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(ms, i_dd - i_ref, 'b-', lw=0.5, label='Dif')
    axes[1, 1].plot(ms, i_di - i_ref, 'r-', lw=0.5, label='Int')
    axes[1, 1].set_ylabel('$i$ error')
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)

    # Inversas: u
    axes[2, 0].plot(ms, u_ref, 'k-', lw=2, label='True $u$')
    axes[2, 0].plot(ms, u_invd, 'b--', lw=1, label=f'Dif 3pt (RMSE={eu_invd:.2e})')
    axes[2, 0].plot(ms, u_invi, 'r:', lw=1, label=f'Int (RMSE={eu_invi:.2e})')
    axes[2, 0].set_ylabel('$u$ [V]')
    axes[2, 0].set_xlabel('Time [ms]')
    axes[2, 0].legend(fontsize=9)
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].set_title('Inversas: $\\hat{u}$')

    # Error inversas
    axes[2, 1].plot(ms, u_invd - u_ref, 'b-', lw=0.5, label='Dif 3pt')
    axes[2, 1].plot(ms, u_invi - u_ref, 'r-', lw=0.5, label='Int')
    axes[2, 1].set_ylabel('$u$ error [V]')
    axes[2, 1].set_xlabel('Time [ms]')
    axes[2, 1].legend(fontsize=9)
    axes[2, 1].grid(True, alpha=0.3)

    plt.suptitle(f'4 Formulaciones vs RK4 (sin ruido, T={T:.0e})', fontsize=13)
    plt.tight_layout()
    plt.savefig('/home/rodo/10Paper/results/test_4formulaciones.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved to results/test_4formulaciones.png")
    plt.close()

    return {
        'omega_dd': omega_dd, 'i_dd': i_dd,
        'omega_di': omega_di, 'i_di': i_di,
        'u_invd': u_invd, 'u_invi': u_invi,
    }


def run_noise_test(T=0.0001, t_final=0.2):
    """Compara inversas con ruido en ω e i."""
    np.random.seed(42)
    n = int(t_final / T)

    # Ground truth RK4
    u_func = lambda t: 12.0 if t > 0.005 else 0.0
    x0 = np.array([0.0, 0.0])
    t_rk4, states_rk4, inputs_rk4 = mm.simulate(
        (0, t_final), x0, u_func, dt_rk4=1e-6
    )
    step = int(T / 1e-6)
    idx = np.arange(0, len(t_rk4), step)[:n]
    t_ref = t_rk4[idx]
    omega_ref = states_rk4[idx, 0]
    i_ref = states_rk4[idx, 1]
    u_ref = inputs_rk4[idx]

    skip = int(0.01 / T)
    s = slice(skip, None)

    noise_levels = [0.01, 0.05, 0.1, 0.5]

    print(f"\n{'σ':>6s}  {'Inv.Dif 2pt':>12s}  {'Inv.Dif 3pt':>12s}  {'Inv.Integral':>12s}  {'Ratio 3pt/Int':>14s}")
    print("-" * 65)

    results = {}
    for sigma in noise_levels:
        np.random.seed(42)
        omega_noisy = omega_ref + np.random.normal(0, sigma, n)
        i_noisy = i_ref + np.random.normal(0, sigma, n)

        u_d2 = inverse_differential(omega_noisy, i_noisy, T, n_points=2)
        u_d3 = inverse_differential(omega_noisy, i_noisy, T, n_points=3)
        u_int = inverse_integral(omega_noisy, i_noisy, T)

        mask2 = ~np.isnan(u_d2[s])
        mask3 = ~np.isnan(u_d3[s])
        maski = ~np.isnan(u_int[s])

        rmse_d2 = np.sqrt(np.mean((u_d2[s][mask2] - u_ref[s][mask2])**2))
        rmse_d3 = np.sqrt(np.mean((u_d3[s][mask3] - u_ref[s][mask3])**2))
        rmse_int = np.sqrt(np.mean((u_int[s][maski] - u_ref[s][maski])**2))

        ratio = rmse_d3 / rmse_int if rmse_int > 0 else np.inf
        print(f"{sigma:6.2f}  {rmse_d2:12.4f}  {rmse_d3:12.4f}  {rmse_int:12.4f}  {ratio:14.1f}x")

        results[sigma] = {
            'u_d2': u_d2, 'u_d3': u_d3, 'u_int': u_int,
            'rmse_d2': rmse_d2, 'rmse_d3': rmse_d3, 'rmse_int': rmse_int,
        }

    # --- Plot: directa con ruido en u vs inversa con ruido en ω,i ---
    fig, axes = plt.subplots(len(noise_levels), 2, figsize=(14, 3.5*len(noise_levels)),
                              sharex=True)
    ms = t_ref * 1000

    for row, sigma in enumerate(noise_levels):
        r = results[sigma]

        # Col 0: Directa con u ruidosa → ω suavizado
        np.random.seed(42)
        u_noisy = u_ref + np.random.normal(0, sigma*100, n)  # σ_u escalado
        omega_dd_noisy, _ = direct_differential(
            u_noisy, omega_ref[0], omega_ref[1], i_ref[0], i_ref[1], T, n
        )
        omega_di_noisy, _ = direct_integral(
            u_noisy, omega_ref[0], None, i_ref[0], None, T, n
        )

        ax = axes[row, 0]
        ax.plot(ms, omega_ref, 'k-', lw=1.5, label='True $\\omega$')
        ax.plot(ms, omega_dd_noisy, 'b-', lw=0.3, alpha=0.5, label='Dif')
        ax.plot(ms, omega_di_noisy, 'r-', lw=0.3, alpha=0.5, label='Int')
        ax.set_ylabel('$\\omega$')
        ax.legend(fontsize=7, loc='right')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Directa: $u$ ruidosa ($\\sigma_u$={sigma*100:.0f}V)', fontsize=9)

        # Col 1: Inversa con ω,i ruidosos → u reconstruida
        ax = axes[row, 1]
        ax.plot(ms, u_ref, 'k-', lw=2, label='True $u$')
        ax.plot(ms, r['u_d3'], 'b-', lw=0.3, alpha=0.4,
                label=f'Dif 3pt ({r["rmse_d3"]:.2f}V)')
        ax.plot(ms, r['u_int'], 'r-', lw=0.8, alpha=0.7,
                label=f'Integral ({r["rmse_int"]:.2f}V)')
        ax.set_ylabel('$u$ [V]')
        ax.legend(fontsize=7, loc='right')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Inversa: $\\omega,i$ ruidosos ($\\sigma$={sigma})', fontsize=9)
        ax.set_ylim(-5, 20)

    axes[-1, 0].set_xlabel('Time [ms]')
    axes[-1, 1].set_xlabel('Time [ms]')

    plt.suptitle('Simetría: Directa suaviza ruido en u → Inversa integral suaviza ruido en ω,i',
                 fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('/home/rodo/10Paper/results/test_4form_noise.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved to results/test_4form_noise.png")
    plt.close()


if __name__ == '__main__':
    print("=== 4 Formulaciones vs RK4 (sin ruido) ===\n")
    run_test(T=0.0001, t_final=0.2)

    print("\n\n=== Test con ruido: Inversa diferencial vs integral ===")
    run_noise_test(T=0.0001, t_final=0.2)
