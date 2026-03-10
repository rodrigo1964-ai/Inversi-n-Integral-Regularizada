"""
Test directo: HFNN regressor vs RK4 para el motor DC.

Sistema 2x2, primer orden:
    ω' + (b·ω + N_load(ω) - K_t·i) / J = 0      → g1
    i' + (R·i + K_e·ω - u) / L(i) = 0             → g2

Excitación: u(t) conocida (step 12V).
Sin ruido. Compara regressor homotópico contra RK4.

Author: Rodolfo H. Rodrigo
"""

import sys
sys.path.insert(0, '/home/rodo/10Paper/regressor')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from solver_system import solve_system
import motor_model as mm


def build_motor_system():
    """
    Build the motor as a 2x2 first-order system for solve_system.

    Variables: q = [ω, i]
    Discretized with 3-point backward differences.

    Residual form (y' + f(y) = u convention):
        g1 = ωp + (b·ω + N_load(ω) - K_t·i) / J = 0
        g2 = ip + (R·i + K_e·ω - u) / L(i) = 0

    Note: g2 uses u as excitation (excitations[1][k]).
          g1 has no external excitation (excitations[0][k] = 0).
    """

    # --- Funciones residuo ---
    # Firma: F(x, y, z, w, xp, yp, zp, wp, xpp, ypp, zpp, wpp, t)
    #   x=ω, y=i, z=0, w=0
    #   xp=ωp, yp=ip, zp=0, wp=0

    def g1(x, y, z, w, xp, yp, zp, wp, xpp, ypp, zpp, wpp, t):
        omega, i = x, y
        omega_p = xp
        return omega_p + (mm.b * omega + mm.N_load(omega) - mm.K_t * i) / mm.J

    def g2(x, y, z, w, xp, yp, zp, wp, xpp, ypp, zpp, wpp, t):
        omega, i = x, y
        i_p = yp
        return i_p + (mm.R * i + mm.K_e * omega) / mm.L_sat(i)
        # Nota: el término -u/L(i) va como excitación

    funcs = [g1, g2]

    # --- Jacobiano analítico (con regla de la cadena) ---
    # dgi/dqj[k] = ∂gi/∂qj + ∂gi/∂qpj · 3/(2T)
    # T no se conoce acá, se pasa como argumento implícito...
    # Pero solve_system espera funciones que reciben los args completos.
    # El Jacobiano ya incluye la contribución de la derivada discreta.

    # Para g1 = ωp + (b·ω + c·ω² - K_t·i) / J
    # ∂g1/∂ω = (b + 2c·ω) / J,  ∂g1/∂ωp = 1
    # → dg1/dω[k] = (b + 2c·ω) / J + 1 · 3/(2T)

    # ∂g1/∂i = -K_t / J,  ∂g1/∂ip = 0
    # → dg1/di[k] = -K_t / J

    # Para g2 = ip + (R·i + K_e·ω) / L(i)
    # ∂g2/∂ω = K_e / L(i),  ∂g2/∂ωp = 0
    # → dg2/dω[k] = K_e / L(i)

    # ∂g2/∂i = [R·L(i) - (R·i + K_e·ω)·L'(i)] / L(i)²,  ∂g2/∂ip = 1
    # → dg2/di[k] = [R/L(i) - (R·i + K_e·ω)·L'(i)/L(i)²] + 3/(2T)

    # Nota: necesitamos T para construir el Jacobiano.
    # solve_system pasa los args padded. Pero T no está ahí.
    # Necesitamos que T esté disponible. Lo capturamos en un closure.

    return funcs, g1, g2


def build_jac_hess(T):
    """Build Jacobian and Hessian functions with T captured in closure."""

    c = mm.C_LOAD

    def j00(x, y, z, w, xp, yp, zp, wp, xpp, ypp, zpp, wpp, t):
        omega = x
        return (mm.b + 2*c*omega) / mm.J + 3/(2*T)

    def j01(x, y, z, w, xp, yp, zp, wp, xpp, ypp, zpp, wpp, t):
        return -mm.K_t / mm.J

    def j10(x, y, z, w, xp, yp, zp, wp, xpp, ypp, zpp, wpp, t):
        omega, i = x, y
        return mm.K_e / mm.L_sat(i)

    def j11(x, y, z, w, xp, yp, zp, wp, xpp, ypp, zpp, wpp, t):
        omega, i = x, y
        L_i = mm.L_sat(i)
        dL = mm.dL_di(i)
        return mm.R / L_i - (mm.R * i + mm.K_e * omega) * dL / L_i**2 + 3/(2*T)

    jac_funcs = [[j00, j01], [j10, j11]]

    # --- Hessiano ---
    # d²g1/dω²[k] = 2c/J (constante)
    # d²g1/dωdi, d²g1/di² = 0
    # d²g2/dω² = 0
    # d²g2/dωdi = -K_e·L'(i)/L(i)²
    # d²g2/di² = complejo, calcular numéricamente por ahora

    def h000(x, y, z, w, xp, yp, zp, wp, xpp, ypp, zpp, wpp, t):
        return 2*c / mm.J

    def h001(x, y, z, w, xp, yp, zp, wp, xpp, ypp, zpp, wpp, t):
        return 0.0

    def h010(x, y, z, w, xp, yp, zp, wp, xpp, ypp, zpp, wpp, t):
        return 0.0

    def h011(x, y, z, w, xp, yp, zp, wp, xpp, ypp, zpp, wpp, t):
        return 0.0

    def h100(x, y, z, w, xp, yp, zp, wp, xpp, ypp, zpp, wpp, t):
        return 0.0

    def h101(x, y, z, w, xp, yp, zp, wp, xpp, ypp, zpp, wpp, t):
        i = y
        L_i = mm.L_sat(i)
        dL = mm.dL_di(i)
        return -mm.K_e * dL / L_i**2

    def h110(x, y, z, w, xp, yp, zp, wp, xpp, ypp, zpp, wpp, t):
        # same as h101 by symmetry
        i = y
        L_i = mm.L_sat(i)
        dL = mm.dL_di(i)
        return -mm.K_e * dL / L_i**2

    def h111(x, y, z, w, xp, yp, zp, wp, xpp, ypp, zpp, wpp, t):
        omega, i = x, y
        L_i = mm.L_sat(i)
        dL = mm.dL_di(i)
        d2L = mm.d2L_di2(i)
        val_num = mm.R * i + mm.K_e * omega

        # d/di[R/L - val·L'/L²]
        # = -R·L'/L² - (R·L'/L² + val·(d2L·L² - 2L·L'²)/L⁴)
        # = -R·L'/L² - R·L'/L² - val·(d2L/L² - 2·L'²/L³)
        # = -2R·L'/L² - val·d2L/L² + 2·val·L'²/L³
        return (-2*mm.R*dL/L_i**2
                - val_num*d2L/L_i**2
                + 2*val_num*dL**2/L_i**3)

    hess_funcs = [
        [[h000, h001], [h010, h011]],
        [[h100, h101], [h110, h111]]
    ]

    return jac_funcs, hess_funcs


def run_test(T=0.0001, t_final=0.2):
    """Run HFNN regressor vs RK4 for DC motor step response."""

    n = int(t_final / T)

    # Excitación: step 12V at t=5ms, g1 sin excitación
    # Pero la excitación de g2 es u/L(i), no u directamente.
    # g2 = ip + (Ri + Ke·ω)/L(i) - u/L(i) = 0
    # Entonces excitations[1][k] = u_k / L(i_k)
    # Problema: L(i_k) depende de i_k que aún no se conoce.
    #
    # Alternativa: reformular g2 para que la excitación sea u directamente.
    # Multiplicar toda g2 por L(i):
    #   L(i)·ip + R·i + K_e·ω = u
    # Pero entonces g2 depende de ip de forma no lineal (L(i)·ip).
    #
    # Más simple: usar excitación constante y pasar u[k] directamente.
    # Redefinir g2 incluyendo -u/L(i) internamente? No, porque u varía.
    #
    # La forma correcta: la excitación debe ser tal que
    #   g2(...) - exc[k] = 0
    # Si g2 = ip + (Ri + Ke·ω)/L(i), necesitamos exc[k] = u[k]/L(i_k).
    # Pero i_k no se conoce a priori.
    #
    # Solución: reformular g2 multiplicando por L(i):
    #   G2 = L(i)·ip + Ri + Ke·ω - u = 0
    # Ahora exc2[k] = u[k] (independiente de i).
    # Pero G2 es no lineal en (i, ip) porque L(i)·ip es producto de variables.

    # Reformulamos:
    def g1(x, y, z, w, xp, yp, zp, wp, xpp, ypp, zpp, wpp, t):
        omega, i = x, y
        return xp + (mm.b * omega + mm.N_load(omega) - mm.K_t * i) / mm.J

    def g2(x, y, z, w, xp, yp, zp, wp, xpp, ypp, zpp, wpp, t):
        omega, i = x, y
        i_p = yp
        L_i = mm.L_sat(i)
        return L_i * i_p + mm.R * i + mm.K_e * omega  # excitation = u[k]

    funcs = [g1, g2]

    # Jacobiano para g2 reformulada: G2 = L(i)·ip + R·i + K_e·ω
    # dG2/dω[k] = K_e
    # dG2/di[k] = L'(i)·ip + R + L(i)·3/(2T)    (chain rule: ∂G2/∂i + ∂G2/∂ip · 3/(2T))
    #           = L'(i)·ip + R + L(i)·3/(2T)

    c = mm.C_LOAD

    def j00(x, y, z, w, xp, yp, zp, wp, xpp, ypp, zpp, wpp, t):
        omega = x
        return (mm.b + 2*c*omega) / mm.J + 3/(2*T)

    def j01(x, y, z, w, xp, yp, zp, wp, xpp, ypp, zpp, wpp, t):
        return -mm.K_t / mm.J

    def j10(x, y, z, w, xp, yp, zp, wp, xpp, ypp, zpp, wpp, t):
        return mm.K_e

    def j11(x, y, z, w, xp, yp, zp, wp, xpp, ypp, zpp, wpp, t):
        i = y
        i_p = yp
        L_i = mm.L_sat(i)
        dL = mm.dL_di(i)
        return dL * i_p + mm.R + L_i * 3/(2*T)

    jac_funcs = [[j00, j01], [j10, j11]]

    # Hessiano
    def h000(x, y, z, w, xp, yp, zp, wp, xpp, ypp, zpp, wpp, t):
        return 2*c / mm.J

    def h001(x, y, z, w, xp, yp, zp, wp, xpp, ypp, zpp, wpp, t):
        return 0.0

    def h010(x, y, z, w, xp, yp, zp, wp, xpp, ypp, zpp, wpp, t):
        return 0.0

    def h011(x, y, z, w, xp, yp, zp, wp, xpp, ypp, zpp, wpp, t):
        return 0.0

    def h100(x, y, z, w, xp, yp, zp, wp, xpp, ypp, zpp, wpp, t):
        return 0.0

    def h101(x, y, z, w, xp, yp, zp, wp, xpp, ypp, zpp, wpp, t):
        return 0.0

    def h110(x, y, z, w, xp, yp, zp, wp, xpp, ypp, zpp, wpp, t):
        return 0.0

    def h111(x, y, z, w, xp, yp, zp, wp, xpp, ypp, zpp, wpp, t):
        i = y
        i_p = yp
        d2L = mm.d2L_di2(i)
        dL = mm.dL_di(i)
        # d/di[dL·ip + R + L·3/(2T)] = d2L·ip + dL·3/(2T)
        # Pero ip depende de i[k] via chain rule: dip/di[k] = 3/(2T)
        # Full: d²G2/di[k]² = d2L·ip + dL·(3/(2T)) + dL·(3/(2T))
        #                    = d2L·ip + 2·dL·3/(2T)
        return d2L * i_p + 2 * dL * 3/(2*T)

    hess_funcs = [
        [[h000, h001], [h010, h011]],
        [[h100, h101], [h110, h111]]
    ]

    # --- Ground truth RK4 ---
    u_func = lambda t: 12.0 if t > 0.005 else 0.0
    x0 = np.array([0.0, 0.0])
    t_rk4, states_rk4, inputs_rk4 = mm.simulate(
        (0, t_final), x0, u_func, dt_rk4=1e-6
    )

    # Downsample RK4 to match T
    step = int(T / 1e-6)
    idx = np.arange(0, len(t_rk4), step)[:n]
    t_ref = t_rk4[idx]
    omega_ref = states_rk4[idx, 0]
    i_ref = states_rk4[idx, 1]
    u_ref = inputs_rk4[idx]

    # --- HFNN regressor ---
    exc_g1 = np.zeros(n)          # g1 no tiene excitación
    exc_g2 = u_ref.copy()         # g2 excitación = u[k]

    ic = [[omega_ref[0], omega_ref[1]],
          [i_ref[0], i_ref[1]]]

    omega_hfnn, i_hfnn = solve_system(
        funcs, jac_funcs, hess_funcs, None,
        [exc_g1, exc_g2], ic, T, n
    )

    # --- Errores ---
    err_omega = np.max(np.abs(omega_hfnn - omega_ref))
    err_i = np.max(np.abs(i_hfnn - i_ref))

    print(f"T = {T:.0e}, n = {n}")
    print(f"  ω max error: {err_omega:.4e}")
    print(f"  i max error: {err_i:.4e}")

    # --- Plot ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    ms = t_ref * 1000

    axes[0].plot(ms, omega_ref, 'b-', lw=1.5, label='RK4')
    axes[0].plot(ms, omega_hfnn, 'r--', lw=1, label='HFNN')
    axes[0].set_ylabel('$\\omega$ [rad/s]')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f'Direct HFNN vs RK4  (T={T:.0e}, max err $\\omega$={err_omega:.2e})')

    axes[1].plot(ms, i_ref, 'b-', lw=1.5, label='RK4')
    axes[1].plot(ms, i_hfnn, 'r--', lw=1, label='HFNN')
    axes[1].set_ylabel('$i$ [A]')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title(f'Current (max err i={err_i:.2e})')

    axes[2].plot(ms, omega_hfnn - omega_ref, 'b-', lw=0.5, label='$\\omega$ error')
    axes[2].plot(ms, i_hfnn - i_ref, 'r-', lw=0.5, label='$i$ error')
    axes[2].set_ylabel('Error')
    axes[2].set_xlabel('Time [ms]')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/rodo/10Paper/results/test_direct_hfnn.png', dpi=300, bbox_inches='tight')
    print("  Saved to results/test_direct_hfnn.png")
    plt.close()

    return err_omega, err_i


if __name__ == '__main__':
    print("=== HFNN Direct vs RK4 (sin ruido) ===\n")
    run_test(T=0.0001, t_final=0.2)
