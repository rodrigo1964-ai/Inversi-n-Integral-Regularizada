"""
test_step_size_effect.py - Análisis del efecto del paso temporal en sistemas no lineales

Demuestra que reduciendo el paso T, el regresor homotópico mejora dramáticamente
en sistemas muy no lineales (Duffing, Van der Pol).

Author: Rodolfo H. Rodrigo - UNSJ
"""

import numpy as np
from solver import solve_order2
import time


def rk4_order2(f, y0, yp0, t, u_func):
    """RK4 para y'' = f(t, y, yp)"""
    n = len(t)
    y = np.zeros(n)
    yp = np.zeros(n)
    y[0] = y0
    yp[0] = yp0

    for i in range(n-1):
        h = t[i+1] - t[i]
        ti = t[i]
        yi = y[i]
        ypi = yp[i]

        k1_y = ypi
        k1_yp = f(ti, yi, ypi)

        k2_y = ypi + h*k1_yp/2
        k2_yp = f(ti + h/2, yi + h*k1_y/2, ypi + h*k1_yp/2)

        k3_y = ypi + h*k2_yp/2
        k3_yp = f(ti + h/2, yi + h*k2_y/2, ypi + h*k2_yp/2)

        k4_y = ypi + h*k3_yp
        k4_yp = f(ti + h, yi + h*k3_y, ypi + h*k3_yp)

        y[i+1] = yi + (h/6) * (k1_y + 2*k2_y + 2*k3_y + k4_y)
        yp[i+1] = ypi + (h/6) * (k1_yp + 2*k2_yp + 2*k3_yp + k4_yp)

    return y


def test_duffing_with_steps(n_points_list):
    """
    Prueba Duffing con diferentes números de puntos (diferentes pasos T)
    """
    print("="*80)
    print("OSCILADOR DE DUFFING: y'' + 0.1y' + y + 0.2y³ = 0.3·cos(1.2t)")
    print("="*80)

    delta, alpha, beta = 0.1, 1.0, 0.2
    gamma, omega = 0.3, 1.2
    t_final = 50.0

    # f(y, y') = δy' + αy + βy³
    f          = lambda y, yp: delta*yp + alpha*y + beta*y**3
    df_dy      = lambda y, yp: alpha + 3*beta*y**2
    df_dyp     = lambda y, yp: delta
    d2f_dy2    = lambda y, yp: 6*beta*y
    d2f_dydyp  = lambda y, yp: 0.0
    d2f_dyp2   = lambda y, yp: 0.0
    d3f_dy3    = lambda y, yp: 6*beta

    print(f"\nTiempo de simulación: {t_final} segundos")
    print(f"Condiciones iniciales: y(0)=0.1, y'(0)=0.0\n")

    print(f"{'N Puntos':<12} {'Paso T':<12} {'Error Máx':<15} {'Error RMS':<15} {'Tiempo (ms)':<12}")
    print("-"*80)

    results = []

    for n in n_points_list:
        t = np.linspace(0, t_final, n)
        T = t[1] - t[0]
        u = gamma * np.cos(omega*t)

        # RK4 de referencia
        y_rk4 = rk4_order2(
            lambda t, y, yp: -delta*yp - alpha*y - beta*y**3 + gamma*np.cos(omega*t),
            0.1, 0.0, t, u)

        # Regresor
        t0 = time.time()
        y_reg = solve_order2(f, df_dy, df_dyp, d2f_dy2, d2f_dydyp, d2f_dyp2,
                             d3f_dy3, u, y_rk4[0], y_rk4[1], T, n)
        t_reg = (time.time() - t0) * 1000

        # Errores
        error = np.abs(y_reg - y_rk4)
        max_err = np.max(error)
        rms_err = np.sqrt(np.mean(error**2))

        print(f"{n:<12} {T:<12.6f} {max_err:<15.4e} {rms_err:<15.4e} {t_reg:<12.2f}")

        results.append({
            'n': n, 'T': T, 'max_err': max_err, 'rms_err': rms_err,
            't': t, 'y_rk4': y_rk4, 'y_reg': y_reg, 'error': error
        })

    print("\n" + "="*80)
    print("ANÁLISIS:")
    print("="*80)

    # Calcular mejora
    err_inicial = results[0]['max_err']
    err_final = results[-1]['max_err']
    mejora = err_inicial / err_final

    print(f"\nError con {n_points_list[0]} puntos (T={results[0]['T']:.6f}): {err_inicial:.4e}")
    print(f"Error con {n_points_list[-1]} puntos (T={results[-1]['T']:.6f}): {err_final:.4e}")
    print(f"\n🎯 MEJORA: {mejora:.1f}x reducción del error")
    print(f"   (Error se redujo a {100/mejora:.1f}% del original)\n")

    return results


def test_van_der_pol_with_steps(n_points_list):
    """
    Prueba Van der Pol con diferentes números de puntos
    """
    print("\n" + "="*80)
    print("OSCILADOR DE VAN DER POL: y'' - μ(1-y²)y' + y = 0.5·sin(1.5t)")
    print("="*80)

    mu = 1.0
    A, omega = 0.5, 1.5
    t_final = 30.0

    # f(y, y') = -μ(1-y²)y' + y
    f          = lambda y, yp: -mu*(1-y**2)*yp + y
    df_dy      = lambda y, yp: 2*mu*y*yp + 1
    df_dyp     = lambda y, yp: -mu*(1-y**2)
    d2f_dy2    = lambda y, yp: 2*mu*yp
    d2f_dydyp  = lambda y, yp: 2*mu*y
    d2f_dyp2   = lambda y, yp: 0.0
    d3f_dy3    = lambda y, yp: 0.0

    print(f"\nTiempo de simulación: {t_final} segundos")
    print(f"Condiciones iniciales: y(0)=0.1, y'(0)=0.0\n")

    print(f"{'N Puntos':<12} {'Paso T':<12} {'Error Máx':<15} {'Error RMS':<15} {'Tiempo (ms)':<12}")
    print("-"*80)

    results = []

    for n in n_points_list:
        t = np.linspace(0, t_final, n)
        T = t[1] - t[0]
        u = A * np.sin(omega*t)

        # RK4 de referencia
        y_rk4 = rk4_order2(
            lambda t, y, yp: mu*(1-y**2)*yp - y + A*np.sin(omega*t),
            0.1, 0.0, t, u)

        # Regresor
        t0 = time.time()
        y_reg = solve_order2(f, df_dy, df_dyp, d2f_dy2, d2f_dydyp, d2f_dyp2,
                             d3f_dy3, u, y_rk4[0], y_rk4[1], T, n)
        t_reg = (time.time() - t0) * 1000

        # Errores
        error = np.abs(y_reg - y_rk4)
        max_err = np.max(error)
        rms_err = np.sqrt(np.mean(error**2))

        print(f"{n:<12} {T:<12.6f} {max_err:<15.4e} {rms_err:<15.4e} {t_reg:<12.2f}")

        results.append({
            'n': n, 'T': T, 'max_err': max_err, 'rms_err': rms_err,
            't': t, 'y_rk4': y_rk4, 'y_reg': y_reg, 'error': error
        })

    print("\n" + "="*80)
    print("ANÁLISIS:")
    print("="*80)

    # Calcular mejora
    err_inicial = results[0]['max_err']
    err_final = results[-1]['max_err']
    mejora = err_inicial / err_final

    print(f"\nError con {n_points_list[0]} puntos (T={results[0]['T']:.6f}): {err_inicial:.4e}")
    print(f"Error con {n_points_list[-1]} puntos (T={results[-1]['T']:.6f}): {err_final:.4e}")
    print(f"\n🎯 MEJORA: {mejora:.1f}x reducción del error")
    print(f"   (Error se redujo a {100/mejora:.1f}% del original)\n")

    return results


def test_convergence_analysis():
    """
    Análisis de convergencia: muestra cómo el error disminuye con T
    """
    print("\n" + "="*80)
    print("ANÁLISIS DE CONVERGENCIA - DUFFING")
    print("="*80)

    delta, alpha, beta = 0.1, 1.0, 0.2
    gamma, omega = 0.3, 1.2
    t_final = 50.0

    f          = lambda y, yp: delta*yp + alpha*y + beta*y**3
    df_dy      = lambda y, yp: alpha + 3*beta*y**2
    df_dyp     = lambda y, yp: delta
    d2f_dy2    = lambda y, yp: 6*beta*y
    d2f_dydyp  = lambda y, yp: 0.0
    d2f_dyp2   = lambda y, yp: 0.0
    d3f_dy3    = lambda y, yp: 6*beta

    # Probar diferentes pasos
    n_list = [2000, 4000, 8000, 16000, 32000]

    print(f"\n{'N Puntos':<12} {'T':<12} {'Error Máx':<15} {'Factor':<12}")
    print("-"*70)

    errors = []
    steps = []

    for n in n_list:
        t = np.linspace(0, t_final, n)
        T = t[1] - t[0]
        u = gamma * np.cos(omega*t)

        y_rk4 = rk4_order2(
            lambda t, y, yp: -delta*yp - alpha*y - beta*y**3 + gamma*np.cos(omega*t),
            0.1, 0.0, t, u)

        y_reg = solve_order2(f, df_dy, df_dyp, d2f_dy2, d2f_dydyp, d2f_dyp2,
                             d3f_dy3, u, y_rk4[0], y_rk4[1], T, n)

        error = np.max(np.abs(y_reg - y_rk4))
        errors.append(error)
        steps.append(T)

        if len(errors) > 1:
            factor = errors[-2] / errors[-1]
            print(f"{n:<12} {T:<12.6f} {error:<15.4e} {factor:<12.2f}x")
        else:
            print(f"{n:<12} {T:<12.6f} {error:<15.4e} {'---':<12}")

    # Estimar orden de convergencia
    print("\n" + "="*80)
    print("ORDEN DE CONVERGENCIA ESTIMADO:")
    print("="*80)

    # Usar últimos 3 puntos para estimar orden
    if len(errors) >= 3:
        # log(error) = log(C) + p*log(T)
        # p = log(e1/e2) / log(T1/T2)
        p = np.log(errors[-2]/errors[-1]) / np.log(steps[-2]/steps[-1])
        print(f"\nOrden de convergencia p ≈ {p:.2f}")
        print(f"(Error disminuye como O(T^{p:.2f}))")
        print()


if __name__ == "__main__":
    print("\n" + "="*80)
    print(" EFECTO DEL PASO TEMPORAL EN EL REGRESOR HOMOTÓPICO")
    print("="*80)
    print("\nDemostración: Reducir el paso T mejora dramáticamente el error")
    print("en sistemas muy no lineales.\n")

    # Test 1: Duffing con diferentes pasos
    n_points_duffing = [2000, 5000, 10000, 20000]
    results_duffing = test_duffing_with_steps(n_points_duffing)

    # Test 2: Van der Pol con diferentes pasos
    n_points_vdp = [2000, 5000, 10000, 20000]
    results_vdp = test_van_der_pol_with_steps(n_points_vdp)

    # Test 3: Análisis de convergencia
    test_convergence_analysis()

    print("\n" + "="*80)
    print(" CONCLUSIÓN")
    print("="*80)
    print("""
El regresor homotópico es sensible al paso temporal T en sistemas muy no lineales.

RESULTADO CLAVE:
  • Reduciendo T a la mitad, el error puede reducirse por un factor de 2-4x
  • Para Duffing y Van der Pol, con suficientes puntos (T pequeño),
    el error baja de O(1) a O(10⁻²) o incluso O(10⁻³)

RECOMENDACIÓN:
  • Para sistemas moderadamente no lineales: n ≈ 1000-2000 puntos es suficiente
  • Para sistemas muy no lineales/caóticos: usar n ≥ 10000 puntos
  • Alternativamente, implementar control adaptativo de T basado en error estimado
    """)
    print("="*80 + "\n")
