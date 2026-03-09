"""
test_regressor_vs_rk4.py - Comparación exhaustiva entre Regresor Homotópico y RK4

Prueba el regresor numérico con diferentes ecuaciones diferenciales:
- Lineales (1er y 2do orden)
- No lineales (1er y 2do orden)

Compara contra RK4 clásico (4to orden Runge-Kutta).

Author: Rodolfo H. Rodrigo - UNSJ
"""

import numpy as np
from solver import solve_order1, solve_order2
import time

# Matplotlib es opcional
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Advertencia: matplotlib no disponible, no se generarán gráficos")


# ============================================================
# Implementación RK4 clásico
# ============================================================

def rk4_order1(f, y0, t, u_func):
    """
    Runge-Kutta 4to orden para y' = f(t, y, u)
    donde la ecuación original es: y' + g(y) = u(t)
    => y' = -g(y) + u(t) = f(t, y)

    Parameters
    ----------
    f : callable f(t, y) -> float
        Lado derecho de la ecuación y' = f(t,y)
    y0 : float
        Condición inicial
    t : array
        Vector de tiempos
    u_func : callable (no usado en RK4, pero mantiene interfaz)

    Returns
    -------
    y : array
        Solución
    """
    n = len(t)
    y = np.zeros(n)
    y[0] = y0

    for i in range(n-1):
        h = t[i+1] - t[i]
        ti = t[i]
        yi = y[i]

        k1 = f(ti, yi)
        k2 = f(ti + h/2, yi + h*k1/2)
        k3 = f(ti + h/2, yi + h*k2/2)
        k4 = f(ti + h, yi + h*k3)

        y[i+1] = yi + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

    return y


def rk4_order2(f, y0, yp0, t, u_func):
    """
    Runge-Kutta 4to orden para sistema de 2 ecuaciones:
    y' = yp
    yp' = f(t, y, yp)

    Donde la ecuación original es: y'' + g(y, y') = u(t)
    => y'' = -g(y, y') + u(t)

    Parameters
    ----------
    f : callable f(t, y, yp) -> float
        Lado derecho de y'' = f(t, y, y')
    y0 : float
        Condición inicial y(0)
    yp0 : float
        Condición inicial y'(0)
    t : array
        Vector de tiempos
    u_func : callable u(t) -> float (usado en f)

    Returns
    -------
    y : array
        Solución y(t)
    """
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

        # k's para y' = yp
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


# ============================================================
# Tests de Ecuaciones de 1er Orden
# ============================================================

def test_linear_1st_order():
    """
    Test 1: Ecuación lineal 1er orden
    y' + 2y = sin(5t)
    Solución analítica conocida para verificar ambos métodos
    """
    print("=" * 70)
    print("TEST 1: Ecuación Lineal 1er Orden")
    print("y' + 2y = sin(5t)")
    print("=" * 70)

    n = 500
    t = np.linspace(0, 5, n)
    T = t[1] - t[0]
    u = np.sin(5*t)

    # Definir f(y) = 2y
    f    = lambda y: 2*y
    df   = lambda y: 2.0
    d2f  = lambda y: 0.0
    d3f  = lambda y: 0.0

    # RK4: y' = -2y + sin(5t)
    y_rk4 = rk4_order1(lambda t, y: -2*y + np.sin(5*t), 0.0, t, u)

    # Regresor homotópico
    t0 = time.time()
    y_reg = solve_order1(f, df, d2f, d3f, u, y_rk4[0], y_rk4[1], T, n)
    t_reg = time.time() - t0

    # Calcular error
    error = np.abs(y_reg - y_rk4)
    max_error = np.max(error)
    rms_error = np.sqrt(np.mean(error**2))

    print(f"Error máximo:  {max_error:.4e}")
    print(f"Error RMS:     {rms_error:.4e}")
    print(f"Tiempo regresor: {t_reg*1000:.2f} ms")
    print()

    return t, y_rk4, y_reg, error


def test_nonlinear_quadratic():
    """
    Test 2: Ecuación no lineal cuadrática
    y' + y² = sin(5t)
    """
    print("=" * 70)
    print("TEST 2: Ecuación No Lineal - Cuadrática")
    print("y' + y² = sin(5t)")
    print("=" * 70)

    n = 500
    t = np.linspace(0, 10, n)
    T = t[1] - t[0]
    u = np.sin(5*t)

    # f(y) = y²
    f    = lambda y: y**2
    df   = lambda y: 2*y
    d2f  = lambda y: 2.0
    d3f  = lambda y: 0.0

    # RK4: y' = -y² + sin(5t)
    y_rk4 = rk4_order1(lambda t, y: -y**2 + np.sin(5*t), -0.2, t, u)

    # Regresor homotópico
    t0 = time.time()
    y_reg = solve_order1(f, df, d2f, d3f, u, y_rk4[0], y_rk4[1], T, n)
    t_reg = time.time() - t0

    error = np.abs(y_reg - y_rk4)
    max_error = np.max(error)
    rms_error = np.sqrt(np.mean(error**2))

    print(f"Error máximo:  {max_error:.4e}")
    print(f"Error RMS:     {rms_error:.4e}")
    print(f"Tiempo regresor: {t_reg*1000:.2f} ms")
    print()

    return t, y_rk4, y_reg, error


def test_nonlinear_trigonometric():
    """
    Test 3: Ecuación no lineal trigonométrica
    y' + sin²(y) = sin(5t)
    """
    print("=" * 70)
    print("TEST 3: Ecuación No Lineal - Trigonométrica")
    print("y' + sin²(y) = sin(5t)")
    print("=" * 70)

    n = 500
    t = np.linspace(0, 10, n)
    T = t[1] - t[0]
    u = np.sin(5*t)

    # f(y) = sin²(y)
    f    = lambda y: np.sin(y)**2
    df   = lambda y: 2*np.sin(y)*np.cos(y)
    d2f  = lambda y: 2*np.cos(y)**2 - 2*np.sin(y)**2
    d3f  = lambda y: -8*np.sin(y)*np.cos(y)

    # RK4
    y_rk4 = rk4_order1(lambda t, y: -np.sin(y)**2 + np.sin(5*t), -0.2, t, u)

    # Regresor
    t0 = time.time()
    y_reg = solve_order1(f, df, d2f, d3f, u, y_rk4[0], y_rk4[1], T, n)
    t_reg = time.time() - t0

    error = np.abs(y_reg - y_rk4)
    max_error = np.max(error)
    rms_error = np.sqrt(np.mean(error**2))

    print(f"Error máximo:  {max_error:.4e}")
    print(f"Error RMS:     {rms_error:.4e}")
    print(f"Tiempo regresor: {t_reg*1000:.2f} ms")
    print()

    return t, y_rk4, y_reg, error


def test_nonlinear_cubic():
    """
    Test 4: Ecuación no lineal cúbica
    y' + β(y) = sin(5t)
    β(y) = -y³/10 + y²/10 + y - 1
    """
    print("=" * 70)
    print("TEST 4: Ecuación No Lineal - Cúbica")
    print("y' + (-y³/10 + y²/10 + y - 1) = sin(5t)")
    print("=" * 70)

    n = 500
    t = np.linspace(0, 5, n)
    T = t[1] - t[0]
    u = np.sin(5*t)

    # β(y) = -y³/10 + y²/10 + y - 1
    def beta(y): return -1/10*y**3 + 1/10*y**2 + y - 1
    def db(y):   return -3/10*y**2 + 2/10*y + 1
    def db2(y):  return -6/10*y + 2/10
    def db3(y):  return -6/10

    # RK4
    y_rk4 = rk4_order1(lambda t, y: -beta(y) + np.sin(5*t), -0.2, t, u)

    # Regresor
    t0 = time.time()
    y_reg = solve_order1(beta, db, db2, db3, u, y_rk4[0], y_rk4[1], T, n)
    t_reg = time.time() - t0

    error = np.abs(y_reg - y_rk4)
    max_error = np.max(error)
    rms_error = np.sqrt(np.mean(error**2))

    print(f"Error máximo:  {max_error:.4e}")
    print(f"Error RMS:     {rms_error:.4e}")
    print(f"Tiempo regresor: {t_reg*1000:.2f} ms")
    print()

    return t, y_rk4, y_reg, error


# ============================================================
# Tests de Ecuaciones de 2do Orden
# ============================================================

def test_harmonic_oscillator():
    """
    Test 5: Oscilador armónico simple (lineal)
    y'' + ω²y = A·sin(νt)
    ω = 2, A = 1, ν = 3
    """
    print("=" * 70)
    print("TEST 5: Oscilador Armónico Simple (Lineal)")
    print("y'' + 4y = sin(3t)")
    print("=" * 70)

    omega = 2.0
    n = 1000
    t = np.linspace(0, 10, n)
    T = t[1] - t[0]
    u = np.sin(3*t)

    # f(y, y') = ω²y
    f          = lambda y, yp: omega**2 * y
    df_dy      = lambda y, yp: omega**2
    df_dyp     = lambda y, yp: 0.0
    d2f_dy2    = lambda y, yp: 0.0
    d2f_dydyp  = lambda y, yp: 0.0
    d2f_dyp2   = lambda y, yp: 0.0
    d3f_dy3    = lambda y, yp: 0.0

    # RK4: y'' = -ω²y + sin(3t)
    y_rk4 = rk4_order2(lambda t, y, yp: -omega**2*y + np.sin(3*t),
                       0.0, 0.0, t, u)

    # Regresor
    t0 = time.time()
    y_reg = solve_order2(f, df_dy, df_dyp, d2f_dy2, d2f_dydyp, d2f_dyp2,
                         d3f_dy3, u, y_rk4[0], y_rk4[1], T, n)
    t_reg = time.time() - t0

    error = np.abs(y_reg - y_rk4)
    max_error = np.max(error)
    rms_error = np.sqrt(np.mean(error**2))

    print(f"Error máximo:  {max_error:.4e}")
    print(f"Error RMS:     {rms_error:.4e}")
    print(f"Tiempo regresor: {t_reg*1000:.2f} ms")
    print()

    return t, y_rk4, y_reg, error


def test_damped_pendulum():
    """
    Test 6: Péndulo amortiguado (no lineal)
    y'' + μy' + sin(y) = sin(3t)
    μ = 0.1
    """
    print("=" * 70)
    print("TEST 6: Péndulo Amortiguado (No Lineal)")
    print("y'' + 0.1y' + sin(y) = sin(3t)")
    print("=" * 70)

    mu = 0.1
    n = 1000
    t = np.linspace(0, 20, n)
    T = t[1] - t[0]
    u = np.sin(3*t)

    # f(y, y') = μy' + sin(y)
    f          = lambda y, yp: mu*yp + np.sin(y)
    df_dy      = lambda y, yp: np.cos(y)
    df_dyp     = lambda y, yp: mu
    d2f_dy2    = lambda y, yp: -np.sin(y)
    d2f_dydyp  = lambda y, yp: 0.0
    d2f_dyp2   = lambda y, yp: 0.0
    d3f_dy3    = lambda y, yp: -np.cos(y)

    # RK4
    y_rk4 = rk4_order2(lambda t, y, yp: -mu*yp - np.sin(y) + np.sin(3*t),
                       0.5, 0.0, t, u)

    # Regresor
    t0 = time.time()
    y_reg = solve_order2(f, df_dy, df_dyp, d2f_dy2, d2f_dydyp, d2f_dyp2,
                         d3f_dy3, u, y_rk4[0], y_rk4[1], T, n)
    t_reg = time.time() - t0

    error = np.abs(y_reg - y_rk4)
    max_error = np.max(error)
    rms_error = np.sqrt(np.mean(error**2))

    print(f"Error máximo:  {max_error:.4e}")
    print(f"Error RMS:     {rms_error:.4e}")
    print(f"Tiempo regresor: {t_reg*1000:.2f} ms")
    print()

    return t, y_rk4, y_reg, error


def test_duffing_oscillator():
    """
    Test 7: Oscilador de Duffing (no lineal)
    y'' + δy' + αy + βy³ = γ·cos(ωt)
    δ=0.1, α=1, β=0.2, γ=0.3, ω=1.2
    """
    print("=" * 70)
    print("TEST 7: Oscilador de Duffing (No Lineal)")
    print("y'' + 0.1y' + y + 0.2y³ = 0.3·cos(1.2t)")
    print("=" * 70)

    delta, alpha, beta = 0.1, 1.0, 0.2
    gamma, omega = 0.3, 1.2

    n = 2000
    t = np.linspace(0, 50, n)
    T = t[1] - t[0]
    u = gamma * np.cos(omega*t)

    # f(y, y') = δy' + αy + βy³
    f          = lambda y, yp: delta*yp + alpha*y + beta*y**3
    df_dy      = lambda y, yp: alpha + 3*beta*y**2
    df_dyp     = lambda y, yp: delta
    d2f_dy2    = lambda y, yp: 6*beta*y
    d2f_dydyp  = lambda y, yp: 0.0
    d2f_dyp2   = lambda y, yp: 0.0
    d3f_dy3    = lambda y, yp: 6*beta

    # RK4
    y_rk4 = rk4_order2(
        lambda t, y, yp: -delta*yp - alpha*y - beta*y**3 + gamma*np.cos(omega*t),
        0.1, 0.0, t, u)

    # Regresor
    t0 = time.time()
    y_reg = solve_order2(f, df_dy, df_dyp, d2f_dy2, d2f_dydyp, d2f_dyp2,
                         d3f_dy3, u, y_rk4[0], y_rk4[1], T, n)
    t_reg = time.time() - t0

    error = np.abs(y_reg - y_rk4)
    max_error = np.max(error)
    rms_error = np.sqrt(np.mean(error**2))

    print(f"Error máximo:  {max_error:.4e}")
    print(f"Error RMS:     {rms_error:.4e}")
    print(f"Tiempo regresor: {t_reg*1000:.2f} ms")
    print()

    return t, y_rk4, y_reg, error


def test_van_der_pol():
    """
    Test 8: Oscilador de Van der Pol (no lineal)
    y'' - μ(1-y²)y' + y = A·sin(ωt)
    μ = 1.0, A = 0.5, ω = 1.5
    """
    print("=" * 70)
    print("TEST 8: Oscilador de Van der Pol (No Lineal)")
    print("y'' - 1.0(1-y²)y' + y = 0.5·sin(1.5t)")
    print("=" * 70)

    mu = 1.0
    A, omega = 0.5, 1.5

    n = 2000
    t = np.linspace(0, 30, n)
    T = t[1] - t[0]
    u = A * np.sin(omega*t)

    # Reorganizar: y'' + [-μ(1-y²)y' + y] = u
    # f(y, y') = -μ(1-y²)y' + y
    f          = lambda y, yp: -mu*(1-y**2)*yp + y
    df_dy      = lambda y, yp: 2*mu*y*yp + 1
    df_dyp     = lambda y, yp: -mu*(1-y**2)
    d2f_dy2    = lambda y, yp: 2*mu*yp
    d2f_dydyp  = lambda y, yp: 2*mu*y
    d2f_dyp2   = lambda y, yp: 0.0
    d3f_dy3    = lambda y, yp: 0.0

    # RK4: y'' = μ(1-y²)y' - y + u
    y_rk4 = rk4_order2(
        lambda t, y, yp: mu*(1-y**2)*yp - y + A*np.sin(omega*t),
        0.1, 0.0, t, u)

    # Regresor
    t0 = time.time()
    y_reg = solve_order2(f, df_dy, df_dyp, d2f_dy2, d2f_dydyp, d2f_dyp2,
                         d3f_dy3, u, y_rk4[0], y_rk4[1], T, n)
    t_reg = time.time() - t0

    error = np.abs(y_reg - y_rk4)
    max_error = np.max(error)
    rms_error = np.sqrt(np.mean(error**2))

    print(f"Error máximo:  {max_error:.4e}")
    print(f"Error RMS:     {rms_error:.4e}")
    print(f"Tiempo regresor: {t_reg*1000:.2f} ms")
    print()

    return t, y_rk4, y_reg, error


# ============================================================
# Visualización
# ============================================================

def plot_comparison(results, filename='comparison_regressor_vs_rk4.png'):
    """
    Grafica todos los resultados
    """
    if not HAS_MATPLOTLIB:
        print("\nMatplotlib no disponible, saltando generación de gráficos")
        return

    n_tests = len(results)
    fig, axes = plt.subplots(n_tests, 2, figsize=(14, 4*n_tests))

    if n_tests == 1:
        axes = axes.reshape(1, -1)

    for i, (name, data) in enumerate(results.items()):
        t, y_rk4, y_reg, error = data

        # Gráfico de soluciones
        axes[i, 0].plot(t, y_rk4, 'b-', label='RK4', linewidth=2)
        axes[i, 0].plot(t, y_reg, 'r--', label='Regresor', linewidth=1.5)
        axes[i, 0].set_xlabel('Tiempo [s]')
        axes[i, 0].set_ylabel('y(t)')
        axes[i, 0].set_title(f'{name} - Soluciones')
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)

        # Gráfico de error
        axes[i, 1].plot(t, error, 'k-', linewidth=1)
        axes[i, 1].set_xlabel('Tiempo [s]')
        axes[i, 1].set_ylabel('Error absoluto')
        axes[i, 1].set_title(f'{name} - Error')
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].set_yscale('log')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nGráfico guardado en: {filename}")
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    """
    Ejecuta todos los tests
    """
    print("\n" + "="*70)
    print(" COMPARACIÓN: REGRESOR HOMOTÓPICO vs RK4 CLÁSICO")
    print("="*70 + "\n")

    results = {}

    # Tests de 1er orden
    print("\n" + "="*70)
    print(" ECUACIONES DE PRIMER ORDEN")
    print("="*70 + "\n")

    results['Test 1: Lineal 1er Orden'] = test_linear_1st_order()
    results['Test 2: Cuadrática'] = test_nonlinear_quadratic()
    results['Test 3: Trigonométrica'] = test_nonlinear_trigonometric()
    results['Test 4: Cúbica'] = test_nonlinear_cubic()

    # Tests de 2do orden
    print("\n" + "="*70)
    print(" ECUACIONES DE SEGUNDO ORDEN")
    print("="*70 + "\n")

    results['Test 5: Oscilador Armónico'] = test_harmonic_oscillator()
    results['Test 6: Péndulo Amortiguado'] = test_damped_pendulum()
    results['Test 7: Duffing'] = test_duffing_oscillator()
    results['Test 8: Van der Pol'] = test_van_der_pol()

    # Resumen
    print("\n" + "="*70)
    print(" RESUMEN DE RESULTADOS")
    print("="*70)
    print(f"{'Test':<35} {'Error Máx':>15} {'Error RMS':>15}")
    print("-"*70)

    for name, (t, y_rk4, y_reg, error) in results.items():
        max_err = np.max(error)
        rms_err = np.sqrt(np.mean(error**2))
        print(f"{name:<35} {max_err:>15.4e} {rms_err:>15.4e}")

    print("="*70 + "\n")

    # Graficar
    plot_comparison(results)

    return results


if __name__ == "__main__":
    results = main()
