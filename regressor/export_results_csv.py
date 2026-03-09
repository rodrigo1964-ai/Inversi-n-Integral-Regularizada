"""
export_results_csv.py - Exporta resultados a CSV para análisis posterior

Author: Rodolfo H. Rodrigo - UNSJ
"""

import numpy as np
import csv
from test_regressor_vs_rk4 import (
    test_linear_1st_order,
    test_nonlinear_quadratic,
    test_nonlinear_trigonometric,
    test_nonlinear_cubic,
    test_harmonic_oscillator,
    test_damped_pendulum,
    test_duffing_oscillator,
    test_van_der_pol
)


def export_test_to_csv(test_func, filename):
    """
    Ejecuta un test y exporta los resultados a CSV
    """
    print(f"Exportando {filename}...")
    t, y_rk4, y_reg, error = test_func()

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['tiempo', 'y_rk4', 'y_regresor', 'error_absoluto'])

        for i in range(len(t)):
            writer.writerow([t[i], y_rk4[i], y_reg[i], error[i]])

    print(f"  ✓ Guardado: {filename}")
    print(f"    Puntos: {len(t)}")
    print(f"    Error máx: {np.max(error):.4e}")
    print()


def export_summary_to_csv(filename='resumen_comparacion.csv'):
    """
    Exporta un resumen de todos los tests
    """
    print(f"Generando resumen en {filename}...")

    tests = [
        ('Test 1: Lineal 1er Orden', test_linear_1st_order),
        ('Test 2: Cuadrática', test_nonlinear_quadratic),
        ('Test 3: Trigonométrica', test_nonlinear_trigonometric),
        ('Test 4: Cúbica', test_nonlinear_cubic),
        ('Test 5: Oscilador Armónico', test_harmonic_oscillator),
        ('Test 6: Péndulo Amortiguado', test_damped_pendulum),
        ('Test 7: Duffing', test_duffing_oscillator),
        ('Test 8: Van der Pol', test_van_der_pol),
    ]

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Test', 'Error_Maximo', 'Error_RMS', 'N_Puntos'])

        for name, func in tests:
            t, y_rk4, y_reg, error = func()
            max_err = np.max(error)
            rms_err = np.sqrt(np.mean(error**2))
            writer.writerow([name, max_err, rms_err, len(t)])

    print(f"  ✓ Resumen guardado: {filename}\n")


if __name__ == "__main__":
    print("\n" + "="*70)
    print(" EXPORTACIÓN DE RESULTADOS A CSV")
    print("="*70 + "\n")

    # Exportar cada test individual
    export_test_to_csv(test_linear_1st_order, 'test1_lineal_1er_orden.csv')
    export_test_to_csv(test_nonlinear_quadratic, 'test2_cuadratica.csv')
    export_test_to_csv(test_nonlinear_trigonometric, 'test3_trigonometrica.csv')
    export_test_to_csv(test_nonlinear_cubic, 'test4_cubica.csv')
    export_test_to_csv(test_harmonic_oscillator, 'test5_oscilador_armonico.csv')
    export_test_to_csv(test_damped_pendulum, 'test6_pendulo_amortiguado.csv')
    export_test_to_csv(test_duffing_oscillator, 'test7_duffing.csv')
    export_test_to_csv(test_van_der_pol, 'test8_van_der_pol.csv')

    # Exportar resumen
    export_summary_to_csv()

    print("="*70)
    print(" ✓ EXPORTACIÓN COMPLETA")
    print("="*70 + "\n")
