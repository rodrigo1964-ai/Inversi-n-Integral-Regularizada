# Benchmark de Configuraciones - Homotopy Regressors

## 📁 Archivos Generados

### Reportes y Documentación

| Archivo | Descripción | Líneas |
|---------|-------------|--------|
| **REPORTE_BENCHMARK.md** | Reporte técnico completo con análisis detallado | 472 |
| **RESUMEN_EJECUTIVO.txt** | Resumen ejecutivo visual con gráficos ASCII | 183 |
| **README_BENCHMARK.md** | Este archivo (índice de documentos) | - |

### Scripts de Benchmark

| Archivo | Descripción | Casos | Configs |
|---------|-------------|-------|---------|
| **benchmark_comparison.py** | Comparación 0i vs 1i (solo 3pt) | 5 | 4 |
| **benchmark_full.py** | Benchmark completo con todos los ejemplos | 8 | 4 |
| **benchmark_3pt_vs_4pt.py** | Benchmark 3pt vs 4pt (COMPLETO) | 5 | 8 |
| **tabla_iteracion.py** | Tablas comparativas reorganizadas | 5 | 8 |

### Archivos de Configuración

| Archivo | Descripción |
|---------|-------------|
| **setup.py** | Configuración del paquete Python |
| **venv/** | Entorno virtual con dependencias |

---

## 🚀 Guía de Uso Rápida

### 1. Ver el Resumen Ejecutivo

```bash
cat RESUMEN_EJECUTIVO.txt
```

**Contenido:** Hallazgos principales, tablas comparativas, recomendaciones

### 2. Leer el Reporte Completo

```bash
# En tu editor favorito
vim REPORTE_BENCHMARK.md
# o
code REPORTE_BENCHMARK.md
```

**Contenido:** Metodología, análisis detallado, fundamento teórico, apéndices

### 3. Ejecutar el Benchmark Completo

```bash
source venv/bin/activate
python benchmark_3pt_vs_4pt.py
```

**Output:** Tabla con 8 configuraciones × 5 ejemplos (40 pruebas)

### 4. Ver Comparación de Iteraciones

```bash
source venv/bin/activate
python tabla_iteracion.py
```

**Output:** Dos tablas (2 términos y 3 términos) comparando con/sin iteración

### 5. Ejecutar Todos los Ejemplos

```bash
source venv/bin/activate
python benchmark_full.py
```

**Output:** Tabla con 8 ejemplos (incluyendo B, C, fricción)

---

## 📊 Resultados Principales

### Configuración Óptima: **0i-2p-4pt**

- **0 iteraciones** (sin recalcular)
- **2 términos** (z₁ + z₂)
- **4 puntos** backward

### Mejoras vs Implementación Actual (1i-3p-3pt)

| Ejemplo | Error Actual | Error Óptimo | Mejora |
|---------|-------------|--------------|--------|
| Ej1 (y'+y²) | 4.41e-02 | 4.05e-04 | **109×** |
| Ej2 (y'+sin²y) | 2.19e-02 | 2.12e-04 | **103×** |
| Ej3 (y'+β) | 7.62e-04 | 7.64e-05 | **10×** |
| Ej5 (péndulo) | 1.62e-02 | 4.12e-04 | **39×** |
| EjA (complejo) | 6.34e-01 | 2.57e-02 | **25×** |
| **Media geométrica** | 2.18e-02 | 3.52e-04 | **62×** |

### Velocidad

- **3× más rápido** (1 evaluación de f vs 3)
- Sin iteraciones → comportamiento predecible
- Ideal para microcontroladores

---

## 🔬 Metodología

### Casos de Prueba

1. **Ej1:** `y' + y² = sin(5t)` - No linealidad cuadrática
2. **Ej2:** `y' + sin²(y) = sin(5t)` - No linealidad trigonométrica
3. **Ej3:** `y' + β(y) = sin(5t)` - Polinomio cúbico
4. **Ej5:** `y'' + 0.1y' + sin(y) = sin(3t)` - Péndulo amortiguado
5. **EjA:** `y'' + ay' + by'(y²-1) + cyy' + y = sin(t)` - Sistema complejo

### Configuraciones

| ID | Términos | Iteraciones | Puntos |
|----|----------|-------------|--------|
| 0i-2p-3pt | 2 | 0 | 3 |
| 0i-3p-3pt | 3 | 0 | 3 |
| 1i-2p-3pt | 2 | 1 | 3 |
| 1i-3p-3pt | 3 | 1 | 3 |
| 0i-2p-4pt | 2 | 0 | 4 |
| 0i-3p-4pt | 3 | 0 | 4 |
| 1i-2p-4pt | 2 | 1 | 4 |
| 1i-3p-4pt | 3 | 1 | 4 |

### Métrica

**Error máximo absoluto:** `max|y_solver - y_ref|`

**Solver de referencia:** RK45 (scipy) con tolerancia 1e-8

---

## 💡 Recomendaciones de Implementación

### 1. Modificar `solver.py`

Cambiar configuración por defecto a:

```python
def solve_order1(f, df, d2f, d3f, u, y0, y1, y2, T, n,
                 n_terms=2, n_iterations=0, n_points=4):
    """
    Recomendado: n_terms=2, n_iterations=0, n_points=4
    """
    # Implementación basada en benchmark_3pt_vs_4pt.py
```

### 2. Condiciones Iniciales

Con 4 puntos se necesitan **3 valores iniciales**: y₀, y₁, y₂

Opciones:
- Calcular y₂ con RK4 si solo se tienen y₀, y₁
- Usar Taylor de orden 2
- Pedir al usuario los 3 valores

### 3. Parámetros Configurables

Mantener flexibilidad para casos especiales:

```python
# Configuración por defecto (óptima)
y = solve_order1(f, df, d2f, d3f, u, y0, y1, y2, T, n)

# Casos stiff (experimentar)
y = solve_order1(f, df, d2f, d3f, u, y0, y1, y2, T, n, n_terms=3)

# Fallback a 3 puntos si y₂ no disponible
y = solve_order1(f, df, d2f, d3f, u, y0, y1, None, T, n, n_points=3)
```

---

## 📈 Análisis Detallado

### ¿Por qué 0 iteraciones es mejor?

**Sin iteración (0i):**
- Calcula z₁, z₂, z₃ con el MISMO g inicial
- Equivalente a Taylor completo de orden superior
- No introduce errores acumulativos

**Con iteración (1i):**
- Recalcula g después de cada término
- Errores de z₁ se propagan a z₂ y z₃
- Amplificación de errores numéricos

**Resultado:** 0i es **6-105× más preciso**

### ¿Por qué 4 puntos es mejor?

- **Orden de precisión:** O(h³) vs O(h²)
- **Menor error de truncamiento**
- **Mejor para funciones oscilatorias**
- **Costo marginal:** +1 muestra de historia

**Resultado:** 4pt es **5-39× más preciso**

### ¿Por qué 2 términos es suficiente?

- z₁: captura comportamiento lineal
- z₂: corrige curvatura (2do orden)
- z₃: aporta <3% mejora adicional

**Resultado:** 2 términos tiene **relación costo/beneficio óptima**

---

## 🔄 Reproducibilidad

### Requisitos

```bash
python >= 3.8
numpy >= 2.4.2
scipy >= 1.17.1
sympy >= 1.14.0
```

### Instalación

```bash
# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install numpy scipy sympy

# O instalar el paquete en modo desarrollo
pip install -e .
```

### Ejecutar Benchmarks

```bash
# Activar entorno
source venv/bin/activate

# Benchmark completo (RECOMENDADO)
python benchmark_3pt_vs_4pt.py

# Comparación iteraciones
python tabla_iteracion.py

# Todos los ejemplos
python benchmark_full.py
```

**Tiempo estimado:** ~60 segundos total

---

## 📞 Contacto

**Autor:** Rodolfo H. Rodrigo
**Institución:** UNSJ
**Proyecto:** homotopy_regressors
**Fecha:** 25 de Febrero, 2026

---

## 📝 Licencia

[Agregar licencia del proyecto]

---

## 🙏 Agradecimientos

- Método de Liao y teoría HAM (Homotopy Analysis Method)
- Scipy por los solvers de referencia
- [Otros agradecimientos]

---

**Última actualización:** 25/02/2026
