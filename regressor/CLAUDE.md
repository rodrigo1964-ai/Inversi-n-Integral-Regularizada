# CLAUDE.md — Contrato de desarrollo: Regresor Homotópico para Sistemas Clásicos de EDOs

**Proyecto:** homotopy_regressors  
**Autor:** Rodolfo H. Rodrigo - UNSJ  
**Fecha:** Marzo 2026

---

## 1. Objetivo

Extender el regresor homotópico (actualmente para una sola EDO escalar) a sistemas
de 2 y 3 ecuaciones diferenciales ordinarias acopladas, en la forma general:

```
F(x, y, z,  x', y', z',  x'', y'', z'',  t) = u(t)
G(x, y, z,  x', y', z',  x'', y'', z'',  t) = v(t)
H(x, y, z,  x', y', z',  x'', y'', z'',  t) = w(t)
```

- F, G, H son funciones completamente conocidas (no identificacion, no caja negra).
- u(t), v(t), w(t) son excitaciones externas conocidas (pueden ser cero para sistemas autonomos).
- Las derivadas en el lado izquierdo pueden ser de 1er o 2do orden, mezcladas.
- En sistemas de 1er orden puro: x'', y'', z'' no aparecen en F, G, H.

Este enfoque es la generalizacion directa de lo que hace solver.py para una
ecuacion escalar, ahora aplicado a un sistema vectorial.

---

## 2. Contexto del paquete existente

### Archivos existentes - NO modificar salvo indicacion explicita

| Archivo | Contenido |
|---|---|
| solver.py | solve_order1, solve_order2, solve_order1_numeric |
| regressor.py | build_regressor_order1, build_regressor_order2 |
| derivatives.py | discrete_derivatives, build_taylor_matrix |
| parser.py | parse_ode, parse_and_build, show |
| examples.py | 8 ejemplos de validacion escalares |
| __init__.py | Exporta los simbolos publicos del paquete |

### Como funciona el solver escalar actual (referencia)

Para y' + f(y) = u(t), en cada paso k el solver:

1. Parte de y[k] = y[k-1] como estimacion inicial.
2. Forma el residuo discreto:
   g = (3*y[k] - 4*y[k-1] + y[k-2]) / (2T) + f(y[k]) - u[k]
3. Aplica tres correcciones homotopicas sucesivas (z1, z2, z3) usando
   g, g', g'', g''' (las derivadas de g respecto a y[k]).

El sistema vectorial sigue exactamente la misma logica, pero con un
vector residuo G = (g1, g2, g3) y el Jacobiano J = DG (2x2 o 3x3)
en lugar de los escalares g y g'.

---

## 3. Discretizacion del sistema vectorial

### Derivadas discretas (3 puntos hacia atras, igual que solver.py)

```
x'[k]  = (3*x[k] - 4*x[k-1] + x[k-2]) / (2T)
x''[k] = (x[k] - 2*x[k-1] + x[k-2]) / T^2
```

Idem para y[k], z[k].

### Vector residuo en el paso k

```
g1 = F(x[k], y[k], z[k],  xp[k], yp[k], zp[k],  xpp[k], ypp[k], zpp[k],  t[k]) - u[k]
g2 = G(x[k], y[k], z[k],  xp[k], yp[k], zp[k],  xpp[k], ypp[k], zpp[k],  t[k]) - v[k]
g3 = H(x[k], y[k], z[k],  xp[k], yp[k], zp[k],  xpp[k], ypp[k], zpp[k],  t[k]) - w[k]
```

donde xp, yp, zp, xpp, ypp, zpp son las aproximaciones discretas,
todas expresadas en funcion de x[k], y[k], z[k] (las incognitas) y
los valores pasados x[k-1], x[k-2], etc. (conocidos).

### Jacobiano del residuo respecto a las incognitas (x[k], y[k], z[k])

Por la regla de la cadena:

```
dgi/dx[k] = dFi/dx + dFi/dx' * (3/(2T)) + dFi/dx'' * (1/T^2)
dgi/dy[k] = dFi/dy + dFi/dy' * (3/(2T)) + dFi/dy'' * (1/T^2)
dgi/dz[k] = dFi/dz + dFi/dz' * (3/(2T)) + dFi/dz'' * (1/T^2)
```

El Jacobiano J es la matriz (2x2 o 3x3) formada por todas estas derivadas parciales.

---

## 4. Algoritmo homotopico vectorial por paso k

Inicializar: q = [x[k], y[k], z[k]] = [x[k-1], y[k-1], z[k-1]]

**Correccion z1 (Newton vectorial):**
```
G_vec = vector residuo en q
J_mat = Jacobiano en q
dz    = J_mat^{-1} * G_vec
q     = q - dz
```

**Correccion z2 (curvatura - Hessiano):**
```
G_vec = vector residuo en q  (recalcular)
J_mat = Jacobiano en q
H_vec = H[dz, dz]            (producto bilineal Hessiano aplicado a dz)
dz2   = -0.5 * J_mat^{-1} * H_vec
q     = q + dz2
```

**Correccion z3 (tercer orden - tensor):**
```
G_vec = vector residuo en q  (recalcular)
J_mat = Jacobiano en q
H_vec = H[dz, dz2]           (producto bilineal cruzado)
T_vec = T[dz, dz, dz]        (tensor trilineal)
dz3   = -J_mat^{-1} * (H_vec + (1/6)*T_vec)
q     = q + dz3
```

Guardar: x[k], y[k], z[k] = q

### Inversion del sistema lineal

- Para 2x2: usar formula analitica (no np.linalg.solve en el loop):
  ```
  J = [[a,b],[c,d]];  det = a*d - b*c
  J^{-1} * v = (1/det) * [d*v1 - b*v2,  -c*v1 + a*v2]
  ```
- Para 3x3: usar np.linalg.solve (aceptable).

---

## 5. Archivos a crear

### 5.1 solver_system.py

Solver numerico puro para sistemas de 2 y 3 EDOs.

**Interfaz publica:**

```python
def solve_system(funcs, jac_funcs, hess_funcs, tens_funcs,
                 excitations, initial_conditions, T, n):
    """
    Resuelve un sistema de N EDOs acopladas usando el regresor homotopico.

    Parametros
    ----------
    funcs : list of callable, longitud N
        [F, G, H] donde cada funcion tiene firma:
            F(x, y, z, xp, yp, zp, xpp, ypp, zpp, t) -> float
        Para 1er orden: xpp=ypp=zpp=0 siempre.
        Para 2 ecuaciones: z=zp=zpp=0 siempre.

    jac_funcs : list of list of callable, forma NxN
        jac_funcs[i][j] = dgi/dqj ya combinada con la regla de la cadena.

    hess_funcs : list of list of list of callable, forma NxNxN
        hess_funcs[i][j][l] = d^2gi/dqj dql en el punto actual.

    tens_funcs : list o None
        Tensor de tercer orden. Si None, se ignora z3.

    excitations : list of np.ndarray, longitud N
        [u, v, w] arrays de longitud n. Pasar np.zeros(n) para autonomos.

    initial_conditions : list of list of float
        [[x0,x1], [y0,y1], [z0,z1]]  (dos condiciones iniciales por variable)

    T : float
        Periodo de muestreo.

    n : int
        Numero total de puntos.

    Retorna
    -------
    results : list of np.ndarray, longitud N
        [x, y, z] arrays de longitud n.
    """
```

```python
def solve_system_numeric(funcs, excitations, initial_conditions, T, n, h=1e-5):
    """
    Version con Jacobiano, Hessiano y tensor calculados numericamente
    mediante diferencias finitas centradas de orden 4.
    Solo requiere [F, G, H].
    """
```

### 5.2 regressor_system.py

Constructor simbolico usando SymPy. Recibe las expresiones simbolicas de F, G, H
y genera automaticamente todas las derivadas parciales necesarias.

```python
def build_system_regressor(func_exprs, state_syms, order=1):
    """
    Construye el regresor homotopico para un sistema de N EDOs.

    Parametros
    ----------
    func_exprs : list of sympy.Expr
        [F_expr, G_expr, H_expr] en terminos de los simbolos de estado.

    state_syms : list of sympy.Symbol
        Para 1er orden: [x, y, z, xp, yp, zp]
        Para 2do orden: [x, y, z, xp, yp, zp, xpp, ypp, zpp]

    order : int
        Orden maximo de las derivadas presentes (1 o 2).

    Retorna
    -------
    regressor : callable con misma firma que solve_system
    info : dict con todas las derivadas simbolicas y numericas
    """
```

### 5.3 examples_system.py

Cuatro ejemplos clasicos de validacion. Cada uno:
1. Define F, G (y H si aplica) con sus derivadas analiticas explicitas.
2. Obtiene solucion de referencia con scipy.integrate.solve_ivp (RK45, tol=1e-9).
3. Llama a solve_system(...).
4. Imprime error maximo en cada variable.
5. Retorna (t, refs, sols) para graficar.

---

## 6. Los cuatro ejemplos clasicos

### Ejemplo 1 - Lotka-Volterra (2 ec., 1er orden, autonomo)

```
x' = alpha*x - beta*x*y       (presas)
y' = delta*x*y - gamma*y      (depredadores)
```
Parametros: alpha=1.0, beta=0.1, gamma=1.5, delta=0.075
Condiciones iniciales: x(0)=10, y(0)=5
Tiempo: t en [0, 30], n=3000
Excitacion: u=zeros, v=zeros

Forma residuo:
```
g1 = xp[k] - alpha*x[k] + beta*x[k]*y[k]
g2 = yp[k] - delta*x[k]*y[k] + gamma*y[k]
```

Jacobiano analitico d(g1,g2)/d(x[k],y[k]):
```
J[0,0] = 3/(2T) - alpha + beta*y[k]
J[0,1] = beta*x[k]
J[1,0] = -delta*y[k]
J[1,1] = 3/(2T) - delta*x[k] + gamma
```

---

### Ejemplo 2 - Lorenz (3 ec., 1er orden, autonomo)

```
x' = sigma*(y - x)
y' = x*(rho - z) - y
z' = x*y - beta_l*z
```
Parametros: sigma=10, rho=28, beta_l=8/3
Condiciones iniciales: x(0)=1, y(0)=1, z(0)=1
Tiempo: t en [0, 5], n=50000
Excitacion: u=v=w=zeros

Nota: Lorenz es muy sensible al paso temporal.
Usar n=50000 para t en [0,5] da T~1e-4.
Reportar error solo en t en [0,2] donde la solucion es comparable.

---

### Ejemplo 3 - Pendulo doble (2do orden acoplado)

Sistema mecanico con masas m=1, longitudes l=1, g=9.8:

```
(2*th1'' + th2''*cos(th1-th2)) + th2'^2*sin(th1-th2) + 2*(g/l)*sin(th1) = 0
(th1''*cos(th1-th2) + th2'') - th1'^2*sin(th1-th2) + (g/l)*sin(th2) = 0
```

Condiciones iniciales: th1(0)=pi/4, th1'(0)=0, th2(0)=pi/6, th2'(0)=0
Tiempo: t en [0, 10], n=10000
Excitacion: u=v=zeros

Nota importante: th1'' y th2'' estan acoplados linealmente en el lado izquierdo.
Antes de aplicar el regresor, despejar th1'' y th2'' resolviendo el sistema
lineal 2x2 que forman. Esto da las dos ecuaciones en forma F=0, G=0
con th1'' y th2'' aislados.

---

### Ejemplo 4 - Osciladores de Duffing acoplados (forzado)

```
x'' + alpha*x' + beta*x + gamma*x^3 + kappa*(x-y) = A*cos(omega*t)
y'' + alpha*y' + beta*y + gamma*y^3 - kappa*(x-y) = 0
```
Parametros: alpha=0.1, beta=1.0, gamma=0.2, kappa=0.5, A=0.5, omega=1.2
Condiciones iniciales: x(0)=0, xp(0)=0.5, y(0)=0.2, yp(0)=0
Tiempo: t en [0, 50], n=10000
Excitacion: u(t)=A*cos(omega*t), v(t)=zeros

Jacobiano 2x2 d(g1,g2)/d(x[k],y[k]):
```
J[0,0] = 1/T^2 + alpha*3/(2T) + beta + 3*gamma*x[k]^2 + kappa
J[0,1] = -kappa
J[1,0] = -kappa
J[1,1] = 1/T^2 + alpha*3/(2T) + beta + 3*gamma*y[k]^2 + kappa
```

---

## 7. Convenciones de codigo

- Estilo consistente con solver.py existente.
- Docstrings en ingles formato NumPy, comentarios inline en espanol.
- Sin scipy en solver_system.py. Solo en examples_system.py para referencia.
- Sin np.linalg.inv en el loop para 2x2. Si permitido para 3x3.
- Nombres: q para vector de estado del paso k, G_vec para residuo vectorial,
  J_mat para el Jacobiano.
- Las excitaciones autonomas se pasan como np.zeros(n), nunca como None.

---

## 8. Modificaciones a archivos existentes

### __init__.py - agregar al final sin tocar lo existente

```python
from .solver_system import solve_system, solve_system_numeric
from .regressor_system import build_system_regressor
```

---

### Ejemplo 5 - Euler cuerpo rigido (3 ec., 1er orden, autonomo/forzado)

Ecuaciones de Euler para la rotacion libre de un cuerpo rigido asimetrico:

```
I1*w1' = (I2 - I3)*w2*w3 + u1(t)
I2*w2' = (I3 - I1)*w3*w1 + u2(t)
I3*w3' = (I1 - I2)*w1*w2 + u3(t)
```

Parametros: I1=2.0, I2=1.0, I3=0.5  (momentos de inercia principales, todos distintos)
Condiciones iniciales: w1(0)=1.0, w2(0)=0.1, w3(0)=0.5
Tiempo: t en [0, 20], n=2000
Excitacion: u1=u2=u3=zeros (caso autonomo)

Forma residuo (dividiendo por Ii cada ecuacion):
```
g1 = w1p[k] - (I2-I3)/I1 * w2[k]*w3[k]
g2 = w2p[k] - (I3-I1)/I2 * w3[k]*w1[k]
g3 = w3p[k] - (I1-I2)/I3 * w1[k]*w2[k]
```

Jacobiano analitico d(g1,g2,g3)/d(w1[k],w2[k],w3[k]):
```
J[0,0] = 3/(2T)              J[0,1] = -(I2-I3)/I1 * w3[k]   J[0,2] = -(I2-I3)/I1 * w2[k]
J[1,0] = -(I3-I1)/I2 * w3[k]  J[1,1] = 3/(2T)              J[1,2] = -(I3-I1)/I2 * w1[k]
J[2,0] = -(I1-I2)/I3 * w2[k]  J[2,1] = -(I1-I2)/I3 * w1[k]  J[2,2] = 3/(2T)
```

Validacion especial: el caso autonomo conserva dos integrales primeras:
- Energia cinetica:  E = 0.5*(I1*w1^2 + I2*w2^2 + I3*w3^2) = constante
- Momento angular:   L = I1^2*w1^2 + I2^2*w2^2 + I3^2*w3^2 = constante

Agregar en el ejemplo el calculo de la deriva de E y L a lo largo de la
solucion numerica. Un buen regresor debe mantener ambas conservadas.

Opcional (caso forzado): agregar torque externo
```
u1(t) = 0.1*sin(2t),  u2(t) = 0,  u3(t) = 0
```
y comparar contra solve_ivp.

---

## 9. Criterios de exito

Ejecutar python examples_system.py debe mostrar:

| Ejemplo | Variable | Error maximo aceptable |
|---|---|---|
| 1 - Lotka-Volterra | x, y | < 1e-2 |
| 2 - Lorenz (t<=2) | x, y, z | < 1e-1 |
| 3 - Pendulo doble | th1, th2 | < 5e-2 |
| 4 - Duffing acoplado | x, y | < 1e-2 |
| 5 - Euler cuerpo rigido | w1, w2, w3 | < 1e-3 |
| 5 - Euler (conservacion) | deriva E, deriva L | < 1e-4 |

Si un ejemplo no converge con el paso por defecto, reducir T (aumentar n)
antes de concluir que el metodo falla. Documentar el T necesario.

---

## 10. Lo que NO hay que hacer

- No modificar solver.py, regressor.py, parser.py, derivatives.py, examples.py
- No usar np.linalg.inv en el loop interno para 2x2
- No usar scipy.integrate en solver_system.py
- No agregar dependencias nuevas (numpy, sympy, scipy ya estan instaladas)
- No intentar identificar F, G, H: son siempre funciones completamente conocidas

---

## 11. Orden de implementacion sugerido

1. solver_system.py - empezar con caso 2x2, 1er orden
2. Validar con Lotka-Volterra (Ejemplo 1)
3. Extender a 3x3 y validar con Lorenz (Ejemplo 2)
4. Agregar soporte 2do orden y validar con Duffing acoplado (Ejemplo 4)
5. Pendulo doble (Ejemplo 3) - requiere despejar th1'', th2'' antes
6. regressor_system.py - constructor simbolico
7. examples_system.py - los cuatro ejemplos completos
8. Actualizar __init__.py

---

*Fin del contrato - Rodolfo H. Rodrigo / UNSJ / Marzo 2026*
