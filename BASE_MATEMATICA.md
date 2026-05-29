# Base Matemática del Proyecto: Bipartición Óptima en Sistemas Causales Binarios

> **Guía de referencia** — Explica cada concepto matemático, su implementación exacta en el
> código y cómo verificar que funcione correctamente. Los números de línea apuntan al estado
> actual del repositorio.

---

## Índice

1. [Sistema de Variables Binarias y Representación de Estados](#1-sistema-de-variables-binarias-y-representación-de-estados)
2. [Matriz de Probabilidad de Transición (TPM)](#2-matriz-de-probabilidad-de-transición-tpm)
3. [Condicionamiento de Variables de Fondo](#3-condicionamiento-de-variables-de-fondo)
4. [Marginalización](#4-marginalización)
5. [Representación Estado-Nodo e Independencia Condicional](#5-representación-estado-nodo-e-independencia-condicional)
6. [Producto Tensorial (Kronecker por fila)](#6-producto-tensorial-kronecker-por-fila)
7. [Hipercubo n-dimensional y Distancia de Hamming](#7-hipercubo-n-dimensional-y-distancia-de-hamming)
8. [Función de Costo Geométrica T](#8-función-de-costo-geométrica-t)
9. [Earth Mover's Distance (EMD)](#9-earth-movers-distance-emd)
10. [Bipartición Óptima y Discrepancia δ](#10-bipartición-óptima-y-discrepancia-δ)
11. [Cómo verificar cada paso](#11-cómo-verificar-cada-paso)

---

## 1. Sistema de Variables Binarias y Representación de Estados

### Concepto matemático

Un sistema de **n variables binarias** se define como V = {X₁, X₂, …, Xₙ} donde cada Xᵢ ∈ {0, 1}.
El espacio de estados tiene tamaño **2ⁿ** y cada estado es un vector binario de longitud n.

Para indexar estados se usa **codificación big-endian**:

```
estado [b₁, b₂, …, bₙ]  →  índice = b₁·2^(n-1) + b₂·2^(n-2) + … + bₙ·2⁰
```

Ejemplos para n = 3:

| Estado | Índice |
|--------|--------|
| [0,0,0] | 0 |
| [0,0,1] | 1 |
| [0,1,0] | 2 |
| [1,0,0] | 4 |
| [1,1,1] | 7 |

### Implementación

**Archivo:** `src/tpm_loader.py`

```python
def state_to_index(state: list[int]) -> int:
    # big-endian: desplaza y acumula
    result = 0
    for bit in state:
        result = (result << 1) | int(bit)
    return result

def index_to_state(idx: int, n: int) -> list[int]:
    return [(idx >> (n - 1 - i)) & 1 for i in range(n)]
```

### Verificación

```python
from src.tpm_loader import state_to_index, index_to_state

assert state_to_index([1, 0, 0]) == 4     # 1·4 + 0·2 + 0·1 = 4
assert index_to_state(4, 3) == [1, 0, 0]
assert state_to_index([0, 1, 1]) == 3     # 0·4 + 1·2 + 1·1 = 3
assert index_to_state(7, 3) == [1, 1, 1]
```

---

## 2. Matriz de Probabilidad de Transición (TPM)

### Concepto matemático

La TPM es la representación central del sistema. Es una matriz **P** de forma (2ⁿ × 2ⁿ) donde:

```
P[i][j] = P(V_{t+1} = j | V_t = i)
```

**Propiedades que debe cumplir:**
- Cada entrada está en [0, 1]
- Cada **fila suma 1** (distribución de probabilidad sobre los estados siguientes)
- Fila i representa el estado actual en t; columna j, el estado en t+1

**Ejemplo (sistema ABC, n=3):**

```
Estado t \ Estado t+1 →  000  001  010  011  100  101  110  111
000                        1    0    0    0    0    0    0    0
001                        0    0    0    0    1    0    0    0
010                        0    0    1    0    0    0    0    0
...
```

→ desde el estado 000, el sistema va con certeza al estado 000.
→ desde el estado 001, el sistema va con certeza al estado 100.

### Implementación

**Archivo:** `src/tpm_loader.py` — función `load_tpm`

El CSV no tiene encabezados; cada fila es un vector de probabilidades. La función valida:
1. Que el archivo exista
2. Que la matriz sea cuadrada
3. Que el número de estados sea potencia de 2 (`log2(n_rows) ∈ ℤ`)
4. Que cada fila sume 1 (tolerancia 1e-6)

### Verificación

```python
from src.tpm_loader import load_tpm

tpm, n = load_tpm('data/tpm_abcd.csv')
assert tpm.shape == (16, 16)          # 2^4 = 16 estados para ABCD
assert n == 4
import numpy as np
assert np.allclose(tpm.sum(axis=1), 1.0)   # todas las filas suman 1

# Verificar entrada específica del Example 1.2:
# estado 0000 (idx 0) → siguiente estado 0000 con prob 1
assert tpm[0, 0] == 1.0
# estado 1000 (idx 8) → siguiente estado 1101 (idx 13) con prob 1
assert tpm[8, 13] == 1.0
```

---

## 3. Condicionamiento de Variables de Fondo

### Concepto matemático

Dado un sistema V = {X₁, …, Xₙ}, el **sistema candidato** Vᶜ ⊆ V son las variables de interés.
Las variables restantes Vᵇ = V \ Vᶜ son **variables de fondo** (background conditions).

El condicionamiento consiste en **seleccionar las filas** de la TPM donde las variables de fondo
toman su valor en el estado inicial:

```
TPM_condicionada = { fila i de TPM : ∀ k ∈ Vᵇ, X_k^(t) = x_k^inicial }
```

**Ejemplo (Example 1.2):** V = {A,B,C,D}, estado inicial = [1,0,0,0].
D es variable de fondo con valor D=0. Se retienen las 8 filas donde el bit de D en t es 0.

Los estados con D=0 en notación ABCD (big-endian, D es el último bit):
- 0000(0), 0010(2), 0100(4), 0110(6), 1000(8), 1010(10), 1100(12), 1110(14)

→ El resultado es una submatriz de **8 filas × 16 columnas**.

### Implementación

**Archivo:** `src/conditioning.py` — función `condition_tpm`

```python
# Para cada estado posible, verifica si las variables de fondo
# tienen el valor condicionado:
for state_idx in range(2 ** n):
    state = index_to_state(state_idx, n)
    if all(state[bg_var] == bg_val
           for bg_var, bg_val in zip(background_vars, background_values)):
        selected_rows.append(state_idx)

conditioned_tpm = tpm[selected_rows, :]   # (2^n_cand, 2^n)
```

### Verificación

```python
from src.conditioning import condition_tpm, get_background_state

tpm, n = load_tpm('data/tpm_abcd.csv')
initial = [1, 0, 0, 0]
bg_idx = [3]                                       # índice de D en ABCD
bg_val = get_background_state(initial, bg_idx)    # [0]
cond_tpm, n_cand, cand_idx = condition_tpm(tpm, n, bg_idx, bg_val)

assert cond_tpm.shape == (8, 16)    # 2^3 filas (ABC), 16 columnas (ABCD)
assert n_cand == 3
assert cand_idx == [0, 1, 2]        # variables A, B, C
```

---

## 4. Marginalización

### Concepto matemático

La marginalización elimina variables que no son de interés.

#### 4.1 Marginalización en columnas (t+1)

Para eliminar variables de fondo en **t+1**, se agrupan las columnas cuyos estados
proyectados sobre las variables candidatas son iguales:

```
TPM_marg[fila, j'] = Σ_{j : proj(j) = j'} TPM[fila, j]
```

**No requiere re-escalado**: la suma es la probabilidad marginal correcta.
Resultado: (2^n_cand × 2^n_cand).

#### 4.2 Marginalización en filas (t)

Para eliminar variables en **t** (al evaluar biparticiones), se descartan las variables
no deseadas de las filas y se **promedian** las filas con el mismo estado proyectado:

```
TPM_marg[i', j] = (1/|grupo_i'|) · Σ_{i : proj(i) = i'} TPM[i, j]
```

El promedio (en lugar de suma) preserva la propiedad de que cada fila suma 1.

### Implementación

**Archivo:** `src/marginalization.py`

```python
# Columnas (t+1): agrupar por proyección
for col_idx in range(2 ** n_total):
    full_state = index_to_state(col_idx, n_total)
    projected = [full_state[v] for v in keep_vars_t1]
    out_col = state_to_index(projected)
    marginalized[:, out_col] += tpm[:, col_idx]  # sumar

# Filas (t): agrupar y promediar
for row_idx in range(2 ** n_candidate):
    full_state = index_to_state(row_idx, n_candidate)
    projected = [full_state[v] for v in keep_vars_t]
    out_row = state_to_index(projected)
    accumulated[out_row] += tpm[row_idx]
    counts[out_row] += 1
accumulated /= counts  # promedio
```

### Verificación

```python
from src.marginalization import get_candidate_tpm
import numpy as np

cand_tpm = get_candidate_tpm(cond_tpm, n, cand_idx)

assert cand_tpm.shape == (8, 8)                   # sistema candidato ABC: 2^3 × 2^3
assert np.allclose(cand_tpm.sum(axis=1), 1.0)    # sigue siendo distribución de prob.

# Fila 0 (estado ABC=000): va al estado 000 con prob 1
assert cand_tpm[0, 0] == 1.0
# Fila 4 (estado ABC=100): va al estado 110 con prob 1
assert cand_tpm[4, 6] == 1.0
```

---

## 5. Representación Estado-Nodo e Independencia Condicional

### Concepto matemático

El **Teorema de Independencia Condicional** establece que las variables en t+1
son condicionalmente independientes dado el estado completo en t:

```
P(X₁_{t+1}, X₂_{t+1}, …, Xₙ_{t+1} | V_t) =
    P(X₁_{t+1}|V_t) · P(X₂_{t+1}|V_t) · … · P(Xₙ_{t+1}|V_t)
```

Esto permite descomponer la TPM (2ⁿ × 2ⁿ) en **n matrices estado-nodo**, cada una (2ⁿ × 2):

```
M_i[s, v] = P(Xᵢ_{t+1} = v | V_t = s),    v ∈ {0, 1}
```

La columna 0 de M_i es P(Xᵢ=0|estado_t), la columna 1 es P(Xᵢ=1|estado_t).

**Ejemplo con ABC:**

| Estado t | P(A=0) | P(A=1) |
|----------|--------|--------|
| 000      | 1.000  | 0.000  |
| 001      | 0.000  | 1.000  |
| 010      | 1.000  | 0.000  |
| 011      | 0.000  | 1.000  |
| 100      | 0.000  | 1.000  |
| 101      | 0.500  | 0.500  |
| 110      | 0.000  | 1.000  |
| 111      | 0.500  | 0.500  |

El estado 101 tiene P(A=1|101) = 0.5 porque al marginalizar D de t+1 en la TPM original,
la variable A en el siguiente estado es incierta (mezcla de transiciones con D=0 y D=1).

### Implementación

**Archivo:** `src/state_node.py` — función `state_state_to_state_node`

```python
for var_idx in range(n):
    node_mat = np.zeros((num_states, 2))
    for row in range(num_states):
        for col in range(num_states):
            col_state = index_to_state(col, n)
            xi_val = col_state[var_idx]            # 0 o 1
            node_mat[row, xi_val] += tpm[row, col] # acumula probabilidad marginal
    node_matrices.append(node_mat)
```

### Verificación

```python
from src.state_node import state_state_to_state_node
import numpy as np

node_mats = state_state_to_state_node(cand_tpm, 3)

assert len(node_mats) == 3                          # una matriz por variable: A, B, C
assert node_mats[0].shape == (8, 2)                 # (2^3 estados, 2 valores)
assert np.allclose(node_mats[0].sum(axis=1), 1.0)  # P(A=0) + P(A=1) = 1 para cada estado

# Estado 100 (idx 4): P(A=1|100) = 1.0
assert node_mats[0][4, 1] == 1.0
# Estado 101 (idx 5): P(A=1|101) = 0.5
assert abs(node_mats[0][5, 1] - 0.5) < 1e-9
```

---

## 6. Producto Tensorial (Kronecker por fila)

### Concepto matemático

Dado el estado s en t (fijo), la distribución conjunta sobre todos los estados en t+1
se obtiene como el **producto de Kronecker** de las distribuciones individuales:

```
P(V_{t+1} | V_t = s) = M₁[s,:] ⊗ M₂[s,:] ⊗ … ⊗ Mₙ[s,:]
```

Si cada Mᵢ[s,:] = [pᵢ₀, pᵢ₁] (vector de tamaño 2), el Kronecker produce un vector de
tamaño 2ⁿ que es la distribución conjunta sobre todos los 2ⁿ estados en t+1.

**Ejemplo para n=2 con estado s fijo:**

```
[p_A0, p_A1] ⊗ [p_B0, p_B1] = [p_A0·p_B0, p_A0·p_B1, p_A1·p_B0, p_A1·p_B1]
                                  = P(AB=00), P(AB=01), P(AB=10), P(AB=11)
```

Esta operación reconstruye exactamente la TPM original cuando las variables son
condicionalmente independientes (lo que siempre se cumple por el teorema 1.2.1).

### Para biparticiones: producto tensorial generalizado

Al evaluar una bipartición {S1, S2}, cada parte tiene sus propias variables en t y t+1.
La TPM reconstruida es:

```
P_rec(j | i) = P_S1(proj_{S1,t+1}(j) | proj_{S1,t}(i))
             × P_S2(proj_{S2,t+1}(j) | proj_{S2,t}(i))
```

donde:
- `P_S1` = TPM marginalizada a las variables de S1 en t (filas) y t+1 (columnas)
- `P_S2` = ídem para S2
- `proj_{S,t}(i)` = proyección del estado i a las variables de S en t

**Propiedad:** la suma sobre j de P_rec(j|i) es siempre 1, porque:
```
Σⱼ P_rec(j|i) = (Σ_{j_S1} P_S1) × (Σ_{j_S2} P_S2) = 1 × 1 = 1
```

### Implementación

**Archivo:** `src/state_node.py`

- `state_node_to_state_state`: reconstrucción completa de la TPM desde matrices estado-nodo
- `tensor_product_tpm(tpm, n, s1_t, s1_t1, s2_t, s2_t1)`: reconstrucción para una bipartición

```python
# Para cada (fila completa i, columna completa j):
row_s1 = state_to_index([full_state[v] for v in s1_t])   # proyección de i a S1 en t
row_s2 = state_to_index([full_state[v] for v in s2_t])   # proyección de i a S2 en t
col_s1 = state_to_index([full_col[v]   for v in s1_t1])  # proyección de j a S1 en t+1
col_s2 = state_to_index([full_col[v]   for v in s2_t1])  # proyección de j a S2 en t+1
tpm_reconstructed[i, j] = tpm_s1[row_s1, col_s1] * tpm_s2[row_s2, col_s2]
```

### Verificación

```python
from src.state_node import state_node_to_state_state, tensor_product_tpm
import numpy as np

# La reconstrucción completa debe ser idéntica a la original
rec = state_node_to_state_state(node_mats, 3)
assert np.allclose(rec, cand_tpm)       # max error: 0.00e+00 ✓

# Bipartición trivial {ABC}|{} debe dar la TPM original
rec_trivial = tensor_product_tpm(cand_tpm, 3, [0,1,2], [0,1,2], [], [])
assert np.allclose(rec_trivial, cand_tpm)
```

---

## 7. Hipercubo n-dimensional y Distancia de Hamming

### Concepto matemático

Los 2ⁿ estados del sistema se mapean biyectivamente a los **vértices de un hipercubo
n-dimensional**. Cada variable Xᵢ corresponde a una dimensión del cubo.

La **distancia de Hamming** entre dos estados es el número de bits en que difieren:

```
d_H(x, y) = Σᵢ |xᵢ - yᵢ| = Σᵢ (xᵢ XOR yᵢ)  ∈ {0, 1, …, n}
```

En el hipercubo, d_H corresponde exactamente al número mínimo de aristas a recorrer.
Dos estados son **vecinos** (adyacentes) si y solo si d_H = 1.

**Ejemplos con n=3:**

| Estados | d_H |
|---------|-----|
| 000 ↔ 001 | 1 (difieren en bit C) |
| 000 ↔ 011 | 2 (difieren en bits B, C) |
| 000 ↔ 111 | 3 (difieren en todos) |
| 101 ↔ 110 | 2 (difieren en bits B, C) |

**Implementación eficiente:** `d_H(i, j) = popcount(i XOR j)`

```python
hamming_distance(i, j, n) = bin(i ^ j).count("1")
```

### Estructura de vecindad

Los vecinos de un vértice v en el hipercubo n-dimensional son los estados que difieren
en exactamente 1 bit. Para v con n bits, hay exactamente **n vecinos**, uno por bit.

### Implementación

**Archivo:** `src/hypercube.py`

```python
def hamming_distance(i: int, j: int, n: int) -> int:
    return bin(i ^ j).count("1")

def get_neighbors(v: int, n: int) -> list[int]:
    # XOR con potencias de 2 flippea cada bit individualmente
    return [v ^ (1 << bit) for bit in range(n)]
```

### Verificación

```python
from src.hypercube import hamming_distance, get_neighbors

# Distancias conocidas
assert hamming_distance(0, 1, 3) == 1   # 000 ↔ 001
assert hamming_distance(0, 3, 3) == 2   # 000 ↔ 011
assert hamming_distance(0, 7, 3) == 3   # 000 ↔ 111
assert hamming_distance(5, 6, 3) == 2   # 101 ↔ 110

# Vecinos de 000 en n=3: 001(1), 010(2), 100(4)
assert sorted(get_neighbors(0, 3)) == [1, 2, 4]

# Vecinos de 111 en n=3: 110(6), 101(5), 011(3)
assert sorted(get_neighbors(7, 3)) == [3, 5, 6]
```

---

## 8. Función de Costo Geométrica T

### Concepto matemático

Para cada variable Xᵢ con valor X[v] = P(Xᵢ_{t+1}=1 | V_t=v) en cada vértice v,
se define la **función de costo de transición** t(i,j) (GeoMIP, ecuación 3.1):

```
t(i, j) = γ · ( |X[i] − X[j]| + Σ_{k ∈ N_opt(i,j)} t(i, k) )

donde:
  γ = 2^(−d_H(i,j))          factor de decrecimiento exponencial
  N_opt(i,j) = vecinos de i que están en caminos más cortos hacia j
              = { k : k ∈ adj(i),  d_H(k,j) = d_H(i,j) − 1 }
```

La función captura la "inercia" o energía necesaria para transitar entre estados,
ponderada por la estructura topológica del hipercubo.

**Cálculo bottom-up:** t(i,j) para d=d₀ usa t(i,k) para d<d₀, por lo que se
calcula en orden creciente de distancia.

**Ejemplo (variable A, sistema ABC, desde estado 000):**

```
d=1:
  t(000, 001) = 2⁻¹ · |X[000]−X[001]| = 0.5·|0−1| = 0.5
  t(000, 010) = 2⁻¹ · |X[000]−X[010]| = 0.5·|0−0| = 0.0
  t(000, 100) = 2⁻¹ · |X[000]−X[100]| = 0.5·|0−1| = 0.5

d=2 (N_opt(000, 011) = {001, 010}):
  t(000, 011) = 2⁻² · (|X[000]−X[011]| + t(000,001) + t(000,010))
              = 0.25 · (|0−1| + 0.5 + 0.0) = 0.25 · 1.5 = 0.375

d=2 (N_opt(000, 101) = {001, 100}):
  t(000, 101) = 0.25 · (|0−1| + 0.5 + 0.5) = 0.25 · 2.0 = 0.5  (¹)
```

(¹) Nota: el valor depende de los datos de tpm_abcd; con los datos N3C del GeoMIP
los valores coincidirían exactamente con la Tabla 4.2 de ese documento.

**Justificación del factor exponencial:**
El factor γ = 2^(-d) hace que la influencia decaiga con la distancia topológica,
reflejando que las interacciones causales locales (vecinos directos) son más fuertes.
Además, la función es consistente con la EMD: 2^(-(d1+d2)) = 2^(-d1) · 2^(-d2).

### Implementación

**Archivo:** `src/hypercube.py` — función `compute_cost_table`

```python
for i in range(num_states):
    # d=1: solo contribución directa
    for j in adj[i]:
        T[i, j] = 0.5 * abs(node_probs[i] - node_probs[j])

    # d=2..n: acumular vecinos ya calculados
    for d in range(2, n + 1):
        gamma = 2.0 ** (-d)
        for j in range(num_states):
            if hamming_distance(i, j, n) != d:
                continue
            neighbors_toward = [k for k in adj[i]
                                 if hamming_distance(k, j, n) == d - 1]
            neighbor_sum = sum(T[i, k] for k in neighbors_toward)
            T[i, j] = gamma * (abs(node_probs[i] - node_probs[j]) + neighbor_sum)
```

### Verificación

```python
from src.hypercube import compute_cost_table_from_tpm

T_A = compute_cost_table_from_tpm(node_mats[0], 3, 0)

# T[i,i] = 0 siempre
import numpy as np
assert np.all(np.diag(T_A) == 0)

# Verificar que T es no negativa
assert np.all(T_A >= 0)

# Verificar valor concreto: T_A[estado 100, estado 000]
# X_A[100]=1, X_A[000]=0, d=1, gamma=0.5
# t = 0.5 * |1−0| = 0.5
assert abs(T_A[4, 0] - 0.5) < 1e-9

# Verificar tabla completa desde estado 100
# (valores producidos por el código corregido)
expected = {0: 0.5, 1: 0.1875, 2: 0.375, 3: 0.09375,
            4: 0.0, 5: 0.25,   6: 0.0,   7: 0.1875}
for j, val in expected.items():
    assert abs(T_A[4, j] - val) < 1e-6, f"T_A[100,{j}]={T_A[4,j]} ≠ {val}"
```

---

## 9. Earth Mover's Distance (EMD)

### Concepto matemático

La EMD (también llamada distancia de flujo óptimo o distancia de Wasserstein de orden 1)
mide el mínimo "trabajo" para transformar una distribución de probabilidad P en otra Q,
cuando el costo de mover una unidad de masa desde la posición i hasta j es d_H(i,j).

Se formula como un **problema de programación lineal** (Problema del Transporte):

```
minimizar    Σᵢⱼ F[i,j] · d_H(i,j)

sujeto a:    Σⱼ F[i,j] = P[i]    ∀ i   (conservación de oferta)
             Σᵢ F[i,j] = Q[j]    ∀ j   (conservación de demanda)
             F[i,j] ≥ 0
```

La solución F* es el **plan de transporte óptimo**; el valor objetivo F*·D es la EMD.

**Dimensiones del problema** para un sistema de n variables:
- Variables de decisión: F tiene 2ⁿ × 2ⁿ entradas → tamaño (4ⁿ)
- Restricciones: 2ⁿ (oferta) + 2ⁿ (demanda) = 2·2ⁿ ecuaciones

La matriz de costos D[i,j] = d_H(i,j) se construye como:

```python
D = cdist(states, states, metric="hamming") * n
# equivalente a: D[i,j] = popcount(i XOR j)
```

### Implementación

**Archivo:** `src/emd.py`

```python
def emd(p, q, cost_matrix):
    c = cost_matrix.flatten()     # función objetivo lineal

    # Restricciones de oferta: Σⱼ F[i,j] = p[i]
    A_eq = np.zeros((n_src + n_dst, n_vars))
    for i in range(n_src):
        A_eq[i, i*n_dst:(i+1)*n_dst] = 1.0

    # Restricciones de demanda: Σᵢ F[i,j] = q[j]
    for j in range(n_dst):
        A_eq[n_src+j, j::n_dst] = 1.0

    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=[(0,None)]*n_vars,
                     method="highs")
    return float(result.fun)
```

El método `"highs"` de scipy es un solver LP de alta eficiencia (simplex + punto interior).

### Verificación

```python
from src.emd import emd, hamming_distance_matrix
import numpy as np

D = hamming_distance_matrix(3)

# Propiedades de la matriz de costos
assert D.shape == (8, 8)
assert np.all(np.diag(D) == 0)      # d(i,i) = 0
assert np.all(D == D.T)             # simetría
# Desigualdad triangular se cumple por ser distancia Hamming

# Caso trivial: p == q → EMD = 0
p = np.array([1.0, 0, 0, 0, 0, 0, 0, 0])
assert emd(p, p, D) == 0.0

# Caso conocido: mover toda la masa de estado 000 a estado 001 (d_H=1)
q = np.array([0.0, 1, 0, 0, 0, 0, 0, 0])
assert emd(p, q, D) == 1.0          # costo = 1.0 · 1 unidad de masa

# Mover toda la masa de 000 a 111 (d_H=3)
q2 = np.array([0, 0, 0, 0, 0, 0, 0, 1.0])
assert emd(p, q2, D) == 3.0

# Verificar con ejemplo real: bipartición AB|C desde estado 100
from src.state_node import tensor_product_tpm
p_orig = cand_tpm[4, :]   # distribución original desde estado 100
tpm_bip = tensor_product_tpm(cand_tpm, 3, [0,1], [0,1], [2], [2])
q_bip = tpm_bip[4, :]
delta = emd(p_orig, q_bip, D)
# p_orig = [0,0,0,0,0,0,1,0], q_bip = [0.125,0,0.125,0,0.125,0,0.625,0]
# → delta = 0.5 (costo de redistribuir 37.5% de masa desde estado 110)
assert abs(delta - 0.5) < 1e-6
```

---

## 10. Bipartición Óptima y Discrepancia δ

### Concepto matemático

Una **bipartición** del sistema candidato Vᶜ de n variables divide el conjunto de
pares (variable, tiempo) en dos partes S1 y S2:

```
S1 ∪ S2 = { (X₁,t), (X₂,t), …, (Xₙ,t), (X₁,t+1), …, (Xₙ,t+1) }
S1 ∩ S2 = ∅
S1 ≠ ∅, S2 ≠ ∅
```

El número de biparticiones no triviales es:

```
|biparticiones| = 2^(2n-1) - 1
```

Derivación: el conjunto total tiene **2n** elementos. El número de subconjuntos no
vacíos y no totales es 2^(2n) - 2. Dividiendo por 2 para eliminar duplicados simétricos
(S1|S2 = S2|S1) y sumando 1 al índice: **2^(2n-1) - 1**.

| n | Biparticiones |
|---|--------------|
| 2 | 7            |
| 3 | **31**       |
| 4 | 127          |
| 5 | 511          |
| 10 | 524,287    |
| 20 | ~550 mil millones ← congela el PC |

La **discrepancia δ** de una bipartición {S1, S2} dado el estado inicial s es:

```
δ(Vᶜ, {S1,S2}, s) = EMD( P(Vᶜ_{t+1} | V_t = s),  P_rec(Vᶜ_{t+1} | V_t = s) )

donde P_rec se obtiene del producto tensorial generalizado de S1 y S2
```

La **bipartición óptima** minimiza δ sobre todas las biparticiones posibles:

```
{S1*, S2*} = argmin_{S1,S2} δ(Vᶜ, {S1,S2}, s)
```

**Interpretación:** δ=0 significa que el sistema puede ser perfectamente separado en
dos subsistemas independientes desde el estado inicial s. Un δ grande indica que la
partición destruye información relevante sobre la dinámica del sistema.

### Por qué el sistema puede congelar el PC

La complejidad de la búsqueda exhaustiva es **O(2^(2n-1) · LP(n))** donde LP(n) es el
costo de resolver un problema de programación lineal con 4ⁿ variables. Para n=10:
- ~500K problemas LP, cada uno con ~1M variables → horas de cálculo.

Por eso el GeoMIP propone usar la tabla T para identificar candidatos directamente,
reduciendo la complejidad a **O(n · 2ⁿ)** teórico.

El código implementa un límite de seguridad (`max_bipartitions=2000`) para evitar esto.

### Implementación

**Archivo:** `src/bipartition.py`

```python
# Generar el espacio de 2n variables (incluyendo t y t+1)
all_var_time = build_bipartition_variables(n)
# → [(0,'t'),(1,'t'),(2,'t'),(0,'t+1'),(1,'t+1'),(2,'t+1')] para n=3

# Enumerar biparticiones (el primer elemento siempre en S1 para evitar duplicados)
for s1, s2 in generate_bipartitions(all_var_time):
    s1_t, s1_t1, s2_t, s2_t1 = split_bipartition(s1, s2)
    delta = evaluate_bipartition(tpm, n, s1_t, s1_t1, s2_t, s2_t1, initial_idx)
```

### Verificación

```python
from src.bipartition import find_optimal_bipartition, build_bipartition_variables, generate_bipartitions

# Verificar conteo correcto de biparticiones
all_vars = build_bipartition_variables(3)
bips = list(generate_bipartitions(all_vars))
assert len(bips) == 31    # 2^(2·3-1) - 1 = 31 ✓

# Para n=2
all_vars_2 = build_bipartition_variables(2)
bips_2 = list(generate_bipartitions(all_vars_2))
assert len(bips_2) == 7   # 2^(2·2-1) - 1 = 7 ✓

# Ejecutar búsqueda completa para n=3 (31 biparticiones, manejable)
result = find_optimal_bipartition(cand_tpm, 3, 4,
                                   variable_names=['A','B','C'])
assert result['n_evaluated'] == 31
assert result['truncated'] == False
assert 'optimal_s1' in result
assert 'min_delta' in result
assert result['min_delta'] >= 0

# La bipartición óptima encontrada para este sistema/estado tiene δ=0
# (el sistema desde estado 100 es perfectamente separable)
assert result['min_delta'] == 0.0
```

---

## 11. Cómo verificar cada paso

### 11.1 Script de verificación completa

Guardar como `test_pipeline.py` y ejecutar con:
```
PYTHONPATH=src python3 test_pipeline.py
```

```python
"""test_pipeline.py — Verifica el pipeline completo del proyecto."""
import sys, numpy as np
sys.path.insert(0, 'src')

from tpm_loader import load_tpm, state_to_index, index_to_state
from conditioning import condition_tpm, get_background_state
from marginalization import get_candidate_tpm
from state_node import state_state_to_state_node, state_node_to_state_state, tensor_product_tpm
from hypercube import compute_cost_table, hamming_distance, get_neighbors
from emd import emd, hamming_distance_matrix
from bipartition import find_optimal_bipartition, build_bipartition_variables, generate_bipartitions

def check(cond, msg):
    assert cond, f"FALLO: {msg}"
    print(f"  ✓ {msg}")

print("\n=== 1. Indexación de estados ===")
check(state_to_index([1,0,0]) == 4, "state_to_index([1,0,0]) = 4")
check(index_to_state(4, 3) == [1,0,0], "index_to_state(4,3) = [1,0,0]")
check(state_to_index([]) == 0, "estado vacío → índice 0")

print("\n=== 2. Carga TPM ===")
tpm, n = load_tpm('data/tpm_abcd.csv')
check(tpm.shape == (16,16), "TPM ABCD tiene forma (16,16)")
check(n == 4, "n = 4 variables")
check(np.allclose(tpm.sum(axis=1), 1.0), "todas las filas suman 1")
check(tpm[0,0] == 1.0, "estado 0000 → 0000 con prob 1")

print("\n=== 3. Condicionamiento D=0 ===")
bg_val = get_background_state([1,0,0,0], [3])
cond_tpm, n_c, cand_idx = condition_tpm(tpm, 4, [3], bg_val)
check(cond_tpm.shape == (8,16), "TPM condicionada: 8 filas (D=0)")
check(n_c == 3, "sistema candidato tiene 3 variables")

print("\n=== 4. Marginalización → TPM candidata ===")
cand_tpm = get_candidate_tpm(cond_tpm, n, cand_idx)
check(cand_tpm.shape == (8,8), "TPM candidata: 8×8")
check(np.allclose(cand_tpm.sum(axis=1), 1.0), "filas de TPM candidata suman 1")

print("\n=== 5. Estado-nodo ===")
node_mats = state_state_to_state_node(cand_tpm, 3)
check(len(node_mats) == 3, "3 matrices estado-nodo (una por variable)")
for i, M in enumerate(node_mats):
    check(np.allclose(M.sum(axis=1), 1.0), f"variable {i}: P(X=0) + P(X=1) = 1")

print("\n=== 6. Producto tensorial ===")
rec = state_node_to_state_state(node_mats, 3)
check(np.allclose(rec, cand_tpm), "reconstrucción exacta de TPM (error=0)")

print("\n=== 7. Hipercubo y Hamming ===")
check(hamming_distance(0, 1, 3) == 1, "d_H(000,001) = 1")
check(hamming_distance(0, 7, 3) == 3, "d_H(000,111) = 3")
check(sorted(get_neighbors(0, 3)) == [1,2,4], "vecinos de 000: {001,010,100}")

print("\n=== 8. Tabla de costos T ===")
T_A = compute_cost_table(node_mats[0][:,1], 3)
check(np.all(np.diag(T_A) == 0), "T[i,i] = 0 para todo i")
check(np.all(T_A >= 0), "T ≥ 0 (costos no negativos)")
check(abs(T_A[4,0] - 0.5) < 1e-9, "T_A[100,000] = 0.5")

print("\n=== 9. EMD ===")
D = hamming_distance_matrix(3)
p = np.zeros(8); p[0] = 1.0
q = np.zeros(8); q[1] = 1.0
check(emd(p, p, D) == 0.0, "EMD(p,p) = 0")
check(abs(emd(p, q, D) - 1.0) < 1e-9, "EMD(000,001) = 1 (d_H=1)")
q7 = np.zeros(8); q7[7] = 1.0
check(abs(emd(p, q7, D) - 3.0) < 1e-9, "EMD(000,111) = 3 (d_H=3)")

print("\n=== 10. Biparticiones ===")
bips = list(generate_bipartitions(build_bipartition_variables(3)))
check(len(bips) == 31, f"n=3 genera 31 biparticiones, 2^(2·3-1)-1=31")
result = find_optimal_bipartition(cand_tpm, 3, 4, variable_names=['A','B','C'])
check(result['n_evaluated'] == 31, "se evaluaron las 31 biparticiones")
check(result['truncated'] == False, "no se truncó la búsqueda")
check(result['min_delta'] >= 0, "δ mínimo ≥ 0")

print("\n✅ Todos los verificaciones pasaron.")
```

### 11.2 Verificación rápida por consola

Para verificar un paso individualmente sin correr el pipeline completo, basta con
importar el módulo correspondiente y llamar la función con datos de prueba:

```bash
# Desde el directorio raíz del proyecto
PYTHONPATH=src python3 -c "
from hypercube import compute_cost_table
import numpy as np

# Sistema mínimo n=2: X=P(X1=1|estado) = [0,1,0,1]
node_probs = np.array([0.0, 1.0, 0.0, 1.0])
T = compute_cost_table(node_probs, 2)
print('T desde estado 00:')
print(f'  t(00,01) = {T[0,1]:.4f}  (esperado: 0.5)')
print(f'  t(00,10) = {T[0,2]:.4f}  (esperado: 0.0)')
print(f'  t(00,11) = {T[0,3]:.4f}  (esperado: 0.25*(1+0.5+0)=0.375)')
"
```

### 11.3 Casos de prueba incluidos en `data/`

| Archivo | Descripción | n | Uso |
|---------|-------------|---|-----|
| `tpm_abcd.csv` | Example 1.2 de la Guía (sistema ABCD) | 4 | Demo principal |
| `tpm_abc.csv`  | Example 1.1 / 1.5 de la Guía (sistema ABC) | 3 | Verificación pequeña |
| `tpm_2var.csv` | Sistema mínimo 2 variables | 2 | Pruebas unitarias |
| `tpm_3var_prob.csv` | Sistema 3 variables estocástico | 3 | EMD con distribuciones mixtas |
| `tpm_4var_prob.csv` | Sistema 4 variables estocástico | 4 | Pruebas intermedias |
| `tpm_5var.csv` | Sistema 5 variables | 5 | Prueba de rendimiento (511 biparticiones) |

### 11.4 Interpretación de resultados

| δ | Interpretación |
|---|----------------|
| = 0 | Bipartición perfecta: los dos subsistemas son completamente independientes desde el estado inicial |
| < 0.1 | Bipartición casi perfecta: muy poca dependencia entre subsistemas |
| ≈ máximo | Los subsistemas son altamente interdependientes; no existe buena bipartición |

**Nota sobre valores de δ:** el máximo teórico de EMD con distancia Hamming en n variables
es n (cuando toda la masa se mueve de 000…0 a 111…1). En la práctica, δ suele ser mucho
menor porque las distribuciones son similares entre sí.

---

## Apéndice: Mapa de módulos y funciones clave

| Concepto matemático | Módulo | Función(es) |
|---------------------|--------|-------------|
| Codificación big-endian | `src/tpm_loader.py` | `state_to_index`, `index_to_state` |
| Carga y validación TPM | `src/tpm_loader.py` | `load_tpm` |
| Condicionamiento de fondo | `src/conditioning.py` | `condition_tpm`, `get_background_state` |
| Marginalización columnas (t+1) | `src/marginalization.py` | `marginalize_columns`, `get_candidate_tpm` |
| Marginalización filas (t) | `src/marginalization.py` | `marginalize_rows` |
| Descomposición estado-nodo | `src/state_node.py` | `state_state_to_state_node` |
| Producto tensorial (reconstrucción) | `src/state_node.py` | `state_node_to_state_state`, `tensor_product_tpm` |
| Hipercubo: adyacencia | `src/hypercube.py` | `build_hypercube_adjacency`, `get_neighbors` |
| Distancia de Hamming | `src/hypercube.py` | `hamming_distance` |
| Función de costo T (BFS bottom-up) | `src/hypercube.py` | `compute_cost_table`, `compute_cost_table_from_tpm` |
| Matriz de costos Hamming | `src/emd.py` | `hamming_distance_matrix` |
| Earth Mover's Distance | `src/emd.py` | `emd`, `compute_delta` |
| Enumeración de biparticiones | `src/bipartition.py` | `build_bipartition_variables`, `generate_bipartitions`, `split_bipartition` |
| Evaluación discrepancia δ | `src/bipartition.py` | `evaluate_bipartition` |
| Bipartición óptima | `src/bipartition.py` | `find_optimal_bipartition` |
| Interfaz web | `app.py` | Pipeline Streamlit completo |
