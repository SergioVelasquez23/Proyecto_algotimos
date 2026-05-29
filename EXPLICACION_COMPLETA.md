# Explicación completa del proyecto

## 1. Qué hace este proyecto

Este proyecto implementa un pipeline para analizar sistemas causales binarios mediante:

1. Carga de una TPM (Transition Probability Matrix) desde CSV.
2. Condicionamiento por variables de fondo en tiempo t.
3. Marginalización para obtener el subsistema candidato.
4. Conversión a representación estado-nodo.
5. Construcción de hipercubo y tabla de costos T.
6. Evaluación de biparticiones.
7. Cálculo de discrepancia delta con EMD (Earth Mover's Distance) usando Hamming.
8. Selección de la bipartición óptima (mínimo delta).

En resumen, busca qué partición del subsistema conserva mejor la dinámica al reconstruirla como producto tensorial de sus partes.

---

## 2. Estructura del repositorio y rol de cada archivo

- main.py
  - Orquesta el pipeline completo en modo consola.
- app.py
  - Interfaz Streamlit para ejecutar y visualizar cada paso.
- src/tpm_loader.py
  - Carga y valida la TPM.
  - Convierte índice a estado binario y viceversa.
- src/conditioning.py
  - Filtra filas de TPM según variables de fondo fijadas en t.
- src/marginalization.py
  - Marginalización en columnas (t+1) y filas (t).
- src/state_node.py
  - Convierte estado-estado a estado-nodo.
  - Reconstruye TPM por producto tensorial.
- src/hypercube.py
  - Modela estados como vértices de hipercubo.
  - Calcula distancias Hamming y tabla de costos T por BFS.
- src/emd.py
  - Implementa EMD como problema de transporte lineal.
- src/bipartition.py
  - Genera biparticiones y evalúa delta para cada una.

---

## 3. Base matemática de representación de estados

Si el sistema tiene n variables binarias:

- Número de estados posibles: 2^n
- Un estado se representa como vector binario de longitud n.
  - Ejemplo para n=4: [A, B, C, D] = [1, 0, 0, 1]

### 3.1 Conversión estado <-> índice

Se usa codificación binaria big-endian.

- state_to_index([b1, b2, ..., bn])
- index_to_state(i, n)

Ejemplo con n=3:

- estado 101 -> índice 5
- índice 3 -> estado 011

Esto permite indexar filas y columnas de matrices de tamaño 2^n x 2^n.

---

## 4. TPM: definición y validaciones

La TPM P tiene forma (2^n, 2^n):

- Fila i: estado en t
- Columna j: estado en t+1
- Entrada P[i, j] = P(V_{t+1}=j | V_t=i)

### 4.1 Validaciones en src/tpm_loader.py

1. Debe existir el archivo.
2. Debe ser matriz cuadrada.
3. Número de filas/columnas debe ser potencia de 2.
4. Cada fila debe sumar 1 (tolerancia numérica 1e-6).

Si una fila no suma 1, se lanza error con índices de filas inválidas.

---

## 5. Condicionamiento por variables de fondo

Archivo: src/conditioning.py

Función principal: condition_tpm

Entrada:

- TPM completa de n variables.
- Índices de variables de fondo.
- Valores de esas variables en el estado inicial.

Salida:

- TPM condicionada en filas.
- Número de variables candidatas.
- Índices de variables candidatas.

### 5.1 Idea matemática

Si fijas, por ejemplo, D_t = 0, solo se conservan filas cuyo estado cumpla eso.

No se eliminan aún columnas de t+1 en este paso.

Si había n variables y fijas b variables de fondo, sobreviven 2^(n-b) filas.

---

## 6. Marginalización

Archivo: src/marginalization.py

Hay dos operaciones distintas:

### 6.1 Marginalización de columnas (t+1)

Función: marginalize_columns

Objetivo:

- Eliminar variables no candidatas en t+1.
- Agrupar columnas que comparten la misma proyección sobre variables candidatas.
- Sumar sus probabilidades.

Formalmente:

P(X'_cand | estado_t) = sum_{x'_resto} P(X'_cand, x'_resto | estado_t)

Resultado:

- Matriz de tamaño filas_actuales x 2^(n_candidatas)

### 6.2 Marginalización de filas (t)

Función: marginalize_rows

Objetivo:

- Eliminar variables en t que no se quieren mantener.
- Agrupar filas por estado proyectado.
- Promediar las filas agrupadas.

Resultado:

- Matriz de tamaño 2^(vars_t_mantenidas) x columnas_actuales

### 6.3 TPM del sistema candidato

Función: get_candidate_tpm

Hace exactamente la marginalización de columnas sobre la TPM ya condicionada.

Si el subsistema candidato tiene n_c variables, la TPM final candidata es de tamaño:

2^n_c x 2^n_c

---

## 7. Representación estado-nodo

Archivo: src/state_node.py

### 7.1 De estado-estado a estado-nodo

Función: state_state_to_state_node

Para cada variable Xi genera una matriz M_i de tamaño 2^n x 2:

- M_i[s, 0] = P(Xi_{t+1}=0 | estado_t=s)
- M_i[s, 1] = P(Xi_{t+1}=1 | estado_t=s)

Se obtiene sumando columnas de la TPM global según el bit i del estado futuro.

### 7.2 De estado-nodo a estado-estado

Función: state_node_to_state_state

Reconstruye TPM mediante producto de Kronecker por fila:

P(V_{t+1}|V_t=s) = kron_i P(Xi_{t+1}|V_t=s)

Esto impone una factorización condicional de la distribución conjunta futura.

### 7.3 Producto tensorial para una bipartición

Función: tensor_product_tpm

1. Construye TPM marginal para S1 (filas y columnas).
2. Construye TPM marginal para S2.
3. Para cada fila del sistema completo, toma la fila correspondiente en S1 y S2.
4. Combina con Kronecker.
5. Reordena columnas para respetar el orden original de variables.

Salida:

- TPM reconstruida del subsistema completo bajo la bipartición.

---

## 8. Hipercubo y tabla de costos T

Archivo: src/hypercube.py

### 8.1 Geometría del hipercubo

- Cada estado binario es un vértice.
- Dos vértices son vecinos si difieren en 1 bit.
- Distancia usada: Hamming.

Funciones básicas:

- hamming_distance(i, j, n)
- get_neighbors(v, n)
- build_hypercube_adjacency(n)

### 8.2 Tabla de costos T

Función: compute_cost_table

Entrada:

- node_probs: valor asociado a cada vértice, típicamente P(Xi=1|estado_t)

Se calcula T[i, j] con un BFS modificado y factor:

gamma = 2^{-d_H(i,j)}

La implementación agrega:

- componente directa proporcional a |X[i]-X[j]|
- componentes acumuladas por caminos que acercan al destino j

Función: compute_cost_table_from_tpm

- Si recibe matriz 2 columnas, usa la columna P(Xi=1|t).
- Si recibe TPM completa, la extrae marginalizando Xi en t+1.

Importante:

- En el pipeline actual, T se calcula y visualiza, pero no entra en el cálculo final de delta en biparticiones.
- El delta final de optimización usa EMD con Hamming, no la tabla T.

---

## 9. EMD (Earth Mover's Distance)

Archivo: src/emd.py

### 9.1 Matriz de costos

Función: hamming_distance_matrix(n)

Construye matriz D de tamaño 2^n x 2^n:

D[i, j] = distancia de Hamming entre estado i y estado j

### 9.2 Problema de transporte lineal

Función: emd(p, q, D)

Resuelve:

min sum_{i,j} D[i,j] F[i,j]

sujeto a:

- sum_j F[i,j] = p[i]
- sum_i F[i,j] = q[j]
- F[i,j] >= 0

Se usa scipy.optimize.linprog con método highs.

### 9.3 Delta para estado inicial

Función usada en práctica: evaluate_bipartition en src/bipartition.py

- p = fila de TPM original para estado inicial.
- q = fila de TPM reconstruida para estado inicial.
- delta = EMD(p, q, D)

Interpretación:

- delta pequeño: la bipartición preserva bien la dinámica desde ese estado inicial.
- delta grande: la factorización pierde estructura causal/dependencias.

---

## 10. Biparticiones

Archivo: src/bipartition.py

### 10.1 Qué evalúa realmente el código actual

Aunque hay helpers para biparticionar variables en t y t+1, la búsqueda óptima actual en find_optimal_bipartition particiona solo el conjunto de variables del subsistema candidato.

Si hay n_c variables, evalúa:

2^(n_c-1) - 1 biparticiones no triviales

(con una convención para evitar simetrías duplicadas)

### 10.2 Flujo por bipartición

Para cada S1 | S2:

1. Reconstruye TPM con tensor_product_tpm.
2. Extrae distribución original y reconstruida para la fila del estado inicial.
3. Calcula delta con EMD.
4. Guarda resultado.
5. Selecciona mínimo global.

Salida:

- S1 óptima
- S2 óptima
- delta mínimo

---

## 11. Recorrido completo con el ejemplo demo (Example 1.2)

Caso de prueba definido en main.py:

- Variables completas: [A, B, C, D]
- Estado inicial: [1, 0, 0, 0]
- Fondo: [D]
- Candidato: [A, B, C]

### 11.1 Paso 1

- TPM original: 16 x 16 (porque n=4)

### 11.2 Paso 2

- Condición de fondo: D_t = 0
- TPM condicionada: 8 x 16

### 11.3 Paso 3

- Se marginaliza D en t+1
- TPM candidata: 8 x 8

### 11.4 Paso 4

Se obtienen probabilidades P(Xi=1|t) por estado de ABC:

- A: [0.0, 1.0, 0.0, 1.0, 1.0, 0.5, 1.0, 0.5]
- B: [0.0, 0.0, 1.0, 0.0, 1.0, 0.5, 1.0, 0.5]
- C: [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5]

### 11.5 Paso 5

Tablas T por variable:

- T_A max=4.0971, media=0.5986
- T_B max=3.5747, media=0.6004
- T_C max=0.9850, media=0.1719

### 11.6 Paso 6

- Estado inicial candidato: [1, 0, 0]
- Índice de fila: 4

### 11.7 Paso 7

Biparticiones evaluadas:

- [A] | [B, C] -> delta = 0.750000
- [A, B] | [C] -> delta = 0.500000
- [A, C] | [B] -> delta = 0.625000

Resultado:

- Óptima: [A, B] | [C]
- delta mínimo: 0.500000

---

## 12. Complejidad computacional (aproximada)

Sea n_c el número de variables del subsistema candidato.

### 12.1 Carga TPM

- Tamaño matriz: 2^n x 2^n
- Tiempo y memoria crecen como O(4^n)

### 12.2 Marginalización

- Columnas: recorre columnas completas y proyecta estados
- Costo cercano a O(2^n * filas)

### 12.3 Estado-nodo

- Por cada variable recorre todas las celdas de TPM
- Orden aproximado O(n_c * 4^n_c)

### 12.4 Hipercubo + T

- Matriz T es 2^n_c x 2^n_c
- BFS interno para cada par (i,j)
- Costo alto para n_c grande

### 12.5 EMD

Para 2^n_c estados:

- Variables de flujo en LP: (2^n_c)^2 = 4^n_c
- Restricciones: 2 * 2^n_c
- Puede convertirse en cuello de botella principal al crecer n_c

Conclusión práctica:

- Funciona muy bien para n_c pequeño (3, 4, 5)
- Escala rápidamente en costo para n_c grandes

---

## 13. Qué entra, qué sale y en qué formato

### 13.1 Entrada principal

CSV sin encabezados:

- Dimensión 2^n x 2^n
- Valores numéricos
- Cada fila suma 1

### 13.2 Parámetros de ejecución

- Nombres de variables
- Estado inicial binario
- Variables de fondo (opcional)
- Variables candidatas (opcional)

### 13.3 Salida principal

- Listado de delta por bipartición
- Bipartición óptima
- Valor mínimo de delta

---

## 14. Diferencia entre main.py y app.py

main.py:

- Pipeline en consola.
- Ideal para pruebas rápidas y scripts.

app.py:

- Misma lógica del pipeline.
- Añade visualizaciones: heatmaps, barras, cubo 3D (n=3), tabla de resultados.

Importante en Linux:

- En el docstring aparece py -m streamlit run app.py.
- En Linux normalmente el comando correcto es:

python -m streamlit run app.py

---

## 15. Observaciones técnicas importantes del código actual

1. La tabla T del hipercubo se calcula y visualiza, pero no participa en la optimización final de delta.
2. La optimización final usa exclusivamente EMD con Hamming.
3. En src/bipartition.py existen utilidades para biparticiones sobre variables-tiempo, pero la búsqueda óptima activa particiona variables del subsistema (no el conjunto combinado t y t+1).
4. El archivo src/bipartition.py importa compute_delta_all_states pero no lo usa en la ruta principal.
5. El comportamiento es consistente con el resultado mostrado por la demo ejecutada.

---

## 16. Ejecución y verificación

### 16.1 Consola

python main.py

Con argumentos personalizados:

python main.py --csv data/tpm_abcd.csv --vars A B C D --initial 1 0 0 0 --background D --candidate A B C --verbose

### 16.2 Interfaz web

python -m streamlit run app.py

### 16.3 Dependencias

python -m pip install -r requirements.txt

---

## 17. Diccionario rápido de conceptos

- TPM: matriz de transición de probabilidades entre estados.
- Estado-estado: representación conjunta completa en t y t+1.
- Estado-nodo: probabilidades por nodo futuro condicionadas al estado presente.
- Marginalizar: eliminar variables sumando o promediando según dimensión.
- Condicionar: fijar variables de fondo a valores concretos.
- Hipercubo: grafo de estados binarios conectados por flips de un bit.
- Hamming: número de bits distintos entre dos estados.
- EMD: costo mínimo de transformar una distribución en otra.
- Delta: discrepancia entre dinámica original y reconstruida bajo bipartición.

---

## 18. Resumen ejecutivo

Este proyecto toma una dinámica probabilística binaria completa (TPM), la restringe a un subsistema relevante, la factoriza por biparticiones candidatas y mide cuánto se pierde al separar el sistema en partes.

La pérdida se cuantifica con EMD sobre distancia de Hamming desde un estado inicial dado.

La bipartición óptima es la de menor delta.

