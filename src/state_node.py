"""
state_node.py
-------------
Conversión entre representación estado-estado y estado-nodo de la TPM.

Forma estado-estado: (2^n, 2^n)
  - filas: estados en t
  - columnas: estados en t+1 (combinaciones completas)

Forma estado-nodo: lista de n matrices, cada una (2^n, 2)
  - Para cada variable Xi, columna 0 = P(Xi_t+1 = 0 | estado_t)
                            columna 1 = P(Xi_t+1 = 1 | estado_t)

Esta representación permite la descomposición tensorial:
  P(V_t+1 | V_t) = P(X1_t+1|V_t) ⊗ P(X2_t+1|V_t) ⊗ ... ⊗ P(Xn_t+1|V_t)
"""

import numpy as np
from tpm_loader import index_to_state, state_to_index


def state_state_to_state_node(tpm: np.ndarray, n: int) -> list[np.ndarray]:
    """
    Convierte una TPM de forma estado-estado a una lista de n matrices estado-nodo.

    Parámetros
    ----------
    tpm : np.ndarray de forma (2^n, 2^n)
    n   : número de variables

    Retorna
    -------
    node_matrices : lista de n arrays, cada uno de forma (2^n, 2)
                    node_matrices[i][s, v] = P(Xi_t+1 = v | estado_t = s)
    """
    num_states = 2 ** n
    node_matrices = []

    for var_idx in range(n):
        node_mat = np.zeros((num_states, 2), dtype=float)
        for row in range(num_states):
            # Sumar todas las columnas donde Xi_t+1 = 0 ó 1
            for col in range(num_states):
                col_state = index_to_state(col, n)
                xi_val = col_state[var_idx]
                node_mat[row, xi_val] += tpm[row, col]
        node_matrices.append(node_mat)

    return node_matrices


def state_node_to_state_state(node_matrices: list[np.ndarray], n: int) -> np.ndarray:
    """
    Reconstruye la TPM estado-estado a partir de las matrices estado-nodo
    usando el producto tensorial (independencia condicional).

    P(V_t+1|V_t) = ⊗_{i=1}^{n} P(Xi_t+1|V_t)

    El producto tensorial aquí es: dado el estado t fijo (fila), el vector de
    probabilidades en t+1 es el producto de Kronecker de los vectores individuales.

    Parámetros
    ----------
    node_matrices : lista de n arrays de forma (2^n, 2)
    n             : número de variables

    Retorna
    -------
    tpm : np.ndarray de forma (2^n, 2^n)
    """
    num_states = 2 ** n
    tpm = np.zeros((num_states, num_states), dtype=float)

    for row in range(num_states):
        # Vector de probabilidades para cada variable en t+1
        # Producto de Kronecker acumulado
        prob_vec = node_matrices[0][row, :]  # shape (2,)
        for i in range(1, n):
            prob_vec = np.kron(prob_vec, node_matrices[i][row, :])
        tpm[row, :] = prob_vec

    return tpm


def get_node_matrix(tpm: np.ndarray, n: int, var_idx: int) -> np.ndarray:
    """
    Obtiene la matriz estado-nodo para una sola variable var_idx.

    Parámetros
    ----------
    tpm     : TPM estado-estado (2^n, 2^n)
    n       : número de variables
    var_idx : índice de la variable (0-based)

    Retorna
    -------
    node_mat : np.ndarray de forma (2^n, 2)
    """
    num_states = 2 ** n
    node_mat = np.zeros((num_states, 2), dtype=float)
    for row in range(num_states):
        for col in range(num_states):
            col_state = index_to_state(col, n)
            xi_val = col_state[var_idx]
            node_mat[row, xi_val] += tpm[row, col]
    return node_mat


def tensor_product_tpm(
    tpm_full: np.ndarray,
    n: int,
    s1_t: list[int],
    s1_t1: list[int],
    s2_t: list[int],
    s2_t1: list[int],
) -> np.ndarray:
    """
    Calcula la TPM reconstruida para la bipartición {S1, S2} usando producto tensorial.
    Soporta biparticiones donde las variables en t y en t+1 pueden ser distintas
    dentro de cada parte (espacio completo 2^(2n-1)-1 de biparticiones).

    La reconstrucción es:
        P_rec(j | i) = P_S1(proj_s1_t1(j) | proj_s1_t(i))
                     * P_S2(proj_s2_t1(j) | proj_s2_t(i))

    Parámetros
    ----------
    tpm_full : TPM del subsistema (2^n, 2^n)
    n        : número de variables del subsistema
    s1_t     : índices de variables en t asignadas a S1
    s1_t1    : índices de variables en t+1 asignadas a S1
    s2_t     : índices de variables en t asignadas a S2
    s2_t1    : índices de variables en t+1 asignadas a S2

    Retorna
    -------
    tpm_reconstructed : np.ndarray (2^n, 2^n)
    """
    from marginalization import marginalize_rows, marginalize_columns

    # P(s1_t1 | s1_t) — si s1_t vacío, promedia todas las filas (marginal)
    tpm_s1 = marginalize_rows(tpm_full, n, s1_t)
    tpm_s1 = marginalize_columns(tpm_s1, n, s1_t1)

    # P(s2_t1 | s2_t)
    tpm_s2 = marginalize_rows(tpm_full, n, s2_t)
    tpm_s2 = marginalize_columns(tpm_s2, n, s2_t1)

    num_states = 2 ** n
    tpm_reconstructed = np.zeros((num_states, num_states), dtype=float)

    for row_full in range(num_states):
        full_state = index_to_state(row_full, n)
        row_s1 = state_to_index([full_state[v] for v in s1_t])
        row_s2 = state_to_index([full_state[v] for v in s2_t])

        for col_full in range(num_states):
            full_col = index_to_state(col_full, n)
            col_s1 = state_to_index([full_col[v] for v in s1_t1])
            col_s2 = state_to_index([full_col[v] for v in s2_t1])
            tpm_reconstructed[row_full, col_full] = (
                tpm_s1[row_s1, col_s1] * tpm_s2[row_s2, col_s2]
            )

    return tpm_reconstructed
