"""
hypercube.py
------------
Representación geométrica del espacio de estados como hipercubo n-dimensional.

Implementa:
- Construcción del grafo del hipercubo
- Distancia de Hamming entre vértices
- BFS modificado para calcular la tabla de costos T[i,j] = t(i,j)

Función de costo (ec. 3.1 del documento):
  t(i, j) = γ · |X[i] - X[j]| + Σ t(k, j)  para k ∈ N(i,j)
  γ = 2^(-d_H(i,j))

donde X[v] es el valor (probabilidad) asociado al vértice v.
"""

import numpy as np
from collections import deque
from tpm_loader import index_to_state, state_to_index


def hamming_distance(i: int, j: int, n: int) -> int:
    """
    Distancia de Hamming entre dos estados representados como enteros.
    Equivale a contar los bits diferentes (XOR + popcount).
    """
    return bin(i ^ j).count("1")


def get_neighbors(v: int, n: int) -> list[int]:
    """
    Retorna los vecinos del vértice v en el hipercubo n-dimensional.
    (estados que difieren en exactamente 1 bit)
    """
    neighbors = []
    for bit in range(n):
        neighbor = v ^ (1 << bit)
        neighbors.append(neighbor)
    return neighbors


def build_hypercube_adjacency(n: int) -> dict[int, list[int]]:
    """
    Construye el diccionario de adyacencia del hipercubo n-dimensional.

    Retorna
    -------
    adj : dict {vértice: [vecinos]}
    """
    num_states = 2 ** n
    adj = {v: get_neighbors(v, n) for v in range(num_states)}
    return adj


def compute_cost_table(
    node_probs: np.ndarray,
    n: int,
) -> np.ndarray:
    """
    Calcula la tabla de costos T según la ecuación 3.1 del GeoMIP:

        t(i, j) = γ · ( |X[i] − X[j]| + Σ_{k ∈ N_opt(i,j)} t(i, k) )
        γ = 2^(−d_H(i,j))

    donde N_opt(i,j) son los vecinos inmediatos de i en caminos óptimos
    hacia j (nodos a distancia d−1 de j).

    El cálculo es bottom-up: primero d=1, luego d=2, …, d=n,
    garantizando que t(i,k) esté disponible al calcular t(i,j).

    Parámetros
    ----------
    node_probs : array de forma (2^n,) con el valor X[v] de cada vértice.
                 Típicamente P(Xi_t+1=1 | estado_t).
    n          : dimensión del hipercubo (número de variables)

    Retorna
    -------
    T : np.ndarray de forma (2^n, 2^n)
    """
    num_states = 2 ** n
    T = np.zeros((num_states, num_states), dtype=float)
    adj = build_hypercube_adjacency(n)

    for i in range(num_states):
        # d=1: vecinos directos — sin suma recursiva
        for j in adj[i]:
            T[i, j] = 0.5 * abs(node_probs[i] - node_probs[j])

        # d=2..n: acumular desde estados ya calculados a menor distancia
        for d in range(2, n + 1):
            gamma = 2.0 ** (-d)
            for j in range(num_states):
                if hamming_distance(i, j, n) != d:
                    continue
                # Vecinos de i que están un paso más cerca de j
                neighbors_toward = [
                    k for k in adj[i]
                    if hamming_distance(k, j, n) == d - 1
                ]
                neighbor_sum = sum(T[i, k] for k in neighbors_toward)
                T[i, j] = gamma * (abs(node_probs[i] - node_probs[j]) + neighbor_sum)

    return T


def compute_cost_table_from_tpm(
    tpm: np.ndarray,
    n: int,
    var_idx: int,
) -> np.ndarray:
    """
    Calcula la tabla de costos T para la variable var_idx usando su distribución
    P(Xi_t+1 = 1 | estado_t) como valores asociados a los vértices.

    Parámetros
    ----------
    tpm     : TPM estado-estado (2^n, 2^n) o estado-nodo (2^n, 2)
    n       : número de variables
    var_idx : índice de la variable (solo relevante si tpm es estado-estado)

    Retorna
    -------
    T : tabla de costos (2^n, 2^n)
    """
    num_states = 2 ** n

    if tpm.shape[1] == 2:
        # TPM ya está en forma estado-nodo para esta variable
        node_probs = tpm[:, 1]  # P(Xi = 1 | estado_t)
    else:
        # Extraer distribución marginal de Xi desde TPM estado-estado
        from tpm_loader import index_to_state as i2s
        node_probs = np.zeros(num_states, dtype=float)
        for row in range(num_states):
            for col in range(num_states):
                col_state = i2s(col, n)
                if col_state[var_idx] == 1:
                    node_probs[row] += tpm[row, col]

    return compute_cost_table(node_probs, n)
