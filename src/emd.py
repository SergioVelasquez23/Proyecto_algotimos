"""
emd.py
------
Earth Mover's Distance (EMD) con distancia de Hamming como costo de transporte.

La EMD mide el mínimo 'trabajo' necesario para transformar una distribución
de probabilidad P en otra Q. El costo de mover una unidad de masa entre las
posiciones i y j es la distancia de Hamming d_H(i, j).

Se resuelve como un problema de transporte lineal usando scipy.
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linprog


def hamming_distance_matrix(n: int) -> np.ndarray:
    """
    Genera la matriz de costos de Hamming entre todos los pares de estados
    para un sistema de n variables.

    Retorna
    -------
    D : np.ndarray de forma (2^n, 2^n)
        D[i, j] = distancia de Hamming entre estado i y estado j
    """
    num_states = 2 ** n
    # Representar cada estado como vector binario
    states = np.array([
        [(idx >> (n - 1 - bit)) & 1 for bit in range(n)]
        for idx in range(num_states)
    ], dtype=float)

    # Distancia de Hamming = distancia Manhattan en binario = distancia L1
    D = cdist(states, states, metric="hamming") * n
    return D.astype(float)


def emd(
    p: np.ndarray,
    q: np.ndarray,
    cost_matrix: np.ndarray,
) -> float:
    """
    Calcula la Earth Mover's Distance entre distribuciones p y q
    usando la matriz de costos dada.

    El problema de transporte se formula como:
      min  Σ_{i,j} C[i,j] * F[i,j]
      s.t. Σ_j F[i,j] = p[i]  (restricciones de oferta)
           Σ_i F[i,j] = q[j]  (restricciones de demanda)
           F[i,j] >= 0

    Parámetros
    ----------
    p            : distribución fuente (1D, suma = 1)
    q            : distribución destino (1D, suma = 1)
    cost_matrix  : matriz de costos (len(p), len(q))

    Retorna
    -------
    emd_value : float, distancia EMD
    """
    n_src = len(p)
    n_dst = len(q)

    # Variables de decisión: F[i,j] aplanado, tamaño n_src * n_dst
    # Función objetivo: minimizar Σ C[i,j] * F[i,j]
    c = cost_matrix.flatten()

    # Restricciones de igualdad:
    # Oferta: Σ_j F[i,j] = p[i]  → n_src ecuaciones
    # Demanda: Σ_i F[i,j] = q[j] → n_dst ecuaciones
    # Total: n_src + n_dst ecuaciones, pero una es redundante

    n_vars = n_src * n_dst
    A_eq = np.zeros((n_src + n_dst, n_vars), dtype=float)
    b_eq = np.zeros(n_src + n_dst, dtype=float)

    # Restricciones de oferta
    for i in range(n_src):
        A_eq[i, i * n_dst:(i + 1) * n_dst] = 1.0
        b_eq[i] = p[i]

    # Restricciones de demanda
    for j in range(n_dst):
        A_eq[n_src + j, j::n_dst] = 1.0
        b_eq[n_src + j] = q[j]

    # Límites: F[i,j] >= 0
    bounds = [(0.0, None)] * n_vars

    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    if not result.success:
        raise RuntimeError(f"El problema de transporte no convergió: {result.message}")

    return float(result.fun)


def compute_delta(
    p_original: np.ndarray,
    p_reconstructed: np.ndarray,
    n: int,
    initial_state: int,
) -> float:
    """
    Calcula la discrepancia δ entre la distribución original y la reconstruida
    dado un estado inicial específico usando EMD con distancia Hamming.

    δ(V, {S1,S2}) = EMD(P(V_t+1 | V_t=s), P_reconstruida(V_t+1 | V_t=s))

    Parámetros
    ----------
    p_original      : TPM original (2^n, 2^n)
    p_reconstructed : TPM reconstruida por producto tensorial (2^n, 2^n)
    n               : número de variables del subsistema
    initial_state   : índice del estado inicial en t

    Retorna
    -------
    delta : float, discrepancia EMD para ese estado inicial
    """
    D = hamming_distance_matrix(n)
    p = p_original[initial_state, :]
    q = p_reconstructed[initial_state, :]
    return emd(p, q, D)


def compute_delta_all_states(
    p_original: np.ndarray,
    p_reconstructed: np.ndarray,
    n: int,
) -> np.ndarray:
    """
    Calcula δ para todos los estados iniciales posibles.

    Retorna
    -------
    deltas : array de forma (2^n,) con δ para cada estado inicial
    """
    D = hamming_distance_matrix(n)
    num_states = 2 ** n
    deltas = np.zeros(num_states, dtype=float)
    for s in range(num_states):
        deltas[s] = emd(p_original[s, :], p_reconstructed[s, :], D)
    return deltas
