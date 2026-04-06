"""
bipartition.py
--------------
Enumeración y evaluación de todas las biparticiones posibles del subsistema.

Para un subsistema con u variables en t y v variables en t+1:
  Número de biparticiones = 2^(u+v-1) - 1

Cada bipartición divide el conjunto de variables (t y t+1) en dos partes S1, S2
tal que S1 ∪ S2 = V y S1 ∩ S2 = ∅.

Se evalúa la discrepancia δ para cada bipartición usando EMD con Hamming.
"""

import numpy as np
from itertools import combinations
from typing import Iterator
from tpm_loader import index_to_state, state_to_index
from state_node import tensor_product_tpm
from emd import compute_delta_all_states


# ---------------------------------------------------------------------------
# Representación de biparticiones
# ---------------------------------------------------------------------------

def generate_bipartitions(variables: list) -> Iterator[tuple[tuple, tuple]]:
    """
    Genera todas las biparticiones no triviales de una lista de variables.
    Para evitar duplicados, se fija el primer elemento siempre en S1.

    Una bipartición es (S1, S2) donde:
    - S1 ∪ S2 = variables
    - S1 ∩ S2 = ∅
    - S1 ≠ ∅, S2 ≠ ∅

    Parámetros
    ----------
    variables : lista de identificadores de variables
                ej. [('A','t'), ('B','t'), ('C','t'), ('A','t+1'), ...]

    Yields
    ------
    (s1, s2) : tuplas con los elementos de cada parte
    """
    n = len(variables)
    if n < 2:
        return

    # Fijamos variables[0] siempre en S1 para evitar biparticiones simétricas
    rest = variables[1:]
    for r in range(1, n):  # tamaño de S1: 1 a n-1
        for combo in combinations(range(len(rest)), r - 1):
            s1 = [variables[0]] + [rest[i] for i in combo]
            s2 = [v for v in variables if v not in s1]
            if s2:
                yield tuple(s1), tuple(s2)


def build_bipartition_variables(n: int) -> list[tuple[str, str]]:
    """
    Construye la lista de variables del subsistema incluyendo t y t+1.
    Usada para la enumeración de biparticiones.

    Retorna lista de (variable_idx, tiempo) ej:
    [(0,'t'), (1,'t'), (2,'t'), (0,'t+1'), (1,'t+1'), (2,'t+1')]
    """
    vars_t = [(i, 't') for i in range(n)]
    vars_t1 = [(i, 't+1') for i in range(n)]
    return vars_t + vars_t1


def split_bipartition(
    s1: tuple,
    s2: tuple,
    n: int,
) -> tuple[list[int], list[int], list[int], list[int]]:
    """
    Separa una bipartición en sus componentes de t y t+1 para cada parte.

    Retorna
    -------
    s1_t, s1_t1, s2_t, s2_t1 : listas de índices de variables en cada combinación
    """
    s1_t = sorted([v for v, tiempo in s1 if tiempo == 't'])
    s1_t1 = sorted([v for v, tiempo in s1 if tiempo == 't+1'])
    s2_t = sorted([v for v, tiempo in s2 if tiempo == 't'])
    s2_t1 = sorted([v for v, tiempo in s2 if tiempo == 't+1'])
    return s1_t, s1_t1, s2_t, s2_t1


# ---------------------------------------------------------------------------
# Evaluación de biparticiones
# ---------------------------------------------------------------------------

def evaluate_bipartition(
    tpm: np.ndarray,
    n: int,
    vars_s1: list[int],
    vars_s2: list[int],
    initial_state_idx: int,
) -> float:
    """
    Evalúa la discrepancia δ para una bipartición dada del subsistema.

    Usa la función tensor_product_tpm para reconstruir la TPM de la bipartición
    y luego calcula EMD con distancia Hamming.

    Parámetros
    ----------
    tpm              : TPM del subsistema (2^n, 2^n)
    n                : número de variables del subsistema
    vars_s1          : índices de variables en S1 (relativos al subsistema)
    vars_s2          : índices de variables en S2 (relativos al subsistema)
    initial_state_idx: índice del estado inicial en t

    Retorna
    -------
    delta : float, discrepancia EMD
    """
    from emd import emd, hamming_distance_matrix

    tpm_reconstructed = tensor_product_tpm(tpm, n, vars_s1, vars_s2)
    D = hamming_distance_matrix(n)
    p = tpm[initial_state_idx, :]
    q = tpm_reconstructed[initial_state_idx, :]
    return emd(p, q, D)


def find_optimal_bipartition(
    tpm: np.ndarray,
    n: int,
    initial_state_idx: int,
    variable_names: list[str] | None = None,
    verbose: bool = False,
) -> dict:
    """
    Encuentra la bipartición óptima del subsistema minimizando δ.

    Parámetros
    ----------
    tpm               : TPM del subsistema (2^n, 2^n)
    n                 : número de variables del subsistema
    initial_state_idx : índice del estado inicial
    variable_names    : nombres opcionales de las variables para el reporte
    verbose           : si True, imprime el progreso

    Retorna
    -------
    result : dict con keys:
        'optimal_s1'       : lista de índices de variables en S1
        'optimal_s2'       : lista de índices de variables en S2
        'min_delta'        : valor mínimo de δ
        'all_bipartitions' : lista de (s1, s2, delta) para todas las biparticiones
        'n_evaluated'      : número total de biparticiones evaluadas
    """
    if variable_names is None:
        variable_names = [f"X{i+1}" for i in range(n)]

    all_vars = list(range(n))
    best_s1 = None
    best_s2 = None
    best_delta = float("inf")
    all_results = []

    # Generar todas las biparticiones no triviales de las n variables
    # Nota: aquí biparticionamos las VARIABLES del subsistema (no los tiempos)
    # Una bipartición define qué variables van en S1 y cuáles en S2
    for r in range(1, n):
        for combo in combinations(all_vars, r):
            s1 = list(combo)
            s2 = [v for v in all_vars if v not in s1]

            # Evitar duplicados (S1={A,B}, S2={C} es la misma que S1={C}, S2={A,B})
            if s1[0] != 0:
                continue

            try:
                delta = evaluate_bipartition(tpm, n, s1, s2, initial_state_idx)
            except Exception as e:
                if verbose:
                    print(f"  Error en bipartición {s1} | {s2}: {e}")
                continue

            s1_names = [variable_names[i] for i in s1]
            s2_names = [variable_names[i] for i in s2]
            all_results.append((s1_names, s2_names, delta))

            if verbose:
                print(f"  {s1_names} | {s2_names}  →  δ = {delta:.6f}")

            if delta < best_delta:
                best_delta = delta
                best_s1 = s1
                best_s2 = s2

    best_s1_names = [variable_names[i] for i in best_s1] if best_s1 is not None else []
    best_s2_names = [variable_names[i] for i in best_s2] if best_s2 is not None else []

    return {
        "optimal_s1": best_s1_names,
        "optimal_s2": best_s2_names,
        "optimal_s1_idx": best_s1,
        "optimal_s2_idx": best_s2,
        "min_delta": best_delta,
        "all_bipartitions": all_results,
        "n_evaluated": len(all_results),
    }
