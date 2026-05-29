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
) -> tuple[list[int], list[int], list[int], list[int]]:
    """
    Separa una bipartición en sus componentes de t y t+1 para cada parte.

    Retorna
    -------
    s1_t, s1_t1, s2_t, s2_t1 : listas de índices de variables en cada combinación
    """
    s1_t  = sorted(v for v, tiempo in s1 if tiempo == 't')
    s1_t1 = sorted(v for v, tiempo in s1 if tiempo == 't+1')
    s2_t  = sorted(v for v, tiempo in s2 if tiempo == 't')
    s2_t1 = sorted(v for v, tiempo in s2 if tiempo == 't+1')
    return s1_t, s1_t1, s2_t, s2_t1


# ---------------------------------------------------------------------------
# Evaluación de biparticiones
# ---------------------------------------------------------------------------

def evaluate_bipartition(
    tpm: np.ndarray,
    n: int,
    s1_t: list[int],
    s1_t1: list[int],
    s2_t: list[int],
    s2_t1: list[int],
    initial_state_idx: int,
) -> float:
    """
    Evalúa la discrepancia δ para una bipartición dada del subsistema.

    Soporta biparticiones del espacio completo (t, t+1): las variables en t
    y en t+1 de cada parte pueden ser distintas.

    Parámetros
    ----------
    tpm              : TPM del subsistema (2^n, 2^n)
    n                : número de variables del subsistema
    s1_t             : índices de variables en t asignadas a S1
    s1_t1            : índices de variables en t+1 asignadas a S1
    s2_t             : índices de variables en t asignadas a S2
    s2_t1            : índices de variables en t+1 asignadas a S2
    initial_state_idx: índice del estado inicial en t

    Retorna
    -------
    delta : float, discrepancia EMD con métrica Hamming
    """
    from emd import emd, hamming_distance_matrix
    from state_node import tensor_product_tpm

    tpm_reconstructed = tensor_product_tpm(tpm, n, s1_t, s1_t1, s2_t, s2_t1)
    D = hamming_distance_matrix(n)
    p = tpm[initial_state_idx, :]
    q = tpm_reconstructed[initial_state_idx, :]
    return emd(p, q, D)


def _fmt_part(part: tuple, variable_names: list[str]) -> list[str]:
    """Convierte una parte de bipartición a etiquetas legibles, ej. ['A_t', 'B_t+1']."""
    return [f"{variable_names[v]}_{t}" for v, t in part]


def find_optimal_bipartition(
    tpm: np.ndarray,
    n: int,
    initial_state_idx: int,
    variable_names: list[str] | None = None,
    verbose: bool = False,
    max_bipartitions: int = 2000,
) -> dict:
    """
    Encuentra la bipartición óptima del subsistema minimizando δ.

    Enumera el espacio completo de biparticiones sobre (t, t+1):
        P = 2^(2n-1) - 1 biparticiones posibles.
    Si supera max_bipartitions, se trunca y se advierte en 'truncated'.

    Parámetros
    ----------
    tpm               : TPM del subsistema (2^n, 2^n)
    n                 : número de variables del subsistema
    initial_state_idx : índice del estado inicial
    variable_names    : nombres de las variables (len = n)
    verbose           : imprime progreso si True
    max_bipartitions  : límite de seguridad para no congelar el sistema

    Retorna
    -------
    dict con keys: optimal_s1, optimal_s2, min_delta,
                   all_bipartitions, n_evaluated, truncated
    """
    if variable_names is None:
        variable_names = [f"X{i+1}" for i in range(n)]

    all_var_time = build_bipartition_variables(n)   # 2n pares (var, tiempo)
    best_s1_label: list[str] = []
    best_s2_label: list[str] = []
    best_delta = float("inf")
    all_results: list[tuple] = []
    truncated = False

    for s1, s2 in generate_bipartitions(all_var_time):
        if len(all_results) >= max_bipartitions:
            truncated = True
            break

        s1_t, s1_t1, s2_t, s2_t1 = split_bipartition(s1, s2)

        try:
            delta = evaluate_bipartition(
                tpm, n, s1_t, s1_t1, s2_t, s2_t1, initial_state_idx
            )
        except Exception as exc:
            if verbose:
                print(f"  Error {s1} | {s2}: {exc}")
            continue

        s1_label = _fmt_part(s1, variable_names)
        s2_label = _fmt_part(s2, variable_names)
        all_results.append((s1_label, s2_label, delta))

        if verbose:
            print(f"  {s1_label} | {s2_label}  →  δ = {delta:.6f}")

        if delta < best_delta:
            best_delta = delta
            best_s1_label = s1_label
            best_s2_label = s2_label

    return {
        "optimal_s1": best_s1_label,
        "optimal_s2": best_s2_label,
        "min_delta": best_delta,
        "all_bipartitions": all_results,
        "n_evaluated": len(all_results),
        "truncated": truncated,
    }
