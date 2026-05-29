"""
kpartition.py
-------------
Enumeración y evaluación de k-particiones del subsistema candidato.

Una k-partición divide el conjunto de pares (variable, tiempo) en k partes
disjuntas y no vacías. Para k=2 coincide con las biparticiones de bipartition.py.

Número de k-particiones de un conjunto de m elementos = S(m, k) (número de Stirling).
Número total (todas las k) = B(m) (número de Bell).

Con m = 2n (n variables en t y t+1):
  S(6, 2) = 31  para n=3, k=2     → igual que biparticion.py
  S(6, 3) = 90  para n=3, k=3
  B(6)    = 203 para n=3 todas las k

La evaluación de cada k-partición usa el producto tensorial generalizado:
  P_rec(j | i) = ∏_{part in partición} P_part(proj_{part,t+1}(j) | proj_{part,t}(i))
"""

import numpy as np
from itertools import combinations
from typing import Iterator
from tpm_loader import index_to_state, state_to_index
from state_node import tensor_product_tpm
from emd import emd, hamming_distance_matrix


# ---------------------------------------------------------------------------
# Enumeración de k-particiones
# ---------------------------------------------------------------------------

def _generate_kpart_helper(
    elements: list,
    k: int,
    assignment: list,
    n_used: int,
) -> Iterator[list[list]]:
    """
    Genera recursivamente todas las k-particiones canónicas de elements.
    Canonical form: el primer elemento siempre va al grupo 0,
    y los nuevos grupos aparecen en orden estrictamente creciente.
    """
    pos = len(assignment)
    if pos == len(elements):
        if n_used == k:
            parts = [[] for _ in range(k)]
            for i, lbl in enumerate(assignment):
                parts[lbl].append(elements[i])
            yield [list(p) for p in parts]
        return

    # El elemento en 'pos' puede ir a cualquier grupo ya abierto (0..n_used-1)
    # o abrir uno nuevo (n_used), si aún hay grupos disponibles.
    max_label = min(n_used, k - 1)
    for lbl in range(max_label + 1):
        assignment.append(lbl)
        new_n_used = max(n_used, lbl + 1)
        yield from _generate_kpart_helper(elements, k, assignment, new_n_used)
        assignment.pop()


def generate_kpartitions(variables: list, k: int) -> Iterator[tuple[tuple, ...]]:
    """
    Genera todas las k-particiones no ordenadas de la lista de variables.

    El primer elemento siempre queda en la parte 0 para evitar duplicados
    simétricos (las partes no tienen etiqueta).

    Parámetros
    ----------
    variables : lista de identificadores de variables (pares (var_idx, tiempo))
    k         : número de partes (k ≥ 2)

    Yields
    ------
    partición : tupla de k tuplas (una por parte)
    """
    n = len(variables)
    if k < 2 or k > n:
        return
    # Fijamos variables[0] en parte 0; asignamos el resto recursivamente
    for partition in _generate_kpart_helper(list(variables), k, [0], 1):
        yield tuple(tuple(p) for p in partition)


def count_kpartitions(m: int, k: int) -> int:
    """Calcula el número de Stirling S(m, k) de segunda especie."""
    if k == 0:
        return 1 if m == 0 else 0
    if k > m:
        return 0
    # Tabla de programación dinámica
    dp = [[0] * (k + 1) for _ in range(m + 1)]
    dp[0][0] = 1
    for i in range(1, m + 1):
        for j in range(1, k + 1):
            dp[i][j] = j * dp[i - 1][j] + dp[i - 1][j - 1]
    return dp[m][k]


def bell_number(m: int) -> int:
    """Número de Bell B(m) = Σ_k S(m, k) = número total de particiones."""
    total = 0
    for k in range(1, m + 1):
        total += count_kpartitions(m, k)
    return total


# ---------------------------------------------------------------------------
# Construcción del espacio de variables-tiempo (igual que bipartition.py)
# ---------------------------------------------------------------------------

def build_partition_variables(n: int) -> list[tuple[int, str]]:
    """
    Construye la lista de 2n pares (variable_idx, tiempo).
    Ej. para n=3: [(0,'t'),(1,'t'),(2,'t'),(0,'t+1'),(1,'t+1'),(2,'t+1')]
    """
    return [(i, 't') for i in range(n)] + [(i, 't+1') for i in range(n)]


def split_kpartition(
    partition: tuple[tuple, ...],
) -> list[tuple[list[int], list[int]]]:
    """
    Separa cada parte de una k-partición en sus componentes de t y t+1.

    Retorna
    -------
    Lista de k elementos, cada uno (part_t, part_t1) con índices de variables.
    """
    result = []
    for part in partition:
        part_t  = sorted(v for v, tiempo in part if tiempo == 't')
        part_t1 = sorted(v for v, tiempo in part if tiempo == 't+1')
        result.append((part_t, part_t1))
    return result


# ---------------------------------------------------------------------------
# Evaluación de k-particiones
# ---------------------------------------------------------------------------

def reconstruct_kpartition_tpm(
    tpm: np.ndarray,
    n: int,
    parts_split: list[tuple[list[int], list[int]]],
) -> np.ndarray:
    """
    Reconstruye la TPM para una k-partición usando el producto tensorial generalizado.

    P_rec(j | i) = ∏_{p=1}^{k}  P_p(proj_{p,t+1}(j) | proj_{p,t}(i))

    Parámetros
    ----------
    tpm         : TPM del subsistema (2^n, 2^n)
    n           : número de variables
    parts_split : lista de k pares (part_t_vars, part_t1_vars)

    Retorna
    -------
    tpm_rec : np.ndarray (2^n, 2^n) reconstruida
    """
    from marginalization import marginalize_rows, marginalize_columns

    # Marginalizar cada parte a sus propias variables
    marginals = []
    for part_t, part_t1 in parts_split:
        tpm_p = marginalize_rows(tpm, n, part_t)
        tpm_p = marginalize_columns(tpm_p, n, part_t1)
        marginals.append((part_t, part_t1, tpm_p))

    num_states = 2 ** n
    tpm_rec = np.zeros((num_states, num_states), dtype=float)

    for i in range(num_states):
        full_i = index_to_state(i, n)
        for j in range(num_states):
            full_j = index_to_state(j, n)
            prob = 1.0
            for part_t, part_t1, tpm_p in marginals:
                row_p = state_to_index([full_i[v] for v in part_t])
                col_p = state_to_index([full_j[v] for v in part_t1])
                prob *= tpm_p[row_p, col_p]
            tpm_rec[i, j] = prob

    return tpm_rec


def evaluate_kpartition(
    tpm: np.ndarray,
    n: int,
    parts_split: list[tuple[list[int], list[int]]],
    initial_state_idx: int,
) -> float:
    """
    Calcula la discrepancia δ para una k-partición usando EMD con Hamming.

    Parámetros
    ----------
    tpm               : TPM del subsistema (2^n, 2^n)
    n                 : número de variables
    parts_split       : lista de k pares (part_t_vars, part_t1_vars)
    initial_state_idx : índice del estado inicial

    Retorna
    -------
    delta : float
    """
    tpm_rec = reconstruct_kpartition_tpm(tpm, n, parts_split)
    D = hamming_distance_matrix(n)
    p = tpm[initial_state_idx, :]
    q = tpm_rec[initial_state_idx, :]
    return emd(p, q, D)


def _fmt_kpart(partition: tuple[tuple, ...], variable_names: list[str]) -> list[str]:
    """Convierte una k-partición a lista de strings legibles."""
    return [
        "{" + ", ".join(f"{variable_names[v]}_{t}" for v, t in part) + "}"
        for part in partition
    ]


# ---------------------------------------------------------------------------
# Búsqueda exhaustiva de k-partición óptima
# ---------------------------------------------------------------------------

def find_optimal_kpartition(
    tpm: np.ndarray,
    n: int,
    k: int,
    initial_state_idx: int,
    variable_names: list[str] | None = None,
    verbose: bool = False,
    max_partitions: int = 2000,
) -> dict:
    """
    Encuentra la k-partición óptima del subsistema minimizando δ.

    Parámetros
    ----------
    tpm               : TPM del subsistema (2^n, 2^n)
    n                 : número de variables
    k                 : número de partes (k=2 equivale a bipartición)
    initial_state_idx : índice del estado inicial
    variable_names    : nombres de las variables (len = n)
    verbose           : imprime progreso si True
    max_partitions    : límite de seguridad para no congelar el sistema

    Retorna
    -------
    dict con keys: k, optimal_partition, min_delta,
                   all_partitions, n_evaluated, truncated
    """
    if variable_names is None:
        variable_names = [f"X{i+1}" for i in range(n)]

    all_var_time = build_partition_variables(n)
    best_partition_label: list[str] = []
    best_delta = float("inf")
    all_results: list[tuple] = []
    truncated = False

    for partition in generate_kpartitions(all_var_time, k):
        if len(all_results) >= max_partitions:
            truncated = True
            break

        parts_split = split_kpartition(partition)

        try:
            delta = evaluate_kpartition(tpm, n, parts_split, initial_state_idx)
        except Exception as exc:
            if verbose:
                print(f"  Error {partition}: {exc}")
            continue

        part_label = _fmt_kpart(partition, variable_names)
        all_results.append((part_label, delta))

        if verbose:
            print(f"  {' | '.join(part_label)}  →  δ = {delta:.6f}")

        if delta < best_delta:
            best_delta = delta
            best_partition_label = part_label

    return {
        "k": k,
        "optimal_partition": best_partition_label,
        "min_delta": best_delta,
        "all_partitions": all_results,
        "n_evaluated": len(all_results),
        "truncated": truncated,
    }


def find_optimal_partition_any_k(
    tpm: np.ndarray,
    n: int,
    initial_state_idx: int,
    variable_names: list[str] | None = None,
    max_partitions_total: int = 2000,
) -> dict:
    """
    Busca la partición óptima para todos los valores de k de 2 a n.

    Retorna
    -------
    dict con keys: best_k, best_result, results_by_k
    """
    if variable_names is None:
        variable_names = [f"X{i+1}" for i in range(n)]

    results_by_k = {}
    overall_best_delta = float("inf")
    overall_best_k = 2
    overall_best_result = None
    remaining = max_partitions_total

    for k in range(2, n + 1):
        if remaining <= 0:
            break
        result = find_optimal_kpartition(
            tpm, n, k, initial_state_idx,
            variable_names=variable_names,
            max_partitions=remaining,
        )
        results_by_k[k] = result
        remaining -= result["n_evaluated"]

        if result["min_delta"] < overall_best_delta:
            overall_best_delta = result["min_delta"]
            overall_best_k = k
            overall_best_result = result

    return {
        "best_k": overall_best_k,
        "best_result": overall_best_result,
        "results_by_k": results_by_k,
        "overall_min_delta": overall_best_delta,
    }
