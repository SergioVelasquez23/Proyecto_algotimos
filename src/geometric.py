"""
geometric.py
------------
Implementación del algoritmo geométrico (GeoMIP) para identificar
particiones óptimas usando la tabla de costos T.

Referencia: GeoMIP, Sección 4.2.4 — Identificación de Biparticiones Óptimas.

El algoritmo geométrico evita la búsqueda exhaustiva identificando
candidatos directamente desde la tabla T:

  1. Construir T para cada variable (ya implementado en hypercube.py)
  2. Para cada variable v, encontrar los estados j con T_v[s0, j] ≈ 0
     (costo nulo desde el estado inicial s0)
  3. Agrupar variables que comparten el mismo patrón de ceros → candidatos
  4. Evaluar candidatos con EMD exacto
  5. Retornar la partición con menor δ

Complejidad teórica: O(n · 2^n) para construir T vs O(2^(2n-1)) exhaustivo.
"""

import numpy as np
from typing import Iterator
from tpm_loader import index_to_state, state_to_index
from hypercube import compute_cost_table_from_tpm
from state_node import state_state_to_state_node
from kpartition import (
    build_partition_variables,
    split_kpartition,
    evaluate_kpartition,
    _fmt_kpart,
    generate_kpartitions,
)
from emd import emd, hamming_distance_matrix

_EPS = 1e-9


# ---------------------------------------------------------------------------
# Identificación geométrica de candidatos (GeoMIP Sec. 4.2.4)
# ---------------------------------------------------------------------------

def _compute_all_cost_tables(
    tpm: np.ndarray,
    n: int,
) -> list[np.ndarray]:
    """Calcula la tabla T para cada una de las n variables."""
    node_mats = state_state_to_state_node(tpm, n)
    return [compute_cost_table_from_tpm(M, n, v) for v, M in enumerate(node_mats)]


def _zero_pattern(T_v: np.ndarray, s0: int) -> frozenset[int]:
    """
    Retorna el conjunto de estados j (j ≠ s0) donde T_v[s0, j] ≈ 0.
    Si no hay ceros no triviales, retorna el estado con costo mínimo.
    """
    row = T_v[s0, :].copy()
    row[s0] = np.inf  # ignorar la diagonal
    zeros = frozenset(int(j) for j in np.where(row < _EPS)[0])
    if not zeros:
        # Sin ceros exactos: usar el mínimo (criterio robusto)
        zeros = frozenset([int(np.argmin(row))])
    return zeros


def identify_geometric_candidates(
    tpm: np.ndarray,
    n: int,
    initial_state_idx: int,
    k: int = 2,
    cost_tables: list[np.ndarray] | None = None,
) -> list[tuple[tuple, ...]]:
    """
    Identifica k-particiones candidatas usando la tabla de costos T.

    Algoritmo (GeoMIP Sección 4.2.4):
      - Para cada variable v, el patrón de estados con T_v[s0, j] = 0
        revela cuáles variables son "complementarias" (deben ir juntas).
      - Variables con el MISMO patrón de ceros → misma parte.
      - Cada asignación única de patrones → un candidato.

    Parámetros
    ----------
    tpm               : TPM del subsistema (2^n, 2^n)
    n                 : número de variables
    initial_state_idx : índice del estado inicial
    k                 : número de partes deseadas (k ≥ 2)
    cost_tables       : lista de n tablas T (se calculan si es None)

    Retorna
    -------
    Lista de k-particiones candidatas (sobre variable-time pairs)
    """
    if cost_tables is None:
        cost_tables = _compute_all_cost_tables(tpm, n)

    # Patrón de ceros por variable (variables en t+1)
    patterns = {}
    for v in range(n):
        pat = _zero_pattern(cost_tables[v], initial_state_idx)
        if pat not in patterns:
            patterns[pat] = []
        patterns[pat].append(v)

    unique_groups = list(patterns.values())  # grupos de variables por patrón

    # Si hay menos grupos que k, subdividir el grupo más grande
    while len(unique_groups) < k:
        largest = max(unique_groups, key=len)
        if len(largest) < 2:
            break
        idx = unique_groups.index(largest)
        mid = len(largest) // 2
        unique_groups[idx] = largest[:mid]
        unique_groups.append(largest[mid:])

    # Si hay más grupos que k, fusionar los más pequeños
    while len(unique_groups) > k:
        smallest = min(unique_groups, key=len)
        second = min((g for g in unique_groups if g is not smallest), key=len)
        unique_groups.remove(smallest)
        unique_groups.remove(second)
        unique_groups.append(smallest + second)

    # Construir la k-partición sobre el espacio variable-tiempo completo
    # Cada variable v en el grupo g → asigna (v, 't') y (v, 't+1') al mismo grupo
    all_var_time = build_partition_variables(n)
    group_for_var = {}
    for g_idx, group in enumerate(unique_groups):
        for v in group:
            group_for_var[v] = g_idx

    parts = [[] for _ in range(k)]
    for var_time in all_var_time:
        v, t = var_time
        g = group_for_var.get(v, 0)
        parts[g].append(var_time)

    candidate = tuple(tuple(p) for p in parts if p)
    if len(candidate) < 2:
        return []

    # Generar variantes adicionales intercambiando asignaciones de t vs t+1
    candidates = [candidate]
    _add_time_split_variants(candidates, unique_groups, n, k)

    return list(dict.fromkeys(candidates))  # eliminar duplicados preservando orden


def _add_time_split_variants(
    candidates: list,
    variable_groups: list[list[int]],
    n: int,
    k: int,
) -> None:
    """
    Agrega variantes donde las variables en t y t+1 no necesariamente
    están en el mismo grupo (exploración del espacio completo 2n).
    """
    # Para cada variable, probamos separarla en t vs t+1
    all_var_time = build_partition_variables(n)
    group_for_var = {}
    for g_idx, group in enumerate(variable_groups):
        for v in group:
            group_for_var[v] = g_idx

    # Variante: mover cada variable individual al grupo opuesto en t o t+1
    for v_flip in range(n):
        g_orig = group_for_var[v_flip]
        g_other = 1 - g_orig if k == 2 else (g_orig + 1) % k

        for time_flip in ['t', 't+1']:
            parts = [[] for _ in range(k)]
            for var_time in all_var_time:
                v, t = var_time
                if v == v_flip and t == time_flip:
                    parts[g_other].append(var_time)
                else:
                    parts[group_for_var.get(v, 0)].append(var_time)

            parts_nonempty = [p for p in parts if p]
            if len(parts_nonempty) == k:
                candidate = tuple(tuple(p) for p in parts_nonempty)
                if candidate not in candidates:
                    candidates.append(candidate)


# ---------------------------------------------------------------------------
# Búsqueda geométrica de la partición óptima
# ---------------------------------------------------------------------------

def find_optimal_kpartition_geometric(
    tpm: np.ndarray,
    n: int,
    k: int,
    initial_state_idx: int,
    variable_names: list[str] | None = None,
    top_candidates: int = 50,
    verbose: bool = False,
) -> dict:
    """
    Encuentra la k-partición óptima usando el algoritmo geométrico (GeoMIP).

    Pasos:
      1. Calcular tabla T para cada variable → O(n · 2^n)
      2. Identificar candidatos geométricos (patrones de ceros en T)
      3. Evaluar candidatos con EMD exacto
      4. Retornar el de menor δ

    Parámetros
    ----------
    tpm               : TPM del subsistema (2^n, 2^n)
    n                 : número de variables
    k                 : número de partes
    initial_state_idx : índice del estado inicial
    variable_names    : nombres de las n variables
    top_candidates    : máximo de candidatos a evaluar con EMD completo
    verbose           : imprime progreso si True

    Retorna
    -------
    dict con keys: k, optimal_partition, min_delta,
                   all_evaluated, n_candidates, method
    """
    if variable_names is None:
        variable_names = [f"X{i+1}" for i in range(n)]

    if verbose:
        print(f"  [GeoMIP] Calculando tablas T para {n} variables...")

    # Paso 1: calcular T
    cost_tables = _compute_all_cost_tables(tpm, n)

    # Paso 2: identificar candidatos
    candidates = identify_geometric_candidates(
        tpm, n, initial_state_idx, k=k, cost_tables=cost_tables
    )

    if not candidates:
        # Fallback: usar las primeras top_candidates del espacio exhaustivo
        all_var_time = build_partition_variables(n)
        candidates = list(generate_kpartitions(all_var_time, k))[:top_candidates]

    if verbose:
        print(f"  [GeoMIP] {len(candidates)} candidatos identificados")

    # Paso 3: evaluar con L1 (proxy rápido) y ordenar
    D = hamming_distance_matrix(n)
    p_orig = tpm[initial_state_idx, :]
    scored = []

    for candidate in candidates[:top_candidates]:
        parts_split = split_kpartition(candidate)
        try:
            delta = evaluate_kpartition(tpm, n, parts_split, initial_state_idx)
            part_label = _fmt_kpart(candidate, variable_names)
            scored.append((part_label, delta))
            if verbose:
                print(f"    {' | '.join(part_label)}  →  δ = {delta:.6f}")
        except Exception as exc:
            if verbose:
                print(f"    Error: {exc}")

    if not scored:
        return {
            "k": k,
            "optimal_partition": [],
            "min_delta": float("inf"),
            "all_evaluated": [],
            "n_candidates": 0,
            "method": "geometric",
        }

    best_label, best_delta = min(scored, key=lambda x: x[1])

    return {
        "k": k,
        "optimal_partition": best_label,
        "min_delta": best_delta,
        "all_evaluated": scored,
        "n_candidates": len(scored),
        "method": "geometric",
    }


def find_optimal_kpartition_geometric_any_k(
    tpm: np.ndarray,
    n: int,
    initial_state_idx: int,
    variable_names: list[str] | None = None,
    top_candidates: int = 50,
    verbose: bool = False,
) -> dict:
    """
    Aplica la búsqueda geométrica para todos los valores de k (2 a n)
    y retorna la mejor partición global.
    """
    if variable_names is None:
        variable_names = [f"X{i+1}" for i in range(n)]

    cost_tables = _compute_all_cost_tables(tpm, n)
    results_by_k = {}
    best_delta = float("inf")
    best_k = 2
    best_result = None

    for k in range(2, n + 1):
        if verbose:
            print(f"\n  [GeoMIP] k = {k}")

        # Reusar cost_tables ya calculadas
        candidates = identify_geometric_candidates(
            tpm, n, initial_state_idx, k=k, cost_tables=cost_tables
        )
        if not candidates:
            all_var_time = build_partition_variables(n)
            candidates = list(generate_kpartitions(all_var_time, k))[:top_candidates]

        scored = []
        for candidate in candidates[:top_candidates]:
            parts_split = split_kpartition(candidate)
            try:
                delta = evaluate_kpartition(tpm, n, parts_split, initial_state_idx)
                part_label = _fmt_kpart(candidate, variable_names)
                scored.append((part_label, delta))
            except Exception:
                pass

        if scored:
            best_label, best_delta_k = min(scored, key=lambda x: x[1])
            result = {
                "k": k,
                "optimal_partition": best_label,
                "min_delta": best_delta_k,
                "all_evaluated": scored,
                "n_candidates": len(scored),
                "method": "geometric",
            }
            results_by_k[k] = result
            if best_delta_k < best_delta:
                best_delta = best_delta_k
                best_k = k
                best_result = result

    return {
        "best_k": best_k,
        "best_result": best_result,
        "results_by_k": results_by_k,
        "overall_min_delta": best_delta,
    }
