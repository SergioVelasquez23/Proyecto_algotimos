"""
controllers/strategies/geometric.py
------------------------------------
Clase GeometricSIA — Estrategia geométrica para encontrar la k-partición óptima.

Implementa la interfaz requerida por el GeoMIP (Capítulo 5, Entregables):

    class GeometricSIA:
        def __init__(self, tpm, n, initial_state_idx, variable_names)
        def aplicar_estrategia(self, k=2) -> dict
        def calcular_transicion_coste(v, i, j) -> float
        def identificar_candidatos(k) -> list

Algoritmo (GeoMIP Algorithm 2):
  1. Descomponer el sistema en tensores elementales (matrices estado-nodo)
  2. Calcular la tabla de costos T para cada variable
  3. Identificar k-particiones candidatas usando los patrones de T
  4. Evaluar candidatos con EMD exacto
  5. Retornar la partición óptima
"""

import sys
import os

# Agregar el directorio src al path para importar los módulos del proyecto
_SRC = os.path.join(os.path.dirname(__file__), "..", "..")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np
from hypercube import compute_cost_table_from_tpm
from state_node import state_state_to_state_node
from geometric import (
    _compute_all_cost_tables,
    identify_geometric_candidates,
    find_optimal_kpartition_geometric,
    find_optimal_kpartition_geometric_any_k,
)
from kpartition import (
    build_partition_variables,
    split_kpartition,
    evaluate_kpartition,
    find_optimal_kpartition,
    _fmt_kpart,
    count_kpartitions,
)
from emd import hamming_distance_matrix


class GeometricSIA:
    """
    Estrategia geométrica para la búsqueda de la k-partición óptima.

    Implementa el enfoque topológico del GeoMIP usando la tabla de costos T
    para identificar candidatos sin enumerar exhaustivamente todo el espacio.

    Uso
    ---
    sia = GeometricSIA(tpm, n, initial_state_idx=4, variable_names=['A','B','C'])
    result = sia.aplicar_estrategia(k=2)
    print(result['optimal_partition'], result['min_delta'])
    """

    def __init__(
        self,
        tpm: np.ndarray,
        n: int,
        initial_state_idx: int,
        variable_names: list[str] | None = None,
    ):
        self.tpm = tpm
        self.n = n
        self.initial_state_idx = initial_state_idx
        self.variable_names = variable_names or [f"X{i+1}" for i in range(n)]

        # Descomponer en tensores elementales (matrices estado-nodo)
        self.node_matrices = state_state_to_state_node(tpm, n)

        # Calcular tabla de costos T para cada variable (O(n · 2^n))
        self.cost_tables = _compute_all_cost_tables(tpm, n)

    # ------------------------------------------------------------------
    # Método principal requerido por el entregable
    # ------------------------------------------------------------------

    def aplicar_estrategia(
        self,
        k: int = 2,
        top_candidates: int = 50,
        verbose: bool = False,
    ) -> dict:
        """
        Implementa el algoritmo geométrico para encontrar la k-partición óptima.

        Pasos (GeoMIP Algorithm 2):
          1. tensors ← ya calculados en __init__
          2. T ← ya calculada en __init__
          3. candidates ← identificarBiparticionesCandidatas(T)
          4. Bopt ← evaluarCandidatos(candidates)
          5. return Bopt

        Parámetros
        ----------
        k             : número de partes (2 = bipartición)
        top_candidates: máximo de candidatos a evaluar con EMD
        verbose       : mostrar progreso

        Retorna
        -------
        dict con: k, optimal_partition, min_delta, all_evaluated,
                  n_candidates, method='geometric'
        """
        # Paso 3: identificar candidatos geométricos
        candidates = identify_geometric_candidates(
            self.tpm,
            self.n,
            self.initial_state_idx,
            k=k,
            cost_tables=self.cost_tables,
        )

        if verbose:
            print(f"[GeometricSIA] k={k}: {len(candidates)} candidatos identificados")

        # Paso 4: evaluar candidatos con EMD exacto
        scored = []
        for candidate in candidates[:top_candidates]:
            parts_split = split_kpartition(candidate)
            try:
                delta = evaluate_kpartition(
                    self.tpm, self.n, parts_split, self.initial_state_idx
                )
                label = _fmt_kpart(candidate, self.variable_names)
                scored.append((label, delta))
                if verbose:
                    print(f"  {' | '.join(label)}  →  δ = {delta:.6f}")
            except Exception as exc:
                if verbose:
                    print(f"  Error: {exc}")

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

    def aplicar_estrategia_todos_k(
        self,
        top_candidates: int = 50,
        verbose: bool = False,
    ) -> dict:
        """
        Aplica la estrategia geométrica para todos los valores de k (2 a n)
        y retorna la mejor partición global.
        """
        results_by_k = {}
        best_delta = float("inf")
        best_k = 2
        best_result = None

        for k in range(2, self.n + 1):
            if verbose:
                print(f"\n[GeometricSIA] Evaluando k={k}...")
            result = self.aplicar_estrategia(k=k, top_candidates=top_candidates, verbose=verbose)
            results_by_k[k] = result

            if result["min_delta"] < best_delta:
                best_delta = result["min_delta"]
                best_k = k
                best_result = result

        return {
            "best_k": best_k,
            "best_result": best_result,
            "results_by_k": results_by_k,
            "overall_min_delta": best_delta,
        }

    # ------------------------------------------------------------------
    # Métodos auxiliares requeridos por el entregable (Sección 5.1.2)
    # ------------------------------------------------------------------

    def calcular_transicion_coste(self, v: int, i: int, j: int) -> float:
        """
        Calcula el costo de transición t(i, j) para la variable v.

        Parámetros
        ----------
        v : índice de la variable (0-based)
        i : estado inicial (entero)
        j : estado final (entero)

        Retorna
        -------
        float: T_v[i, j]
        """
        return float(self.cost_tables[v][i, j])

    def identificar_candidatos(self, k: int = 2) -> list:
        """
        Identifica k-particiones candidatas basadas en la tabla T.

        Retorna
        -------
        Lista de candidatos (k-particiones de variable-time pairs)
        """
        return identify_geometric_candidates(
            self.tpm,
            self.n,
            self.initial_state_idx,
            k=k,
            cost_tables=self.cost_tables,
        )

    def resumen(self) -> str:
        """Retorna un resumen del sistema configurado."""
        n_states = 2 ** self.n
        lines = [
            f"GeometricSIA — Sistema de {self.n} variables ({n_states} estados)",
            f"  Estado inicial: {self.initial_state_idx}",
            f"  Variables: {self.variable_names}",
            f"  Tabla T calculada para {len(self.cost_tables)} variables",
        ]
        for k in range(2, self.n + 1):
            from kpartition import count_kpartitions
            m = 2 * self.n
            s = count_kpartitions(m, k)
            lines.append(f"  k={k}: S({m},{k}) = {s:,} particiones posibles")
        return "\n".join(lines)
