"""
main.py
-------
Pipeline completo de análisis de sistemas causales binarios.

Flujo:
  CSV → TPM completa
       ↓
  Condicionar (fondo)
       ↓
  Marginalizar columnas D_t+1
       ↓
  TPM del sistema candidato Vc
       ↓
  Convertir a estado-nodo (por variable)
       ↓
  Construir hipercubo + Tabla T (BFS)
       ↓
  Enumerar biparticiones → EMD por cada una
       ↓
  Bipartición óptima (δ mínimo)

Uso desde la terminal:
  python main.py --csv data/tpm_abcd.csv --vars A B C D --initial 1 0 0 0
                 --background D --candidate A B C
"""

import argparse
import sys
import os
import numpy as np

# Añadir src/ al path para imports relativos
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from tpm_loader import load_tpm, index_to_state, state_to_index, print_tpm
from conditioning import condition_tpm, get_background_state
from marginalization import get_candidate_tpm
from state_node import state_state_to_state_node
from hypercube import compute_cost_table_from_tpm, hamming_distance
from bipartition import find_optimal_bipartition


def parse_args():
    parser = argparse.ArgumentParser(
        description="Análisis de bipartición óptima en sistemas causales binarios."
    )
    parser.add_argument(
        "--csv", required=True,
        help="Ruta al archivo CSV con la TPM (ej. data/tpm_abcd.csv)"
    )
    parser.add_argument(
        "--vars", nargs="+", required=True,
        help="Nombres de las variables del sistema completo (ej. A B C D)"
    )
    parser.add_argument(
        "--initial", nargs="+", type=int, required=True,
        help="Estado inicial binario de cada variable (ej. 1 0 0 0)"
    )
    parser.add_argument(
        "--background", nargs="*", default=[],
        help="Variables de fondo (ej. D). Si se omite, se usa el sistema completo."
    )
    parser.add_argument(
        "--candidate", nargs="*", default=None,
        help="Variables del sistema candidato (ej. A B C). Por defecto: todas menos las de fondo."
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Mostrar detalles de cada bipartición evaluada."
    )
    return parser.parse_args()


def run_pipeline(
    csv_path: str,
    variable_names: list[str],
    initial_state: list[int],
    background_var_names: list[str],
    candidate_var_names: list[str] | None = None,
    verbose: bool = False,
):
    separator = "=" * 60

    # ------------------------------------------------------------------
    # PASO 1: Cargar TPM desde CSV
    # ------------------------------------------------------------------
    print(f"\n{separator}")
    print("PASO 1: Cargando TPM desde CSV")
    print(separator)
    tpm_full, n = load_tpm(csv_path)
    print(f"  Variables : {variable_names}")
    print(f"  n         : {n}")
    print(f"  TPM shape : {tpm_full.shape}")
    print(f"  Estado inicial: {dict(zip(variable_names, initial_state))}")

    # ------------------------------------------------------------------
    # PASO 2: Condicionar (variables de fondo en t)
    # ------------------------------------------------------------------
    print(f"\n{separator}")
    print("PASO 2: Condicionamiento (variables de fondo)")
    print(separator)

    if background_var_names:
        background_var_indices = [variable_names.index(v) for v in background_var_names]
        background_values = get_background_state(initial_state, background_var_indices)
        print(f"  Variables de fondo : {background_var_names}")
        print(f"  Valores de fondo   : {dict(zip(background_var_names, background_values))}")

        conditioned_tpm, n_cand, candidate_var_indices = condition_tpm(
            tpm_full, n, background_var_indices, background_values
        )
    else:
        print("  Sin variables de fondo (sistema completo)")
        candidate_var_indices = list(range(n))
        n_cand = n
        conditioned_tpm = tpm_full

    if candidate_var_names is None:
        candidate_var_names = [variable_names[i] for i in candidate_var_indices]

    candidate_var_indices = [variable_names.index(v) for v in candidate_var_names]
    print(f"  Sistema candidato  : {candidate_var_names} (índices {candidate_var_indices})")
    print(f"  TPM condicionada   : {conditioned_tpm.shape}")

    # ------------------------------------------------------------------
    # PASO 3: Marginalizar columnas en t+1 (eliminar variables de fondo)
    # ------------------------------------------------------------------
    print(f"\n{separator}")
    print("PASO 3: Marginalización de columnas (t+1)")
    print(separator)

    candidate_tpm = get_candidate_tpm(conditioned_tpm, n, candidate_var_indices)
    n_c = len(candidate_var_names)
    print(f"  TPM candidata shape : {candidate_tpm.shape}")
    print(f"  ({2**n_c} estados de {candidate_var_names})")

    # ------------------------------------------------------------------
    # PASO 4: Convertir a estado-nodo
    # ------------------------------------------------------------------
    print(f"\n{separator}")
    print("PASO 4: Representación estado-nodo (por variable)")
    print(separator)

    node_matrices = state_state_to_state_node(candidate_tpm, n_c)
    for i, (name, mat) in enumerate(zip(candidate_var_names, node_matrices)):
        print(f"  {name}_t+1 | estado_t  →  shape {mat.shape}")
        # Mostrar la columna P(Xi=1|t) de forma compacta
        probs_1 = mat[:, 1]
        print(f"    P({name}=1|t): {np.round(probs_1, 3).tolist()}")

    # ------------------------------------------------------------------
    # PASO 5: Construir hipercubo y tabla de costos T
    # ------------------------------------------------------------------
    print(f"\n{separator}")
    print("PASO 5: Hipercubo + Tabla de costos T (BFS)")
    print(separator)

    cost_tables = []
    for i, name in enumerate(candidate_var_names):
        node_mat = node_matrices[i]
        T = compute_cost_table_from_tpm(node_mat, n_c, i)
        cost_tables.append(T)
        print(f"  Tabla T para {name}: shape {T.shape}, max={T.max():.4f}, mean={T.mean():.4f}")

    # ------------------------------------------------------------------
    # PASO 6: Encontrar estado inicial en el sistema candidato
    # ------------------------------------------------------------------
    print(f"\n{separator}")
    print("PASO 6: Estado inicial en el sistema candidato")
    print(separator)

    candidate_initial = [initial_state[variable_names.index(v)] for v in candidate_var_names]
    initial_state_idx = state_to_index(candidate_initial)
    print(f"  Estado inicial candidato : {dict(zip(candidate_var_names, candidate_initial))}")
    print(f"  Índice                   : {initial_state_idx}")

    # ------------------------------------------------------------------
    # PASO 7: Enumerar biparticiones y calcular δ (EMD)
    # ------------------------------------------------------------------
    print(f"\n{separator}")
    print("PASO 7: Biparticiones → EMD (δ)")
    print(separator)

    from bipartition import find_optimal_bipartition
    n_possible = 2 ** (n_c - 1) - 1
    print(f"  Número de biparticiones a evaluar: {n_possible}")

    result = find_optimal_bipartition(
        candidate_tpm,
        n_c,
        initial_state_idx,
        variable_names=candidate_var_names,
        verbose=verbose,
    )

    # ------------------------------------------------------------------
    # RESULTADO FINAL
    # ------------------------------------------------------------------
    print(f"\n{separator}")
    print("RESULTADO: BIPARTICIÓN ÓPTIMA")
    print(separator)
    print(f"  S1 = {result['optimal_s1']}")
    print(f"  S2 = {result['optimal_s2']}")
    print(f"  δ mínimo = {result['min_delta']:.6f}")
    print(f"  Biparticiones evaluadas: {result['n_evaluated']}")

    if verbose:
        print(f"\n  Resumen de todas las biparticiones:")
        for s1, s2, delta in sorted(result["all_bipartitions"], key=lambda x: x[2]):
            marker = " ← ÓPTIMA" if s1 == result["optimal_s1"] else ""
            print(f"    {s1} | {s2}  →  δ = {delta:.6f}{marker}")

    return result


def demo():
    """
    Ejecuta el pipeline con los datos del Example 1.2 del documento:
    V = {A, B, C, D}, estado inicial = [1, 0, 0, 0]
    Sistema candidato = {A, B, C}, fondo = {D=0}
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "data", "tpm_abcd.csv")

    return run_pipeline(
        csv_path=csv_path,
        variable_names=["A", "B", "C", "D"],
        initial_state=[1, 0, 0, 0],
        background_var_names=["D"],
        candidate_var_names=["A", "B", "C"],
        verbose=True,
    )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Sin argumentos: ejecutar demo del documento
        print("Sin argumentos detectados. Ejecutando demo (Example 1.2)...\n")
        demo()
    else:
        args = parse_args()

        if len(args.vars) != len(args.initial):
            print("Error: --vars y --initial deben tener la misma longitud.")
            sys.exit(1)

        run_pipeline(
            csv_path=args.csv,
            variable_names=args.vars,
            initial_state=args.initial,
            background_var_names=args.background,
            candidate_var_names=args.candidate,
            verbose=args.verbose,
        )
