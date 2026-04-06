"""
marginalization.py
------------------
Implementa las dos formas de marginalización sobre la TPM:

1. Marginalización en columnas (t+1):
   Elimina variables de fondo en t+1 sumando columnas con igual estado candidato.
   No requiere re-escalado.

2. Marginalización en filas (t):
   Elimina variables del sistema candidato en t descartando sus filas y promediando
   las filas duplicadas resultantes.
"""

import numpy as np
from itertools import product as iproduct
from tpm_loader import index_to_state, state_to_index


# ---------------------------------------------------------------------------
# Marginalización en columnas (t+1)
# ---------------------------------------------------------------------------

def marginalize_columns(
    tpm: np.ndarray,
    n_total: int,
    keep_vars_t1: list[int],
) -> np.ndarray:
    """
    Marginaliza la TPM eliminando columnas de las variables que NO están en
    keep_vars_t1. Agrupa (suma) las columnas que comparten el mismo estado
    proyectado sobre keep_vars_t1.

    Parámetros
    ----------
    tpm           : TPM de forma (n_rows, 2^n_total)
                    n_rows puede ser cualquier número de filas (ya condicionadas)
    n_total       : número de variables en t+1 representadas en las columnas
    keep_vars_t1  : índices (0-based) de las variables a mantener en t+1

    Retorna
    -------
    marginalized  : TPM de forma (n_rows, 2^len(keep_vars_t1))
    """
    n_keep = len(keep_vars_t1)
    n_rows = tpm.shape[0]
    n_out_cols = 2 ** n_keep
    marginalized = np.zeros((n_rows, n_out_cols), dtype=float)

    for col_idx in range(2 ** n_total):
        full_state = index_to_state(col_idx, n_total)
        # Proyectar solo las variables que se mantienen
        projected = [full_state[v] for v in keep_vars_t1]
        out_col = state_to_index(projected)
        marginalized[:, out_col] += tpm[:, col_idx]

    return marginalized


# ---------------------------------------------------------------------------
# Marginalización en filas (t)
# ---------------------------------------------------------------------------

def marginalize_rows(
    tpm: np.ndarray,
    n_candidate: int,
    keep_vars_t: list[int],
) -> np.ndarray:
    """
    Marginaliza la TPM eliminando variables de t que NO están en keep_vars_t.
    Las filas con igual estado proyectado se promedian.

    Parámetros
    ----------
    tpm           : TPM del sistema candidato de forma (2^n_candidate, n_cols)
    n_candidate   : número de variables del sistema candidato en t (filas)
    keep_vars_t   : índices relativos (dentro del sistema candidato) a mantener en t

    Retorna
    -------
    marginalized  : TPM de forma (2^len(keep_vars_t), n_cols)
    """
    n_keep = len(keep_vars_t)
    n_out_rows = 2 ** n_keep
    n_cols = tpm.shape[1]
    accumulated = np.zeros((n_out_rows, n_cols), dtype=float)
    counts = np.zeros(n_out_rows, dtype=int)

    for row_idx in range(2 ** n_candidate):
        full_state = index_to_state(row_idx, n_candidate)
        projected = [full_state[v] for v in keep_vars_t]
        out_row = state_to_index(projected)
        accumulated[out_row] += tpm[row_idx]
        counts[out_row] += 1

    # Promediar filas acumuladas
    for r in range(n_out_rows):
        if counts[r] > 0:
            accumulated[r] /= counts[r]

    return accumulated


# ---------------------------------------------------------------------------
# Utilidad: obtener la TPM del sistema candidato completo
# (marginalizar columnas de variables de fondo en t+1)
# ---------------------------------------------------------------------------

def get_candidate_tpm(
    conditioned_tpm: np.ndarray,
    n_total: int,
    candidate_vars: list[int],
) -> np.ndarray:
    """
    A partir de la TPM condicionada (filas ya filtradas), marginaliza las
    columnas para quedarse solo con las variables candidatas en t+1.

    Parámetros
    ----------
    conditioned_tpm : TPM condicionada (2^n_cand filas, 2^n_total columnas)
    n_total         : número total de variables en t+1 (columnas)
    candidate_vars  : índices de las variables candidatas en el sistema original

    Retorna
    -------
    candidate_tpm : TPM del sistema candidato (2^n_cand, 2^n_cand)
    """
    return marginalize_columns(conditioned_tpm, n_total, candidate_vars)
