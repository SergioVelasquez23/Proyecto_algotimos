"""
conditioning.py
---------------
Condicionamiento de la TPM: dado el estado inicial de las variables de fondo
(background conditions), filtra las filas de la TPM completa donde esas
variables tienen el valor especificado en t.

Ejemplo: sistema V = {A, B, C, D}, estado inicial t = [1,0,0,0]
Si D es variable de fondo con valor 0, se retienen solo las filas donde D_t = 0.
"""

import numpy as np
from tpm_loader import index_to_state, state_to_index


def condition_tpm(
    tpm: np.ndarray,
    n: int,
    background_vars: list[int],
    background_values: list[int],
) -> tuple[np.ndarray, int, list[int]]:
    """
    Condiciona la TPM fijando los valores de las variables de fondo en t.

    Parámetros
    ----------
    tpm               : TPM completa de forma (2^n, 2^n)
    n                 : número total de variables
    background_vars   : índices (0-based) de las variables de fondo
                        ej. [3] para D en {A,B,C,D}
    background_values : valores que toman esas variables en el estado inicial
                        ej. [0] si D_t = 0

    Retorna
    -------
    conditioned_tpm   : submatriz con las filas condicionadas
                        forma (2^(n - len(background_vars)), 2^n)
    n_candidate       : número de variables del sistema candidato (n - len(bg))
    candidate_vars    : índices de las variables candidatas en el sistema original
    """
    if len(background_vars) != len(background_values):
        raise ValueError(
            "background_vars y background_values deben tener la misma longitud."
        )

    candidate_vars = [i for i in range(n) if i not in background_vars]
    selected_rows = []

    for state_idx in range(2 ** n):
        state = index_to_state(state_idx, n)
        # Verificar que todas las variables de fondo toman su valor condicionado
        if all(state[bg_var] == bg_val
               for bg_var, bg_val in zip(background_vars, background_values)):
            selected_rows.append(state_idx)

    conditioned_tpm = tpm[selected_rows, :]  # filas filtradas, todas las columnas
    n_candidate = len(candidate_vars)

    return conditioned_tpm, n_candidate, candidate_vars


def get_background_state(initial_state: list[int], background_vars: list[int]) -> list[int]:
    """
    Extrae los valores de las variables de fondo desde el estado inicial completo.

    Parámetros
    ----------
    initial_state   : estado inicial completo del sistema, ej. [1,0,0,0]
    background_vars : índices de variables de fondo, ej. [3]

    Retorna
    -------
    Lista de valores de fondo, ej. [0]
    """
    return [initial_state[i] for i in background_vars]
