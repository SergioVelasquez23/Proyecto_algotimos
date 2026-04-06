"""
tpm_loader.py
-------------
Carga una TPM desde un archivo CSV en formato estado-estado.

Formato esperado del CSV:
- Sin encabezados (solo valores numéricos)
- Filas: 2^n estados en t
- Columnas: 2^n estados en t+1
- Cada fila debe sumar 1 (distribución de probabilidad)

Los estados se indexan en orden lexicográfico binario:
  fila/col 0 → estado 00..0
  fila/col 1 → estado 00..01
  ...
  fila/col 2^n-1 → estado 11..1
"""

import numpy as np
import csv
from pathlib import Path


def load_tpm(filepath: str) -> tuple[np.ndarray, int]:
    """
    Carga la TPM desde un CSV y devuelve (tpm, n_variables).

    Returns
    -------
    tpm : np.ndarray de forma (2^n, 2^n)
    n   : número de variables del sistema
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {filepath}")

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = []
        for line in reader:
            # Ignorar líneas vacías o comentarios
            stripped = [c.strip() for c in line]
            if not stripped or stripped[0].startswith("#"):
                continue
            try:
                rows.append([float(v) for v in stripped])
            except ValueError:
                # Encabezado textual, ignorar
                continue

    tpm = np.array(rows, dtype=float)

    n_rows, n_cols = tpm.shape
    if n_rows != n_cols:
        raise ValueError(
            f"La TPM debe ser cuadrada. Dimensiones encontradas: {n_rows}x{n_cols}"
        )

    # Verificar que sea potencia de 2
    import math
    if math.log2(n_rows) != int(math.log2(n_rows)):
        raise ValueError(
            f"El número de estados ({n_rows}) debe ser una potencia de 2."
        )

    n = int(math.log2(n_rows))

    # Validación probabilística: cada fila debe sumar ~1
    row_sums = tpm.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        bad = np.where(~np.isclose(row_sums, 1.0, atol=1e-6))[0]
        raise ValueError(
            f"Las filas {bad.tolist()} no suman 1. "
            f"Sumas: {row_sums[bad].tolist()}"
        )

    return tpm, n


def state_to_index(state: list[int]) -> int:
    """Convierte un estado binario [x1, x2, ..., xn] a índice entero (big-endian)."""
    result = 0
    for bit in state:
        result = (result << 1) | int(bit)
    return result


def index_to_state(idx: int, n: int) -> list[int]:
    """Convierte un índice entero a estado binario [x1, x2, ..., xn] de longitud n."""
    return [(idx >> (n - 1 - i)) & 1 for i in range(n)]


def print_tpm(tpm: np.ndarray, n: int, variables: list[str] | None = None) -> None:
    """Imprime la TPM de forma legible con etiquetas de estado."""
    if variables is None:
        variables = [f"X{i+1}" for i in range(n)]

    num_states = 2 ** n
    header_states = [index_to_state(j, n) for j in range(num_states)]
    header = "  ".join(
        "".join(str(b) for b in s) for s in header_states
    )
    print(f"{'t \\ t+1':>8}  {header}")
    print("-" * (8 + 3 * num_states))

    for i in range(num_states):
        row_label = "".join(str(b) for b in index_to_state(i, n))
        row_vals = "  ".join(f"{v:.2f}" for v in tpm[i])
        print(f"{row_label:>8}  {row_vals}")
