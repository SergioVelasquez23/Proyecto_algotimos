"""
app.py
------
Interfaz gráfica web con Streamlit para visualizar el pipeline completo de
análisis de bipartición óptima en sistemas causales binarios.

Ejecutar con:
  py -m streamlit run app.py
"""

import sys
import os
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from tpm_loader import load_tpm, index_to_state, state_to_index
from conditioning import condition_tpm, get_background_state
from marginalization import get_candidate_tpm
from state_node import state_state_to_state_node
from hypercube import compute_cost_table_from_tpm, hamming_distance
from bipartition import find_optimal_bipartition
from kpartition import find_optimal_kpartition, find_optimal_partition_any_k, count_kpartitions, bell_number
from geometric import find_optimal_kpartition_geometric, find_optimal_kpartition_geometric_any_k


# ─────────────────────────────────────────────────────────────
# Configuración de página
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bipartición Óptima",
    page_icon="🔷",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔷 Análisis de Bipartición Óptima")
st.caption("Sistemas causales binarios · Análisis y Diseño de Algoritmos 2025C")


_COL_OPTIMA = "Óptima"

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def tpm_to_df(tpm: np.ndarray, n: int, var_names: list[str]) -> pd.DataFrame:
    num_states = 2 ** n
    labels = ["".join(str(b) for b in index_to_state(i, n)) for i in range(num_states)]
    return pd.DataFrame(tpm, index=labels, columns=labels)


def node_mat_to_df(mat: np.ndarray, n: int) -> pd.DataFrame:
    num_states = 2 ** n
    row_labels = ["".join(str(b) for b in index_to_state(i, n)) for i in range(num_states)]
    return pd.DataFrame(mat, index=row_labels, columns=["P(X=0|t)", "P(X=1|t)"])


def plot_tpm_heatmap(tpm: np.ndarray, n: int, title: str) -> go.Figure:
    num_states = 2 ** n
    labels = ["".join(str(b) for b in index_to_state(i, n)) for i in range(num_states)]
    fig = px.imshow(
        tpm,
        x=labels, y=labels,
        color_continuous_scale="Blues",
        zmin=0, zmax=1,
        labels={"x": "Estado t+1", "y": "Estado t", "color": "P"},
        title=title,
        text_auto=".2f",
    )
    fig.update_layout(height=420, margin=dict(t=40, b=20))
    return fig


def plot_cost_table(T: np.ndarray, n: int, var_name: str) -> go.Figure:
    num_states = 2 ** n
    labels = ["".join(str(b) for b in index_to_state(i, n)) for i in range(num_states)]
    fig = px.imshow(
        T,
        x=labels, y=labels,
        color_continuous_scale="Oranges",
        labels={"x": "j", "y": "i", "color": "t(i,j)"},
        title=f"Tabla de costos T — variable {var_name}",
        text_auto=".2f",
    )
    fig.update_layout(height=380, margin=dict(t=40, b=20))
    return fig


def plot_bipartitions_bar(results: list, optimal_s1: list) -> go.Figure:
    labels = [f"{s1} | {s2}" for s1, s2, _ in results]
    deltas = [d for _, _, d in results]
    colors = [
        "#2ecc71" if s1 == optimal_s1 else "#3498db"
        for s1, _, _ in results
    ]
    fig = go.Figure(go.Bar(
        x=labels,
        y=deltas,
        marker_color=colors,
        text=[f"{d:.4f}" for d in deltas],
        textposition="outside",
    ))
    fig.update_layout(
        title="Discrepancia δ por bipartición (verde = óptima)",
        xaxis_title="Bipartición S1 | S2",
        yaxis_title="δ (EMD)",
        height=380,
        margin=dict(t=50, b=60),
    )
    return fig


def plot_hypercube_3d(n: int, node_probs: np.ndarray, var_name: str) -> go.Figure | None:
    """Visualiza el hipercubo 3D para n=3, coloreando vértices por probabilidad."""
    if n != 3:
        return None

    coords = np.array([index_to_state(i, 3) for i in range(8)], dtype=float)
    labels = ["".join(str(b) for b in index_to_state(i, 3)) for i in range(8)]

    # Aristas del cubo
    edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            if bin(i ^ j).count("1") == 1:  # vecinos Hamming
                edges.append((i, j))

    edge_traces = []
    for i, j in edges:
        edge_traces.append(go.Scatter3d(
            x=[coords[i, 0], coords[j, 0], None],
            y=[coords[i, 1], coords[j, 1], None],
            z=[coords[i, 2], coords[j, 2], None],
            mode="lines",
            line=dict(color="lightgray", width=3),
            showlegend=False,
            hoverinfo="none",
        ))

    node_trace = go.Scatter3d(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
        mode="markers+text",
        marker=dict(
            size=14,
            color=node_probs,
            colorscale="RdYlGn",
            cmin=0, cmax=1,
            colorbar=dict(title=f"P({var_name}=1|t)", thickness=12),
            line=dict(width=1, color="gray"),
        ),
        text=labels,
        textposition="top center",
        textfont=dict(size=11),
        hovertemplate="Estado: %{text}<br>P(%{meta}=1|t) = %{marker.color:.3f}<extra></extra>",
        meta=[var_name] * 8,
        showlegend=False,
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        title=f"Hipercubo 3D — P({var_name}=1 | estado_t)",
        scene=dict(
            xaxis=dict(title="X1", range=[-0.3, 1.3]),
            yaxis=dict(title="X2", range=[-0.3, 1.3]),
            zaxis=dict(title="X3", range=[-0.3, 1.3]),
            aspectmode="cube",
        ),
        height=450,
        margin=dict(t=50),
    )
    return fig


# ─────────────────────────────────────────────────────────────
# Sidebar: Configuración de entrada
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuración")

    # Fuente de datos
    data_source = st.radio(
        "Fuente de datos",
        ["Demo (Example 1.2 del documento)", "Subir CSV propio"],
    )

    if data_source == "Demo (Example 1.2 del documento)":
        demo_mode = True
        csv_path = os.path.join(os.path.dirname(__file__), "data", "tpm_abcd.csv")
        var_names_input = "A B C D"
        initial_input = "1 0 0 0"
        background_input = "D"
        candidate_input = "A B C"
    else:
        demo_mode = False
        uploaded = st.file_uploader("Subir TPM (.csv)", type="csv")
        csv_path = None

        # Auto-detectar n desde el CSV subido para sugerir defaults correctos
        _n_detected = None
        if uploaded is not None:
            try:
                import math, io as _io
                _content = uploaded.read().decode("utf-8")
                uploaded.seek(0)  # reset para lectura posterior
                _rows = [l for l in _content.strip().splitlines()
                         if l.strip() and not l.strip().startswith("#")]
                _ncols = len(_rows[0].split(","))
                if _ncols > 0 and math.log2(_ncols) == int(math.log2(_ncols)):
                    _n_detected = int(math.log2(_ncols))
            except Exception:
                pass

        if _n_detected:
            _default_vars = " ".join([chr(65 + i) for i in range(_n_detected)])  # A B C ...
            _default_init = " ".join(["0"] * _n_detected)
        else:
            _default_vars = ""
            _default_init = ""

        var_names_input = st.text_input(
            "Variables (separadas por espacios)",
            value=_default_vars,
            placeholder="ej: A B C D",
        )
        initial_input = st.text_input(
            "Estado inicial (binario, uno por variable)",
            value=_default_init,
            placeholder="ej: 1 0 0 0",
        )
        background_input = st.text_input(
            "Variables de fondo (vacío = ninguna)",
            value="",
            placeholder="ej: D",
        )
        candidate_input = st.text_input(
            "Variables candidatas (vacío = automático)",
            value="",
            placeholder="vacío = todas menos el fondo",
        )

    st.divider()
    run_btn = st.button("▶ Ejecutar análisis", type="primary", width='stretch')


# ─────────────────────────────────────────────────────────────
# Lógica principal
# ─────────────────────────────────────────────────────────────
if run_btn or demo_mode:

    try:
        # Parsear inputs
        variable_names = var_names_input.strip().split()
        initial_state = list(map(int, initial_input.strip().split()))
        background_vars = background_input.strip().split() if background_input.strip() else []
        candidate_vars = candidate_input.strip().split() if candidate_input.strip() else None

        # Cargar CSV
        if not demo_mode and uploaded is not None:
            uploaded.seek(0)  # reset tras la lectura anticipada del sidebar
            content = uploaded.read().decode("utf-8")
            tmp_path = os.path.join(os.path.dirname(__file__), "_tmp_upload.csv")
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(content)
            csv_path = tmp_path
        elif not demo_mode and uploaded is None:
            st.warning("Sube un archivo CSV para continuar.")
            st.stop()

        # ──────────────────────────────────────────────────
        # PASO 1: Cargar TPM
        # ──────────────────────────────────────────────────
        tpm_full, n = load_tpm(csv_path)

        # Validar que los nombres coincidan con n
        if len(variable_names) != n:
            st.error(
                f"❌ El CSV tiene **{n} variables** ({2**n} estados) "
                f"pero se especificaron **{len(variable_names)} nombres**: `{variable_names}`. "
                f"Ajusta el campo 'Variables' para que tenga exactamente {n} nombres."
            )
            st.stop()

        # Validar estado inicial
        if len(initial_state) != n:
            st.error(
                f"❌ El estado inicial tiene {len(initial_state)} valores pero el sistema tiene {n} variables."
            )
            st.stop()

        # Validar que las variables de fondo existan en variable_names
        invalid_bg = [v for v in background_vars if v not in variable_names]
        if invalid_bg:
            st.error(f"❌ Variables de fondo no reconocidas: `{invalid_bg}`. Deben ser alguna de `{variable_names}`.")
            st.stop()

        # Validar que las variables candidatas existan
        if candidate_vars:
            invalid_cand = [v for v in candidate_vars if v not in variable_names]
            if invalid_cand:
                st.error(f"❌ Variables candidatas no reconocidas: `{invalid_cand}`. Deben ser alguna de `{variable_names}`.")
                st.stop()

        st.header("📥 Paso 1 — TPM Original")
        col1, col2, col3 = st.columns(3)
        col1.metric("Variables (n)", n)
        col2.metric("Estados", 2 ** n)
        col3.metric("Tamaño TPM", f"{2**n} × {2**n}")

        with st.expander("Ver TPM completa", expanded=False):
            st.plotly_chart(
                plot_tpm_heatmap(tpm_full, n, "TPM original (estado-estado)"),
                width='stretch',
            )
            st.dataframe(tpm_to_df(tpm_full, n, variable_names), width='stretch')

        # ──────────────────────────────────────────────────
        # PASO 2: Condicionamiento
        # ──────────────────────────────────────────────────
        st.header("🔒 Paso 2 — Condicionamiento (variables de fondo)")

        if background_vars:
            bg_indices = [variable_names.index(v) for v in background_vars]
            bg_values = get_background_state(initial_state, bg_indices)
            cond_info = {v: val for v, val in zip(background_vars, bg_values)}
            st.info(f"Variables de fondo fijadas: **{cond_info}**")

            conditioned_tpm, n_cand, cand_indices = condition_tpm(
                tpm_full, n, bg_indices, bg_values
            )
        else:
            st.info("Sin variables de fondo. Se usa el sistema completo.")
            cand_indices = list(range(n))
            n_cand = n
            conditioned_tpm = tpm_full

        if candidate_vars is None:
            candidate_vars = [variable_names[i] for i in cand_indices]

        cand_indices = [variable_names.index(v) for v in candidate_vars]
        n_c = len(candidate_vars)

        st.write(f"**Sistema candidato:** `{candidate_vars}`")
        st.write(f"TPM condicionada: `{conditioned_tpm.shape[0]} × {conditioned_tpm.shape[1]}`")

        # ──────────────────────────────────────────────────
        # PASO 3: Marginalización columnas t+1
        # ──────────────────────────────────────────────────
        st.header("📐 Paso 3 — TPM del sistema candidato")

        candidate_tpm = get_candidate_tpm(conditioned_tpm, n, cand_indices)

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                plot_tpm_heatmap(candidate_tpm, n_c, f"TPM candidata {candidate_vars}"),
                width='stretch',
            )
        with col2:
            st.dataframe(
                tpm_to_df(candidate_tpm, n_c, candidate_vars)
                .style.background_gradient(cmap="Blues", vmin=0, vmax=1)
                .format("{:.3f}"),
                width='stretch',
            )

        # ──────────────────────────────────────────────────
        # PASO 4: Estado-nodo
        # ──────────────────────────────────────────────────
        st.header("🔢 Paso 4 — Representación estado-nodo")

        node_matrices = state_state_to_state_node(candidate_tpm, n_c)

        tabs = st.tabs([f"Variable {v}" for v in candidate_vars])
        for i, (tab, var) in enumerate(zip(tabs, candidate_vars)):
            with tab:
                col1, col2 = st.columns(2)
                with col1:
                    df_node = node_mat_to_df(node_matrices[i], n_c)
                    st.dataframe(
                        df_node.style.background_gradient(cmap="RdYlGn", vmin=0, vmax=1)
                        .format("{:.3f}"),
                        width='stretch',
                    )
                with col2:
                    probs = node_matrices[i][:, 1]
                    labels = [
                        "".join(str(b) for b in index_to_state(j, n_c))
                        for j in range(2 ** n_c)
                    ]
                    fig_bar = px.bar(
                        x=labels, y=probs,
                        labels={"x": "Estado t", "y": f"P({var}=1|t)"},
                        title=f"P({var}_t+1 = 1 | estado_t)",
                        color=probs,
                        color_continuous_scale="RdYlGn",
                        range_color=[0, 1],
                    )
                    fig_bar.update_layout(height=300, showlegend=False)
                    st.plotly_chart(fig_bar, width='stretch')

        # ──────────────────────────────────────────────────
        # PASO 5: Hipercubo + Tabla T
        # ──────────────────────────────────────────────────
        st.header("🔷 Paso 5 — Hipercubo y Tabla de costos T")

        cost_tables = []
        tabs_t = st.tabs([f"Variable {v}" for v in candidate_vars])
        for i, (tab, var) in enumerate(zip(tabs_t, candidate_vars)):
            with tab:
                T = compute_cost_table_from_tpm(node_matrices[i], n_c, i)
                cost_tables.append(T)

                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(
                        plot_cost_table(T, n_c, var),
                        width='stretch',
                    )
                with col2:
                    # Hipercubo 3D solo para n=3
                    if n_c == 3:
                        fig_cube = plot_hypercube_3d(n_c, node_matrices[i][:, 1], var)
                        if fig_cube:
                            st.plotly_chart(fig_cube, width='stretch')
                    else:
                        st.info(
                            f"Visualización 3D disponible solo para n=3. "
                            f"Sistema actual: n={n_c}."
                        )
                        st.write(f"**Max T:** `{T.max():.4f}` | **Media T:** `{T.mean():.4f}`")

        # ──────────────────────────────────────────────────
        # PASO 6: Estado inicial en el sistema candidato
        # ──────────────────────────────────────────────────
        candidate_initial = [initial_state[variable_names.index(v)] for v in candidate_vars]
        initial_idx = state_to_index(candidate_initial)

        st.header("🎯 Paso 6 — Estado inicial del sistema candidato")
        cols = st.columns(len(candidate_vars) + 1)
        for ci, (cvar, cval) in enumerate(zip(candidate_vars, candidate_initial)):
            cols[ci].metric(f"{cvar}_t", cval)
        cols[-1].metric("Índice de fila", initial_idx)

        st.write(
            f"Distribución en t+1 dado estado inicial "
            f"`{''.join(map(str, candidate_initial))}`:"
        )
        dist_initial = candidate_tpm[initial_idx, :]
        labels = [
            "".join(str(b) for b in index_to_state(j, n_c))
            for j in range(2 ** n_c)
        ]
        fig_dist = px.bar(
            x=labels, y=dist_initial,
            labels={"x": "Estado t+1", "y": "Probabilidad"},
            title=f"P(V_t+1 | estado_t = {''.join(map(str, candidate_initial))})",
            color=dist_initial,
            color_continuous_scale="Blues",
        )
        fig_dist.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_dist, width='stretch')

        # ──────────────────────────────────────────────────
        # PASO 7: Biparticiones + EMD
        # ──────────────────────────────────────────────────
        st.header("⚖️ Paso 7 — Particiones y discrepancia δ")

        tab_bip, tab_kpart, tab_geo = st.tabs([
            "🔁 Biparticiones (fuerza bruta)",
            "🔀 K-Particiones (fuerza bruta)",
            "🌐 Solución Geométrica (GeoMIP)",
        ])

        MAX_PART = 2000

        # ──────────────────────────────────────────────────
        # TAB 1: Biparticiones (fuerza bruta, k=2)
        # ──────────────────────────────────────────────────
        with tab_bip:
            n_possible_full = 2 ** (2 * n_c - 1) - 1
            st.write(
                f"Espacio completo (t + t+1): **{n_possible_full}** biparticiones "
                f"— límite de seguridad: **{MAX_PART}**"
            )
            if n_possible_full > MAX_PART:
                st.warning(
                    f"{n_c} variables → {n_possible_full:,} biparticiones. "
                    f"Se evaluarán las primeras {MAX_PART}."
                )

            with st.spinner("Calculando δ para cada bipartición..."):
                result = find_optimal_bipartition(
                    candidate_tpm, n_c, initial_idx,
                    variable_names=candidate_vars,
                    verbose=False, max_bipartitions=MAX_PART,
                )

            if result.get("truncated"):
                st.info(f"⚠️ Truncado en {result['n_evaluated']} biparticiones.")

            st.plotly_chart(
                plot_bipartitions_bar(result["all_bipartitions"], result["optimal_s1"]),
                width='stretch',
            )
            df_bip = pd.DataFrame([
                {"S1": ", ".join(s1), "S2": ", ".join(s2),
                 "δ (EMD)": round(delta, 6),
                 _COL_OPTIMA: "✅" if s1 == result["optimal_s1"] else ""}
                for s1, s2, delta in sorted(result["all_bipartitions"], key=lambda x: x[2])
            ])
            st.dataframe(df_bip, width='stretch', hide_index=True)

            col1, col2, col3 = st.columns(3)
            col1.success(f"**S1** = `{', '.join(result['optimal_s1'])}`")
            col2.error(f"**S2** = `{', '.join(result['optimal_s2'])}`")
            col3.metric("δ mínimo", f"{result['min_delta']:.6f}")

        # ──────────────────────────────────────────────────
        # TAB 2: K-Particiones (fuerza bruta)
        # ──────────────────────────────────────────────────
        with tab_kpart:
            m = 2 * n_c
            b = bell_number(m)
            st.write(
                f"Número de Bell B({m}) = **{b:,}** particiones totales "
                f"(todas las k). Límite de seguridad: **{MAX_PART}**"
            )

            # Tabla de conteo por k
            count_rows = []
            for kk in range(2, n_c + 1):
                s_mk = count_kpartitions(m, kk)
                count_rows.append({"k": kk, "S(2n,k)": f"{s_mk:,}",
                                    "Evaluable": "✅" if s_mk <= MAX_PART else "⚠️ truncado"})
            st.dataframe(pd.DataFrame(count_rows), width='stretch', hide_index=True)

            k_sel = st.selectbox(
                "Número de partes k:", options=list(range(2, n_c + 1)), index=0,
                key="k_select"
            )
            buscar_todos_k = st.checkbox("Buscar la k óptima (todas las k)", value=False)

            if st.button("▶ Ejecutar k-partición", key="run_kpart"):
                with st.spinner(f"Calculando k={k_sel}-partición..."):
                    if buscar_todos_k:
                        kresult = find_optimal_partition_any_k(
                            candidate_tpm, n_c, initial_idx,
                            variable_names=candidate_vars,
                            max_partitions_total=MAX_PART,
                        )
                        st.success(f"Mejor k = **{kresult['best_k']}** con δ = **{kresult['overall_min_delta']:.6f}**")
                        for kk, kr in kresult["results_by_k"].items():
                            with st.expander(f"k={kk}  →  δ = {kr['min_delta']:.6f}"):
                                st.write("**Partición óptima:**", " | ".join(kr["optimal_partition"]))
                                df_kr = pd.DataFrame([
                                    {"Partición": " | ".join(lbl), "δ": round(d, 6)}
                                    for lbl, d in sorted(kr["all_partitions"], key=lambda x: x[1])
                                ])
                                st.dataframe(df_kr, width='stretch', hide_index=True)
                    else:
                        kresult = find_optimal_kpartition(
                            candidate_tpm, n_c, k_sel, initial_idx,
                            variable_names=candidate_vars,
                            max_partitions=MAX_PART,
                        )
                        if kresult.get("truncated"):
                            st.info(f"⚠️ Truncado en {kresult['n_evaluated']} particiones.")
                        st.success(f"Óptima: **{' | '.join(kresult['optimal_partition'])}** con δ = **{kresult['min_delta']:.6f}**")
                        df_kr = pd.DataFrame([
                            {"Partición": " | ".join(lbl), "δ": round(d, 6),
                             _COL_OPTIMA: "✅" if lbl == kresult["optimal_partition"] else ""}
                            for lbl, d in sorted(kresult["all_partitions"], key=lambda x: x[1])
                        ])
                        st.dataframe(df_kr, width='stretch', hide_index=True)

        # ──────────────────────────────────────────────────
        # TAB 3: Solución geométrica (GeoMIP Algorithm 2)
        # ──────────────────────────────────────────────────
        with tab_geo:
            st.write(
                "Usa la tabla de costos **T** para identificar candidatos directamente "
                "sin evaluar todo el espacio. Complejidad O(n·2ⁿ) vs O(2^(2n-1)) exhaustivo."
            )
            k_geo = st.selectbox(
                "Número de partes k:", options=list(range(2, n_c + 1)), index=0,
                key="k_geo_select"
            )
            buscar_todos_k_geo = st.checkbox("Buscar la k óptima (todas las k)", value=False, key="geo_all_k")

            if st.button("▶ Ejecutar solución geométrica", key="run_geo"):
                with st.spinner("Aplicando algoritmo geométrico (GeoMIP)..."):
                    if buscar_todos_k_geo:
                        geo_result = find_optimal_kpartition_geometric_any_k(
                            candidate_tpm, n_c, initial_idx,
                            variable_names=candidate_vars,
                        )
                        st.success(
                            f"Mejor k = **{geo_result['best_k']}** con δ = **{geo_result['overall_min_delta']:.6f}**"
                        )
                        for kk, kr in geo_result["results_by_k"].items():
                            with st.expander(f"k={kk}  →  δ = {kr['min_delta']:.6f}  ({kr['n_candidates']} candidatos)"):
                                st.write("**Partición óptima:**", " | ".join(kr["optimal_partition"]))
                                df_geo = pd.DataFrame([
                                    {"Candidato": " | ".join(lbl), "δ": round(d, 6)}
                                    for lbl, d in sorted(kr["all_evaluated"], key=lambda x: x[1])
                                ])
                                st.dataframe(df_geo, width='stretch', hide_index=True)
                    else:
                        geo_result = find_optimal_kpartition_geometric(
                            candidate_tpm, n_c, k_geo, initial_idx,
                            variable_names=candidate_vars,
                        )
                        st.info(f"{geo_result['n_candidates']} candidatos evaluados (método geométrico)")
                        st.success(
                            f"Óptima: **{' | '.join(geo_result['optimal_partition'])}** "
                            f"con δ = **{geo_result['min_delta']:.6f}**"
                        )
                        df_geo = pd.DataFrame([
                            {"Candidato": " | ".join(lbl), "δ": round(d, 6),
                             _COL_OPTIMA: "✅" if lbl == geo_result["optimal_partition"] else ""}
                            for lbl, d in sorted(geo_result["all_evaluated"], key=lambda x: x[1])
                        ])
                        st.dataframe(df_geo, width='stretch', hide_index=True)

        # ──────────────────────────────────────────────────
        # RESULTADO FINAL (bipartición de referencia)
        # ──────────────────────────────────────────────────
        st.divider()
        st.header("🏆 Resultado: Bipartición Óptima (k=2)")

        col1, col2, col3 = st.columns(3)
        col1.success(f"**S1** = `{', '.join(result['optimal_s1'])}`")
        col2.error(f"**S2** = `{', '.join(result['optimal_s2'])}`")
        col3.metric("δ mínimo", f"{result['min_delta']:.6f}")

        st.balloons()

    except Exception as e:
        st.error(f"**Error:** {e}")
        st.exception(e)

else:
    # Pantalla de bienvenida
    st.info("👈 Configura los parámetros en el panel izquierdo y pulsa **Ejecutar análisis**.")

    st.markdown("""
    ### ¿Qué hace este programa?

    Implementa el pipeline completo de análisis de bipartición óptima:

    | Paso | Descripción |
    |------|-------------|
    | 1 | Carga la **TPM** desde un CSV |
    | 2 | **Condiciona** las variables de fondo en t |
    | 3 | **Marginaliza** las columnas en t+1 |
    | 4 | Convierte a **estado-nodo** por variable |
    | 5 | Construye el **hipercubo** y la tabla de costos T (BFS) |
    | 6 | Identifica el **estado inicial** en el sistema candidato |
    | 7 | Evalúa todas las **biparticiones** con EMD + Hamming |
    | ★ | Retorna la **bipartición óptima** (δ mínimo) |

    ### Formato del CSV
    ```
    1,0,0,0,0,0,0,0
    0,0,0,0,0,0,1,0
    ...
    ```
    - Sin encabezados
    - Cada fila debe sumar 1
    - Tamaño: 2ⁿ × 2ⁿ
    """)
