import json
import os
from pathlib import Path
import requests

import streamlit as st
import pandas as pd
import duckdb

from trace_viewer import render_trace_viewer, TRACE_VIEWER_ENABLED


st.set_page_config(page_title="RAG Experiments", layout="wide")

ONEDRIVE_DB_URL = os.getenv(
    "ONEDRIVE_DB_URL",
    "https://grupoarpada-my.sharepoint.com/:u:/p/pcuervo/IQAl8U_XCr2iSJSTQE4kHYwIAU_go9Hkiktksk4RsO2veXs?e=VICW8o",
)
ALWAYS_REFRESH_DB = os.getenv("ALWAYS_REFRESH_DB", "0") == "1"


def _inject_styles():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #f7f9fc 0%, #ffffff 35%, #f7f9fc 100%);
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1400px;
        }
        h1, h2, h3 {
            letter-spacing: -0.02em;
        }
        .section-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: #0f172a;
            margin-top: 0.25rem;
            margin-bottom: 0.35rem;
        }
        .section-caption {
            font-size: 0.9rem;
            color: #475569;
            margin-bottom: 0.5rem;
        }
        .pill {
            display: inline-block;
            padding: 0.25rem 0.6rem;
            border-radius: 999px;
            background: #eef2ff;
            color: #3730a3;
            font-size: 0.8rem;
            font-weight: 600;
            margin-right: 0.4rem;
        }
        .pill-muted {
            background: #f1f5f9;
            color: #64748b;
        }
        .metrics-wrap {
            display: flex;
            gap: 0.75rem;
            flex-wrap: wrap;
            margin-bottom: 0.5rem;
        }
        .metric-card {
            padding: 0.75rem 0.9rem;
            border-radius: 12px;
            border: 1px solid #e2e8f0;
            background: #ffffff;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
            min-width: 160px;
        }
        .metric-label {
            font-size: 0.78rem;
            color: #64748b;
        }
        .metric-value {
            font-size: 1.2rem;
            font-weight: 700;
            color: #0f172a;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _get_db_path():
    env_path = os.getenv("EXPERIMENTS_DB_PATH")
    if env_path:
        return Path(env_path)
    # Default: use dashboard-local data folder
    return Path(__file__).resolve().parent / "data" / "experiments.duckdb"


def _download_db_from_onedrive(target_path: Path, overwrite: bool = False) -> bool:
    if not ONEDRIVE_DB_URL:
        return False

    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists() and not overwrite:
        return True
    url = ONEDRIVE_DB_URL
    if "download=1" not in url:
        joiner = "&" if "?" in url else "?"
        url = f"{url}{joiner}download=1"

    try:
        with requests.get(url, stream=True, timeout=60) as resp:
            content_type = resp.headers.get("Content-Type", "")
            if resp.status_code != 200 or "text/html" in content_type:
                return False
            tmp_path = target_path.with_suffix(".tmp")
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            tmp_path.replace(target_path)
        return True
    except Exception:
        return False


def _fetch_df(query, params=None):
    db_path = _get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        if params is None:
            return con.execute(query).fetchdf()
        return con.execute(query, params).fetchdf()
    finally:
        con.close()


def _fetch_all(query, params=None):
    db_path = _get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        if params is None:
            return con.execute(query).fetchall()
        return con.execute(query, params).fetchall()
    finally:
        con.close()


def main():
    _inject_styles()
    st.markdown("## RAG Experimentos")
    st.caption(
        "Explora ejecuciones, compara metricas y revisa artefactos. "
        "Selecciona un run o una partida desde las tablas para ver su detalle."
    )
    db_path = _get_db_path()
    col1, col2 = st.columns([1, 1])
    with col1:
        refresh_db = st.button("Actualizar DB desde OneDrive")
    with col2:
        refresh = st.button("Refrescar datos")

    if ALWAYS_REFRESH_DB or refresh_db or not db_path.exists():
        ok = _download_db_from_onedrive(db_path, overwrite=True)
        if not ok:
            st.warning(
                "No se pudo descargar la base DuckDB desde OneDrive. "
                "Es posible que el enlace haya caducado o no tenga permisos. "
                "Actualiza ONEDRIVE_DB_URL o sube un enlace nuevo."
            )
            st.stop()
    if "runs_cache" not in st.session_state:
        st.session_state["runs_cache"] = None
    if refresh or refresh_db or ALWAYS_REFRESH_DB or st.session_state["runs_cache"] is None:
        runs_columns = [row[1] for row in _fetch_all("PRAGMA table_info('runs')")]
        select_model_context = (
            "model_context" if "model_context" in runs_columns else "NULL AS model_context"
        )
        select_model_bbdd = (
            "model_bbdd" if "model_bbdd" in runs_columns else "NULL AS model_bbdd"
        )
        select_vs_context = (
            "vector_store_context"
            if "vector_store_context" in runs_columns
            else "NULL AS vector_store_context"
        )
        select_vs_bbdd = (
            "vector_store_bbdd"
            if "vector_store_bbdd" in runs_columns
            else "NULL AS vector_store_bbdd"
        )
        runs = _fetch_df(
            f"""
            SELECT run_id, created_at, completed_at,
                   {select_model_context}, {select_model_bbdd},
                   {select_vs_context}, {select_vs_bbdd},
                   k_context, k_bbdd, concurrency, input_path, bbdd_obra_path, bbdd_estudios_path,
                   precision_semantica_mean, recall_semantico_mean, f1_semantico_mean,
                   f2_semantico_mean, num_queries
            FROM runs
            ORDER BY created_at DESC
            """
        )
        st.session_state["runs_cache"] = runs
    else:
        runs = st.session_state["runs_cache"]

    if runs.empty:
        st.info("No hay runs registrados en la base de datos.")
        return

    st.markdown('<div class="section-title">Runs</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">Vista general de ejecuciones recientes.</div>',
        unsafe_allow_html=True,
    )
    max_rows = st.selectbox(
        "Mostrar",
        options=[5, 10, 20, 50],
        index=2,
        help="Limita el número de runs mostrados para rendimiento.",
    )
    if "model_context" not in runs.columns:
        runs["model_context"] = None
    if "model_bbdd" not in runs.columns:
        runs["model_bbdd"] = None
    if "vector_store_context" not in runs.columns:
        runs["vector_store_context"] = None
    if "vector_store_bbdd" not in runs.columns:
        runs["vector_store_bbdd"] = None

    metric_cols = [
        "precision_semantica_mean",
        "recall_semantico_mean",
        "f1_semantico_mean",
        "f2_semantico_mean",
    ]
    def _obra_from_path(path_value: str) -> str:
        if not path_value:
            return ""
        # Normaliza separadores para que funcione en Windows/Linux
        text = str(path_value).replace("\\", "/")
        name = text.rstrip("/").split("/")[-1]
        if "." in name:
            name = name.rsplit(".", 1)[0]

        if name.startswith("benchmark_"):
            name = name[len("benchmark_"):]
        return name

    if "bbdd_obra_path" in runs.columns:
        runs["obra"] = runs["bbdd_obra_path"].apply(_obra_from_path)
    else:
        runs["obra"] = ""
    hidden_cols = [
        "completed_at",
        "k_context",
        "k_bbdd",
        "concurrency",
        "input_path",
        "bbdd_obra_path",
        "bbdd_estudios_path",
    ]
    ordered_cols = [
        "run_id",
        "created_at",
        "obra",
        "model_context",
        "model_bbdd",
        *metric_cols,
        "num_queries",
        *hidden_cols,
    ]
    runs_view = runs.loc[
        :,
        [c for c in ordered_cols if c in runs.columns and c not in hidden_cols],
    ].head(max_rows)

    styled = runs_view.style.background_gradient(
        cmap="RdYlGn",
        subset=metric_cols,
        vmin=0.0,
        vmax=1.0,
    )

    runs_selection = st.dataframe(
        styled,
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="runs_table",
    )

    selected_rows = None
    if hasattr(runs_selection, "selection"):
        selected_rows = runs_selection.selection.rows
    else:
        sel_state = st.session_state.get("runs_table", {})
        selected_rows = sel_state.get("selection", {}).get("rows")

    selected_run_id = None
    if selected_rows:
        row_idx = selected_rows[0]
        if row_idx < len(runs_view):
            selected_run_id = runs_view.iloc[row_idx]["run_id"]

    if not selected_run_id:
        if runs_view.empty:
            st.info("No hay runs visibles para seleccionar.")
            return
        selected_run_id = runs_view.iloc[0]["run_id"]

    run_details = runs.loc[runs["run_id"] == selected_run_id].iloc[0].to_dict()

    run_details["model_context"] = run_details.get("model_context")
    run_details["model_bbdd"] = run_details.get("model_bbdd")

    ordered_run_details = {}
    ordered_keys = [
        "run_id",
        "obra",
        "model_context",
        "model_bbdd",
        "vector_store_context",
        "vector_store_bbdd",
        "precision_semantica_mean",
        "recall_semantico_mean",
        "f1_semantico_mean",
        "f2_semantico_mean",
        "num_queries",
        "k_context",
        "k_bbdd",
        "concurrency",
        "input_path",
        "bbdd_obra_path",
        "bbdd_estudios_path",
        "created_at",
        "completed_at",
    ]
    for key in ordered_keys:
        if key in run_details:
            ordered_run_details[key] = run_details[key]
    for key, value in run_details.items():
        if key not in ordered_run_details:
            ordered_run_details[key] = value

    st.markdown('<div class="section-title">Detalle del run</div>', unsafe_allow_html=True)
    run_meta = []
    run_meta.append(f'<span class="pill">RUN {selected_run_id[:8]}</span>')
    if ordered_run_details.get("obra"):
        run_meta.append(f'<span class="pill pill-muted">{ordered_run_details["obra"]}</span>')
    st.markdown(" ".join(run_meta), unsafe_allow_html=True)

    metrics = [
        ("Precision", ordered_run_details.get("precision_semantica_mean")),
        ("Recall", ordered_run_details.get("recall_semantico_mean")),
        ("F1", ordered_run_details.get("f1_semantico_mean")),
        ("F2", ordered_run_details.get("f2_semantico_mean")),
    ]
    metric_cards = []
    for label, value in metrics:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            display = "n/a"
        else:
            display = f"{float(value):.3f}"
        metric_cards.append(
            f"<div class='metric-card'><div class='metric-label'>{label}</div>"
            f"<div class='metric-value'>{display}</div></div>"
        )
    if metric_cards:
        st.markdown(
            f"<div class='metrics-wrap'>{''.join(metric_cards)}</div>",
            unsafe_allow_html=True,
        )

    with st.expander("Ver detalles completos del run", expanded=False):
        st.json(ordered_run_details)

    run_artifacts = _fetch_all(
        "SELECT name, content FROM artifacts WHERE run_id = ? AND query_id IS NULL ORDER BY name",
        [selected_run_id],
    )

    if run_artifacts:
        st.markdown('<div class="section-title">Prompts del run</div>', unsafe_allow_html=True)
        for name, content in run_artifacts:
            with st.expander(name, expanded=False):
                st.text_area(name, value=content, height=240)

    st.divider()

    base_query_cols = """
        query_id, row_index, codigo, concepto, descripcion,
        precision_semantica, recall_semantico, f1_semantico, f2_semantico
    """
    json_query_cols = """
        codigos_predichos_json, ground_truth_json,
        conceptos_predichos_json, conceptos_ground_truth_json,
        missing_ground_truth_json, conceptos_missing_ground_truth_json
    """
    select_cols = base_query_cols + "," + json_query_cols

    queries = _fetch_df(
        f"""
        SELECT {select_cols}
        FROM queries
        WHERE run_id = ?
        ORDER BY row_index ASC
        """,
        [selected_run_id],
    )

    if queries.empty:
        st.info("Este run no tiene queries registradas.")
        return

    st.markdown('<div class="section-title">Partidas</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">Selecciona una fila para ver el detalle.</div>',
        unsafe_allow_html=True,
    )
    hidden_query_cols = [
        "query_id",
        "row_index",
        "codigos_predichos_json",
        "ground_truth_json",
        "conceptos_predichos_json",
        "conceptos_ground_truth_json",
        "conceptos_missing_ground_truth_json"
    ]
    queries_view = queries.loc[:, [c for c in queries.columns if c not in hidden_query_cols]]
    query_selection = st.dataframe(
        queries_view,
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="queries_table",
    )

    selected_q_rows = None
    if hasattr(query_selection, "selection"):
        selected_q_rows = query_selection.selection.rows
    else:
        sel_state = st.session_state.get("queries_table", {})
        selected_q_rows = sel_state.get("selection", {}).get("rows")

    selected_query_id = None
    if selected_q_rows:
        row_idx = selected_q_rows[0]
        if row_idx < len(queries):
            selected_query_id = queries.iloc[row_idx]["query_id"]

    if not selected_query_id:
        selected_query_id = queries.iloc[0]["query_id"]

    query_row = _fetch_df(
        "SELECT * FROM queries WHERE query_id = ?",
        [selected_query_id],
    ).iloc[0].to_dict()

    ordered_query_details = {}
    ordered_query_keys = [
        "query_id",
        "run_id",
        "capitulo",
        "codigo",
        "concepto",
        "descripcion",
        "precision_semantica",
        "recall_semantico",
        "f1_semantico",
        "f2_semantico",
        "codigos_predichos_json",
        "ground_truth_json",
        "conceptos_predichos_json",
        "conceptos_ground_truth_json",
        "missing_ground_truth_json",
        "conceptos_missing_ground_truth_json",
        "created_at",
        "updated_at",
        "row_index",
    ]
    for key in ordered_query_keys:
        if key in query_row:
            ordered_query_details[key] = query_row[key]
    for key, value in query_row.items():
        if key not in ordered_query_details:
            ordered_query_details[key] = value

    st.markdown('<div class="section-title">Detalle de la query</div>', unsafe_allow_html=True)
    with st.expander("Ver detalle completo de la query", expanded=False):
        st.json(ordered_query_details)

    artifacts = _fetch_all(
        "SELECT name, content FROM artifacts WHERE query_id = ? ORDER BY name",
        [selected_query_id],
    )

    if artifacts:
        st.markdown('<div class="section-title">Artefactos de la query</div>', unsafe_allow_html=True)
        for name, content in artifacts:
            with st.expander(name, expanded=False):
                if name in {"response_bbdd", "response_context"} and TRACE_VIEWER_ENABLED:
                    # Use the new Trace Viewer for response artifacts
                    try:
                        parsed = json.loads(content)
                        if (
                            isinstance(parsed, dict)
                            and "razonamiento" in parsed
                            and isinstance(parsed["razonamiento"], str)
                        ):
                            parsed["razonamiento"] = parsed["razonamiento"].replace("\\n", "\n")
                        render_trace_viewer(parsed, artifact_name=f"{selected_query_id}_{name}")
                    except json.JSONDecodeError:
                        st.text_area(name, value=content, height=320)
                    except Exception as e:
                        st.warning(f"Error al renderizar trace: {str(e)}")
                        st.text_area(name, value=content, height=320)
                elif name == "output":
                    # Show output as parsed JSON
                    try:
                        parsed = json.loads(content)
                        st.json(parsed, expanded=True)
                    except json.JSONDecodeError:
                        st.text_area(name, value=content, height=320)
                else:
                    st.text_area(name, value=content, height=320)


if __name__ == "__main__":
    main()
