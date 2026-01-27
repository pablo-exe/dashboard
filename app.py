import json
import os
from pathlib import Path
import requests

import streamlit as st
import pandas as pd
import duckdb


st.set_page_config(page_title="RAG Experiments", layout="wide")

ONEDRIVE_DB_URL = os.getenv(
    "ONEDRIVE_DB_URL",
    "https://grupoarpada-my.sharepoint.com/:u:/p/pcuervo/IQAl8U_XCr2iSJSTQE4kHYwIAU_go9Hkiktksk4RsO2veXs?e=VICW8o",
)
ALWAYS_REFRESH_DB = os.getenv("ALWAYS_REFRESH_DB", "0") == "1"


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


def _safe_str(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value)


def _run_label(row):
    created = _safe_str(row.get("created_at"))
    model = _safe_str(row.get("model"))
    short = _safe_str(row.get("run_id"))[:8]
    return f"{short} | {model} | {created}"


def _query_label(row):
    idx = row.get("row_index")
    codigo = _safe_str(row.get("codigo"))
    f2 = row.get("f2_semantico")
    f2_str = f"{f2:.3f}" if isinstance(f2, (int, float)) and not pd.isna(f2) else "n/a"
    return f"{idx} | {codigo} | f2={f2_str}"


def main():
    st.title("RAG Experimentos")
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
        runs = _fetch_df(
            """
            SELECT run_id, created_at, completed_at, model, k_context, k_bbdd, concurrency,
                   input_path, bbdd_obra_path, bbdd_estudios_path,
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

    st.subheader("Runs")
    max_rows = st.selectbox(
        "Mostrar",
        options=[50, 100, 200, 500, 1000],
        index=2,
        help="Limita el número de runs mostrados para rendimiento.",
    )

    metric_cols = [
        "precision_semantica_mean",
        "recall_semantico_mean",
        "f1_semantico_mean",
        "f2_semantico_mean",
    ]
    ordered_cols = [
        "run_id",
        "created_at",
        *metric_cols,
        "completed_at",
        "model",
        "k_context",
        "k_bbdd",
        "concurrency",
        "num_queries",
        "input_path",
        "bbdd_obra_path",
        "bbdd_estudios_path",
    ]
    runs_view = runs.loc[:, [c for c in ordered_cols if c in runs.columns]].head(max_rows)

    def _metric_style(val):
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return ""
        return ""

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

    run_labels = {row["run_id"]: _run_label(row) for _, row in runs.iterrows()}
    selected_run_id = st.selectbox(
        "Selecciona un run",
        runs["run_id"].tolist(),
        format_func=lambda rid: run_labels.get(rid, rid),
    )

    selected_rows = None
    if hasattr(runs_selection, "selection"):
        selected_rows = runs_selection.selection.rows
    else:
        sel_state = st.session_state.get("runs_table", {})
        selected_rows = sel_state.get("selection", {}).get("rows")

    if selected_rows:
        row_idx = selected_rows[0]
        if row_idx < len(runs_view):
            selected_run_id = runs_view.iloc[row_idx]["run_id"]

    run_details = runs.loc[runs["run_id"] == selected_run_id].iloc[0].to_dict()

    st.markdown("### Detalle del run")
    st.json(run_details)

    run_artifacts = _fetch_all(
        "SELECT name, content FROM artifacts WHERE run_id = ? AND query_id IS NULL ORDER BY name",
        [selected_run_id],
    )

    if run_artifacts:
        st.markdown("### Artefactos del run")
        for name, content in run_artifacts:
            with st.expander(name, expanded=False):
                st.text_area(name, value=content, height=240)

    show_json_cols = st.checkbox(
        "Mostrar columnas JSON en la tabla de queries",
        value=False,
        help="Incluye codigos/ground_truth/conceptos como JSON (puede ensanchar la tabla).",
    )

    base_query_cols = """
        query_id, row_index, codigo, concepto, descripcion,
        precision_semantica, recall_semantico, f1_semantico, f2_semantico
    """
    json_query_cols = """
        codigos_predichos_json, ground_truth_json,
        conceptos_predichos_json, conceptos_ground_truth_json,
        missing_ground_truth_json, conceptos_missing_ground_truth_json
    """
    select_cols = base_query_cols + ("," + json_query_cols if show_json_cols else "")

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

    st.subheader("Queries")
    query_selection = st.dataframe(
        queries,
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="queries_table",
    )

    query_labels = {row["query_id"]: _query_label(row) for _, row in queries.iterrows()}
    selected_query_id = st.selectbox(
        "Selecciona una query",
        queries["query_id"].tolist(),
        format_func=lambda qid: query_labels.get(qid, qid),
    )

    selected_q_rows = None
    if hasattr(query_selection, "selection"):
        selected_q_rows = query_selection.selection.rows
    else:
        sel_state = st.session_state.get("queries_table", {})
        selected_q_rows = sel_state.get("selection", {}).get("rows")

    if selected_q_rows:
        row_idx = selected_q_rows[0]
        if row_idx < len(queries):
            selected_query_id = queries.iloc[row_idx]["query_id"]

    query_row = _fetch_df(
        "SELECT * FROM queries WHERE query_id = ?",
        [selected_query_id],
    ).iloc[0].to_dict()

    st.markdown("### Detalle de la query")
    st.json(query_row)

    artifacts = _fetch_all(
        "SELECT name, content FROM artifacts WHERE query_id = ? ORDER BY name",
        [selected_query_id],
    )

    if artifacts:
        st.markdown("### Artefactos de la query")
        for name, content in artifacts:
            with st.expander(name, expanded=True):
                if name in {"response_bbdd", "response_context"}:
                    try:
                        parsed = json.loads(content)
                        st.json(parsed)
                    except Exception:
                        st.text_area(name, value=content, height=320)
                else:
                    st.text_area(name, value=content, height=320)


if __name__ == "__main__":
    main()
