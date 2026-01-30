"""
Trace Viewer Component for Streamlit
=====================================

A modern, interactive LLM trace visualization component that displays
OpenAI Responses API objects in a structured, user-friendly format.

Features:
- Timeline view of all trace steps
- Collapsible step details
- Filter toggles for step types
- Search within trace
- Copy functionality
- Raw JSON view with syntax highlighting
- Reasoning content protection (reveal toggle)
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Optional

import streamlit as st

from trace_parser import (
    TraceModel,
    TraceStep,
    StepKind,
    parse_response,
)


# Configuration flags - can be overridden via environment variables
TRACE_VIEWER_ENABLED = os.getenv("TRACE_VIEWER_ENABLED", "1") == "1"


def _get_step_icon(kind: StepKind) -> str:
    """Get an emoji icon for a step kind."""
    icons = {
        StepKind.REASONING: "🧠",
        StepKind.TOOL_CALL: "🔧",
        StepKind.TOOL_RESULT: "📥",
        StepKind.OUTPUT: "💬",
        StepKind.SYSTEM: "⚙️",
        StepKind.ERROR: "❌",
        StepKind.UNKNOWN: "❓",
    }
    return icons.get(kind, "•")


def _get_step_color(kind: StepKind) -> str:
    """Get a CSS color class for a step kind."""
    colors = {
        StepKind.REASONING: "#8b5cf6",  # Purple
        StepKind.TOOL_CALL: "#3b82f6",  # Blue
        StepKind.TOOL_RESULT: "#10b981",  # Green
        StepKind.OUTPUT: "#06b6d4",  # Cyan
        StepKind.SYSTEM: "#6b7280",  # Gray
        StepKind.ERROR: "#ef4444",  # Red
        StepKind.UNKNOWN: "#9ca3af",  # Light gray
    }
    return colors.get(kind, "#6b7280")


def _inject_trace_viewer_styles():
    """Inject custom CSS for the trace viewer."""
    st.markdown(
        """
        <style>
        .trace-viewer-container {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        
        .trace-meta-bar {
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
            padding: 0.75rem 1rem;
            background: #f8fafc;
            border-radius: 8px;
            margin-bottom: 1rem;
            border: 1px solid #e2e8f0;
        }
        
        .trace-meta-item {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.85rem;
        }
        
        .trace-meta-label {
            color: #64748b;
            font-weight: 500;
        }
        
        .trace-meta-value {
            color: #0f172a;
            font-weight: 600;
        }
        
        .trace-step-card {
            padding: 0.75rem 1rem;
            margin-bottom: 0.5rem;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
            background: #ffffff;
            transition: all 0.15s ease;
        }
        
        .trace-step-card:hover {
            border-color: #cbd5e1;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        .trace-step-header {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            cursor: pointer;
        }
        
        .trace-step-icon {
            font-size: 1.1rem;
        }
        
        .trace-step-title {
            font-weight: 600;
            color: #0f172a;
            font-size: 0.9rem;
        }
        
        .trace-step-badge {
            display: inline-block;
            padding: 0.15rem 0.5rem;
            border-radius: 999px;
            font-size: 0.7rem;
            font-weight: 600;
            text-transform: uppercase;
        }
        
        .trace-step-summary {
            color: #64748b;
            font-size: 0.85rem;
            margin-top: 0.25rem;
            line-height: 1.4;
        }
        
        .trace-step-content {
            margin-top: 0.75rem;
            padding-top: 0.75rem;
            border-top: 1px solid #f1f5f9;
        }
        
        .trace-output-text {
            background: #f8fafc;
            padding: 1rem;
            border-radius: 6px;
            font-family: ui-monospace, monospace;
            font-size: 0.85rem;
            white-space: pre-wrap;
            word-break: break-word;
            max-height: 400px;
            overflow-y: auto;
            border: 1px solid #e2e8f0;
        }
        
        .trace-tool-input {
            background: #eff6ff;
            padding: 0.75rem;
            border-radius: 6px;
            font-family: ui-monospace, monospace;
            font-size: 0.8rem;
            border: 1px solid #bfdbfe;
        }
        
        .trace-tool-output {
            background: #ecfdf5;
            padding: 0.75rem;
            border-radius: 6px;
            font-family: ui-monospace, monospace;
            font-size: 0.8rem;
            border: 1px solid #a7f3d0;
            max-height: 300px;
            overflow-y: auto;
        }
        
        .trace-error-box {
            background: #fef2f2;
            padding: 0.75rem 1rem;
            border-radius: 6px;
            border: 1px solid #fecaca;
            color: #991b1b;
        }
        
        .trace-filter-bar {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
            margin-bottom: 1rem;
        }
        
        .trace-filter-btn {
            padding: 0.35rem 0.75rem;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: 500;
            border: 1px solid #e2e8f0;
            background: #ffffff;
            cursor: pointer;
            transition: all 0.15s ease;
        }
        
        .trace-filter-btn:hover {
            background: #f8fafc;
        }
        
        .trace-filter-btn.active {
            background: #3b82f6;
            color: white;
            border-color: #3b82f6;
        }
        
        .trace-diagnostics {
            padding: 0.75rem 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        
        .trace-diagnostics-error {
            background: #fef2f2;
            border: 1px solid #fecaca;
            color: #991b1b;
        }
        
        .trace-diagnostics-warning {
            background: #fffbeb;
            border: 1px solid #fcd34d;
            color: #92400e;
        }
        
        .trace-final-output {
            background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #7dd3fc;
            margin-top: 1rem;
        }
        
        .trace-final-output-title {
            font-weight: 600;
            color: #0369a1;
            margin-bottom: 0.5rem;
            font-size: 0.9rem;
        }
        
        .trace-token-usage {
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
            padding: 0.5rem 0;
            font-size: 0.8rem;
            color: #64748b;
        }
        
        .trace-copy-btn {
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            background: #f1f5f9;
            border: 1px solid #e2e8f0;
            cursor: pointer;
            color: #475569;
        }
        
        .trace-copy-btn:hover {
            background: #e2e8f0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_meta_bar(trace: TraceModel):
    """Render the metadata bar at the top of the trace viewer."""
    meta = trace.meta
    
    html_parts = ['<div class="trace-meta-bar">']
    
    if meta.model:
        html_parts.append(
            f'<div class="trace-meta-item">'
            f'<span class="trace-meta-label">Model:</span>'
            f'<span class="trace-meta-value">{meta.model}</span>'
            f'</div>'
        )
    
    if meta.status:
        status_color = "#10b981" if meta.status == "completed" else "#f59e0b" if meta.status in ("incomplete", "in_progress") else "#ef4444"
        html_parts.append(
            f'<div class="trace-meta-item">'
            f'<span class="trace-meta-label">Status:</span>'
            f'<span class="trace-meta-value" style="color: {status_color}">{meta.status}</span>'
            f'</div>'
        )
    
    if meta.has_reasoning:
        html_parts.append(
            f'<div class="trace-meta-item">'
            f'<span class="trace-step-badge" style="background: #f3e8ff; color: #7c3aed;">REASONING</span>'
            f'</div>'
        )
    
    if meta.has_tools:
        html_parts.append(
            f'<div class="trace-meta-item">'
            f'<span class="trace-step-badge" style="background: #dbeafe; color: #2563eb;">TOOLS</span>'
            f'</div>'
        )
    
    # Token usage
    if trace.diagnostics.token_usage:
        usage = trace.diagnostics.token_usage
        tokens_str = []
        if usage.input_tokens:
            tokens_str.append(f"In: {usage.input_tokens:,}")
        if usage.output_tokens:
            tokens_str.append(f"Out: {usage.output_tokens:,}")
        if usage.reasoning_tokens:
            tokens_str.append(f"Reasoning: {usage.reasoning_tokens:,}")
        if tokens_str:
            html_parts.append(
                f'<div class="trace-meta-item">'
                f'<span class="trace-meta-label">Tokens:</span>'
                f'<span class="trace-meta-value">{" | ".join(tokens_str)}</span>'
                f'</div>'
            )
    
    html_parts.append('</div>')
    st.markdown("".join(html_parts), unsafe_allow_html=True)


def _render_diagnostics(trace: TraceModel):
    """Render any diagnostics (errors/warnings) at the top."""
    diag = trace.diagnostics
    
    if diag.errors:
        st.markdown(
            f'<div class="trace-diagnostics trace-diagnostics-error">'
            f'<strong>❌ Errors:</strong><br>'
            f'{"<br>".join(diag.errors)}'
            f'</div>',
            unsafe_allow_html=True,
        )
    
    if diag.warnings:
        st.markdown(
            f'<div class="trace-diagnostics trace-diagnostics-warning">'
            f'<strong>⚠️ Warnings:</strong><br>'
            f'{"<br>".join(diag.warnings)}'
            f'</div>',
            unsafe_allow_html=True,
        )


def _render_reasoning_step(step: TraceStep, reveal_key: str):
    """Render a reasoning step."""
    if step.raw_reasoning:
        st.markdown("**Razonamiento:**")
        st.text_area(
            "reasoning_content",
            value=step.raw_reasoning,
            height=min(400, max(100, step.raw_reasoning.count('\n') * 20 + 80)),
            label_visibility="collapsed",
            key=f"reasoning_{step.id}_{reveal_key}",
        )
    else:
        st.markdown(
            f'<div class="trace-step-summary">{step.summary}</div>',
            unsafe_allow_html=True,
        )


def _render_tool_call_step(step: TraceStep):
    """Render a tool call step."""
    st.markdown(
        f'<div class="trace-step-summary">{step.summary}</div>',
        unsafe_allow_html=True,
    )
    
    if step.tool_input:
        st.markdown("**Input:**")
        try:
            input_json = json.dumps(step.tool_input, indent=2, ensure_ascii=False)
            st.markdown(
                f'<div class="trace-tool-input"><pre>{input_json}</pre></div>',
                unsafe_allow_html=True,
            )
        except Exception:
            st.code(str(step.tool_input), language="json")


def _render_tool_result_step(step: TraceStep):
    """Render a tool result step."""
    st.markdown(
        f'<div class="trace-step-summary">{step.summary}</div>',
        unsafe_allow_html=True,
    )
    
    if step.tool_output is not None:
        st.markdown("**Output:**")
        try:
            if isinstance(step.tool_output, (dict, list)):
                # Render as expandable JSON
                st.json(step.tool_output, expanded=False)
            elif isinstance(step.tool_output, str):
                # Try to parse as JSON first
                try:
                    parsed = json.loads(step.tool_output)
                    st.json(parsed, expanded=False)
                except json.JSONDecodeError:
                    # Show as text with proper line breaks
                    st.text_area(
                        "tool_output",
                        value=step.tool_output,
                        height=min(400, max(100, step.tool_output.count('\n') * 20 + 80)),
                        label_visibility="collapsed",
                        key=f"tool_output_{step.id}",
                    )
            else:
                st.text_area(
                    "tool_output",
                    value=str(step.tool_output),
                    height=200,
                    label_visibility="collapsed",
                    key=f"tool_output_str_{step.id}",
                )
        except Exception as e:
            st.error(f"Error rendering output: {str(e)}")
            st.code(str(step.tool_output), language="text")


def _render_output_step(step: TraceStep):
    """Render an output step."""
    if step.output_text:
        # Try to parse as JSON first
        try:
            parsed = json.loads(step.output_text)
            st.json(parsed, expanded=True)
        except (json.JSONDecodeError, TypeError):
            # Show as text with proper formatting
            st.text_area(
                "output_text",
                value=step.output_text,
                height=min(400, max(100, step.output_text.count('\n') * 20 + 80)),
                label_visibility="collapsed",
                key=f"output_{step.id}",
            )


def _render_error_step(step: TraceStep):
    """Render an error step."""
    st.markdown(
        f'<div class="trace-error-box">'
        f'<strong>{step.error_code or "Error"}:</strong> {step.error_message or step.summary}'
        f'</div>',
        unsafe_allow_html=True,
    )


def _render_step(step: TraceStep, viewer_key: str):
    """Render a single trace step."""
    icon = _get_step_icon(step.kind)
    color = _get_step_color(step.kind)
    
    with st.expander(f"{icon} {step.title}", expanded=False):
        # Step badge
        st.markdown(
            f'<span class="trace-step-badge" style="background: {color}20; color: {color};">'
            f'{step.kind.value.upper()}'
            f'</span>',
            unsafe_allow_html=True,
        )
        
        # Render based on step kind
        if step.kind == StepKind.REASONING:
            _render_reasoning_step(step, f"{viewer_key}_reasoning_{step.id}")
        elif step.kind == StepKind.TOOL_CALL:
            _render_tool_call_step(step)
        elif step.kind == StepKind.TOOL_RESULT:
            _render_tool_result_step(step)
        elif step.kind == StepKind.OUTPUT:
            _render_output_step(step)
        elif step.kind == StepKind.ERROR:
            _render_error_step(step)
        else:
            # Unknown/system steps
            st.markdown(f"**Summary:** {step.summary}")
            if step.payload_preview:
                st.json(step.payload_preview)
        
        # Show raw path if available
        if step.payload_raw_path:
            st.caption(f"📍 Path: `{step.payload_raw_path}`")


def _render_final_outputs(trace: TraceModel):
    """Render the final outputs section."""
    if not trace.final_outputs:
        return
    
    st.markdown("### 📤 Output Final")
    
    for idx, output in enumerate(trace.final_outputs):
        role_label = f" ({output.role})" if output.role else ""
        format_label = f" [{output.format}]" if output.format else ""
        
        st.markdown(
            f'<div class="trace-final-output">'
            f'<div class="trace-final-output-title">Output {idx + 1}{role_label}{format_label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        
        # Try to parse as JSON first
        try:
            parsed = json.loads(output.text)
            st.json(parsed, expanded=True)
        except (json.JSONDecodeError, TypeError):
            # Display based on format
            if output.format == "codes":
                st.code(output.text, language="text")
            else:
                st.text_area(
                    f"output_{idx}",
                    value=output.text,
                    height=min(400, max(100, output.text.count('\n') * 20 + 80)),
                    label_visibility="collapsed",
                    key=f"final_output_{idx}_{hash(output.text[:50])}",
                )


def _render_trace_view(trace: TraceModel, viewer_key: str):
    """Render the main trace view."""
    # Metadata bar
    _render_meta_bar(trace)
    
    # Diagnostics
    _render_diagnostics(trace)
    
    # Filter controls
    st.markdown("#### 🔍 Filtros")
    filter_cols = st.columns(6)
    
    show_reasoning = filter_cols[0].checkbox("🧠 Reasoning", value=True, key=f"{viewer_key}_filter_reasoning")
    show_tools = filter_cols[1].checkbox("🔧 Tools", value=True, key=f"{viewer_key}_filter_tools")
    show_output = filter_cols[2].checkbox("💬 Output", value=True, key=f"{viewer_key}_filter_output")
    show_errors = filter_cols[3].checkbox("❌ Errors", value=True, key=f"{viewer_key}_filter_errors")
    
    # Search
    search_query = st.text_input(
        "🔎 Buscar en trace",
        key=f"{viewer_key}_search",
        placeholder="Buscar texto...",
    )
    
    # Filter steps
    filtered_steps = []
    for step in trace.steps:
        # Apply kind filters
        if step.kind == StepKind.REASONING and not show_reasoning:
            continue
        if step.kind in (StepKind.TOOL_CALL, StepKind.TOOL_RESULT) and not show_tools:
            continue
        if step.kind == StepKind.OUTPUT and not show_output:
            continue
        if step.kind == StepKind.ERROR and not show_errors:
            continue
        
        # Apply search filter
        if search_query:
            search_lower = search_query.lower()
            searchable = f"{step.title} {step.summary}".lower()
            if step.output_text:
                searchable += f" {step.output_text.lower()}"
            if step.tool_name:
                searchable += f" {step.tool_name.lower()}"
            if search_lower not in searchable:
                continue
        
        filtered_steps.append(step)
    
    # Step count
    st.caption(f"Mostrando {len(filtered_steps)} de {len(trace.steps)} pasos")
    
    # Render steps
    if not filtered_steps:
        st.info("No hay pasos que coincidan con los filtros.")
    else:
        st.markdown("#### 📋 Timeline")
        for step in filtered_steps:
            _render_step(step, viewer_key)
    
    # Final outputs
    _render_final_outputs(trace)


def _render_raw_json_view(trace: TraceModel, viewer_key: str):
    """Render the raw JSON view."""
    st.markdown("#### 📄 JSON Original")
    st.caption("Este es el JSON exacto almacenado, sin modificaciones.")
    
    # Copy button (using Streamlit's native support)
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("📋 Copiar", key=f"{viewer_key}_copy_json"):
            try:
                json_str = json.dumps(trace.raw, indent=2, ensure_ascii=False)
                st.session_state[f"{viewer_key}_copied"] = True
                st.toast("JSON copiado al portapapeles (usa Ctrl+C en el área de texto)")
            except Exception:
                pass
    
    # JSON display
    try:
        if trace.raw is not None:
            st.json(trace.raw, expanded=True)
        else:
            st.warning("No hay datos JSON disponibles.")
    except Exception as e:
        st.error(f"Error al mostrar JSON: {str(e)}")
        st.text_area(
            "Raw content",
            value=str(trace.raw),
            height=400,
            key=f"{viewer_key}_raw_text",
        )


def render_trace_viewer(
    data: Any,
    artifact_name: str = "response",
    default_tab: str = "trace",
) -> None:
    """
    Render the LLM Trace Viewer component.
    
    This is the main entry point for displaying trace data in Streamlit.
    It provides a tabbed interface with:
    - Trace view: Structured visualization of the response
    - Raw JSON view: Original JSON data
    
    Args:
        data: Raw response data (dict or JSON string)
        artifact_name: Name of the artifact (used for unique keys)
        default_tab: Which tab to show by default ("trace" or "json")
    """
    if not TRACE_VIEWER_ENABLED:
        # Fall back to simple JSON display
        try:
            if isinstance(data, str):
                data = json.loads(data)
            st.json(data, expanded=True)
        except Exception:
            st.text_area("Content", value=str(data), height=400)
        return
    
    # Inject styles
    _inject_trace_viewer_styles()
    
    # Parse the response
    trace = parse_response(data)
    
    # Generate unique key for this viewer
    viewer_key = f"trace_viewer_{artifact_name}_{hash(str(data)[:100]) % 10000}"
    
    # Tabs
    tab_trace, tab_json = st.tabs(["🔍 Trace", "📄 Raw JSON"])
    
    with tab_trace:
        if trace.has_errors() and not trace.steps:
            st.error("Error al parsear el response. Revisa la pestaña 'Raw JSON'.")
            for err in trace.diagnostics.errors:
                st.caption(f"• {err}")
        else:
            _render_trace_view(trace, viewer_key)
    
    with tab_json:
        _render_raw_json_view(trace, viewer_key)


def render_artifact_with_trace(
    name: str,
    content: str,
    use_trace_viewer: bool = True,
) -> None:
    """
    Render an artifact, using the trace viewer for supported types.
    
    This is a convenience function that determines whether to use the
    trace viewer based on the artifact name and content.
    
    Args:
        name: Artifact name (e.g., "response_bbdd", "response_context")
        content: Artifact content (JSON string)
        use_trace_viewer: Whether to try using the trace viewer
    """
    # List of artifact names that should use the trace viewer
    trace_viewer_artifacts = {"response_bbdd", "response_context", "output"}
    
    if use_trace_viewer and name in trace_viewer_artifacts and TRACE_VIEWER_ENABLED:
        try:
            parsed = json.loads(content)
            # Check if it looks like a response object
            if isinstance(parsed, dict):
                render_trace_viewer(parsed, artifact_name=name)
                return
        except json.JSONDecodeError:
            pass
    
    # Fall back to text area for non-trace artifacts
    st.text_area(name, value=content, height=320)
