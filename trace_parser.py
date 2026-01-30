"""
Trace Parser Module
====================

Parses OpenAI Responses API JSON objects into a normalized TraceModel structure.
This enables the LLM Trace Viewer to display model behavior in a structured way.

The parser is tolerant to schema variations and never throws on parse errors.
Instead, it returns diagnostics with warnings and a minimal trace.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class StepKind(Enum):
    """Types of steps that can appear in a trace."""
    REASONING = "reasoning"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    OUTPUT = "output"
    SYSTEM = "system"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass
class TraceStep:
    """Represents a single step in the trace timeline."""
    id: str
    index: int
    kind: StepKind
    title: str
    summary: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    source_node_refs: list[str] = field(default_factory=list)
    payload_preview: Optional[dict[str, Any]] = None
    payload_raw_path: Optional[str] = None
    # For reasoning steps - raw content that may be sensitive
    raw_reasoning: Optional[str] = None
    # For tool calls
    tool_name: Optional[str] = None
    tool_input: Optional[dict[str, Any]] = None
    # For tool results
    tool_output: Optional[Any] = None
    # For output steps
    output_text: Optional[str] = None
    output_format: Optional[str] = None
    # Error details
    error_message: Optional[str] = None
    error_code: Optional[str] = None


@dataclass
class FinalOutput:
    """Represents a final output from the model."""
    text: str
    format: Optional[str] = None
    output_index: int = 0
    role: Optional[str] = None


@dataclass
class TokenUsage:
    """Token usage statistics."""
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None


@dataclass
class Diagnostics:
    """Diagnostics information about the parsing process."""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    incomplete: bool = False
    token_usage: Optional[TokenUsage] = None
    finish_reason: Optional[str] = None


@dataclass
class TraceMeta:
    """Metadata about the trace."""
    model: Optional[str] = None
    created_at: Optional[str] = None
    response_id: Optional[str] = None
    status: str = "unknown"
    has_tools: bool = False
    has_reasoning: bool = False
    object_type: Optional[str] = None
    # For chat completions
    system_fingerprint: Optional[str] = None


@dataclass
class TraceModel:
    """
    Normalized representation of an LLM response trace.
    
    This is the internal contract used by the TraceViewer component.
    The raw JSON is preserved as the source of truth.
    """
    meta: TraceMeta
    steps: list[TraceStep]
    final_outputs: list[FinalOutput]
    diagnostics: Diagnostics
    raw: Any  # Exact raw JSON (reference)

    def has_errors(self) -> bool:
        """Check if the trace has any errors."""
        return len(self.diagnostics.errors) > 0 or any(
            s.kind == StepKind.ERROR for s in self.steps
        )

    def get_steps_by_kind(self, kind: StepKind) -> list[TraceStep]:
        """Filter steps by kind."""
        return [s for s in self.steps if s.kind == kind]

    def get_reasoning_steps(self) -> list[TraceStep]:
        """Get all reasoning steps."""
        return self.get_steps_by_kind(StepKind.REASONING)

    def get_tool_steps(self) -> list[TraceStep]:
        """Get all tool-related steps (calls and results)."""
        return [
            s for s in self.steps 
            if s.kind in (StepKind.TOOL_CALL, StepKind.TOOL_RESULT)
        ]


def _generate_step_id(index: int, kind: str, content: str = "") -> str:
    """Generate a unique step ID."""
    hash_input = f"{index}:{kind}:{content[:50]}"
    return hashlib.md5(hash_input.encode()).hexdigest()[:12]


def _safe_get(obj: Any, *keys: str, default: Any = None) -> Any:
    """Safely get nested keys from a dict."""
    current = obj
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key, default)
        else:
            return default
        if current is None:
            return default
    return current


def _format_timestamp(ts: Any) -> Optional[str]:
    """Format a timestamp to ISO string."""
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        try:
            return datetime.fromtimestamp(ts).isoformat()
        except (ValueError, OSError):
            return str(ts)
    return str(ts)


def _extract_reasoning_summary(reasoning_text: str) -> str:
    """Extract a basic summary from reasoning text."""
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        try:
            return datetime.fromtimestamp(ts).isoformat()
        except (ValueError, OSError):
            return str(ts)
    return str(ts)


def _extract_reasoning_summary(reasoning_text: str) -> str:
    """Extract a basic summary from reasoning text."""
    if not reasoning_text:
        return "Reasoning content available"
    
    lines = reasoning_text.strip().split('\n')
    return f"Razonamiento ({len(lines)} líneas, {len(reasoning_text)} caracteres)"


def _parse_openai_responses_api(data: dict, diagnostics: Diagnostics) -> tuple[TraceMeta, list[TraceStep], list[FinalOutput]]:
    """
    Parse OpenAI Responses API format.
    
    This handles the newer Responses API structure with:
    - output array containing message, reasoning, tool_call items
    - tool results in separate items
    """
    meta = TraceMeta()
    steps: list[TraceStep] = []
    final_outputs: list[FinalOutput] = []
    step_index = 0

    # Extract metadata
    meta.model = _safe_get(data, "model")
    meta.response_id = _safe_get(data, "id")
    meta.object_type = _safe_get(data, "object")
    meta.created_at = _format_timestamp(_safe_get(data, "created_at"))
    meta.status = _safe_get(data, "status", default="unknown")
    
    # Token usage
    usage = _safe_get(data, "usage")
    if usage:
        diagnostics.token_usage = TokenUsage(
            input_tokens=_safe_get(usage, "input_tokens"),
            output_tokens=_safe_get(usage, "output_tokens"),
            total_tokens=_safe_get(usage, "total_tokens"),
            reasoning_tokens=_safe_get(usage, "output_tokens_details", "reasoning_tokens"),
        )

    # Check for incomplete/error status
    if meta.status in ("incomplete", "failed", "cancelled"):
        diagnostics.incomplete = True
        incomplete_details = _safe_get(data, "incomplete_details")
        if incomplete_details:
            reason = _safe_get(incomplete_details, "reason", default="unknown")
            diagnostics.warnings.append(f"Response incomplete: {reason}")

    # Parse output array
    output_items = _safe_get(data, "output", default=[])
    if not isinstance(output_items, list):
        output_items = [output_items] if output_items else []

    for item_idx, item in enumerate(output_items):
        if not isinstance(item, dict):
            continue

        item_type = _safe_get(item, "type", default="unknown")
        item_id = _safe_get(item, "id", default=f"item_{item_idx}")

        if item_type == "reasoning":
            meta.has_reasoning = True
            # Handle reasoning content
            content_list = _safe_get(item, "content", default=[])
            reasoning_text = ""
            for content_item in content_list:
                if _safe_get(content_item, "type") == "reasoning_text":
                    reasoning_text += _safe_get(content_item, "text", default="")

            summary_text = _safe_get(item, "summary", default=[])
            summary_str = ""
            if isinstance(summary_text, list):
                for s in summary_text:
                    if isinstance(s, dict):
                        summary_str += _safe_get(s, "text", default="")
                    elif isinstance(s, str):
                        summary_str += s

            step = TraceStep(
                id=_generate_step_id(step_index, "reasoning", item_id),
                index=step_index,
                kind=StepKind.REASONING,
                title="Reasoning",
                summary=_extract_reasoning_summary(reasoning_text or summary_str),
                source_node_refs=[item_id],
                payload_raw_path=f"$.output[{item_idx}]",
                raw_reasoning=reasoning_text or summary_str or None,
            )
            steps.append(step)
            step_index += 1

        elif item_type == "message":
            # Final message output
            role = _safe_get(item, "role", default="assistant")
            content_list = _safe_get(item, "content", default=[])
            
            for content_idx, content_item in enumerate(content_list):
                content_type = _safe_get(content_item, "type", default="text")
                
                if content_type == "output_text" or content_type == "text":
                    text = _safe_get(content_item, "text", default="")
                    
                    final_outputs.append(FinalOutput(
                        text=text,
                        format="text",
                        output_index=len(final_outputs),
                        role=role,
                    ))

                    summary = text[:100] + "..." if len(text) > 100 else text
                    step = TraceStep(
                        id=_generate_step_id(step_index, "output", text[:50]),
                        index=step_index,
                        kind=StepKind.OUTPUT,
                        title=f"Output ({role})",
                        summary=summary,
                        source_node_refs=[item_id],
                        payload_raw_path=f"$.output[{item_idx}].content[{content_idx}]",
                        output_text=text,
                        output_format="text",
                    )
                    steps.append(step)
                    step_index += 1

                elif content_type == "refusal":
                    refusal_text = _safe_get(content_item, "refusal", default="Model refused to respond")
                    diagnostics.warnings.append(f"Model refusal: {refusal_text[:100]}..." if len(refusal_text) > 100 else f"Model refusal: {refusal_text}")
                    
                    step = TraceStep(
                        id=_generate_step_id(step_index, "error", "refusal"),
                        index=step_index,
                        kind=StepKind.ERROR,
                        title="Refusal",
                        summary=refusal_text[:100] + "..." if len(refusal_text) > 100 else refusal_text,
                        source_node_refs=[item_id],
                        payload_raw_path=f"$.output[{item_idx}].content[{content_idx}]",
                        error_message=refusal_text,
                        error_code="refusal",
                    )
                    steps.append(step)
                    step_index += 1

        elif item_type in ("function_call", "tool_call", "function"):
            meta.has_tools = True
            tool_name = _safe_get(item, "name") or _safe_get(item, "function", "name", default="unknown_tool")
            tool_args = _safe_get(item, "arguments") or _safe_get(item, "function", "arguments")
            call_id = _safe_get(item, "call_id") or _safe_get(item, "id", default=item_id)

            # Parse arguments if string
            if isinstance(tool_args, str):
                try:
                    tool_args = json.loads(tool_args)
                except json.JSONDecodeError:
                    pass  # Keep as string

            step = TraceStep(
                id=_generate_step_id(step_index, "tool_call", call_id),
                index=step_index,
                kind=StepKind.TOOL_CALL,
                title=f"Tool: {tool_name}",
                summary=f"Calling {tool_name}",
                source_node_refs=[call_id],
                payload_raw_path=f"$.output[{item_idx}]",
                tool_name=tool_name,
                tool_input=tool_args if isinstance(tool_args, dict) else {"raw": tool_args},
                payload_preview={"tool": tool_name, "call_id": call_id},
            )
            steps.append(step)
            step_index += 1

        elif item_type == "file_search_call":
            meta.has_tools = True
            call_id = _safe_get(item, "id", default=item_id)
            queries = _safe_get(item, "queries", default=[])
            results = _safe_get(item, "results", default=[])
            status = _safe_get(item, "status", default="completed")

            # Tool call step
            step = TraceStep(
                id=_generate_step_id(step_index, "tool_call", call_id),
                index=step_index,
                kind=StepKind.TOOL_CALL,
                title="Tool: file_search",
                summary=f"Searching {len(queries)} queries" if queries else "File search",
                source_node_refs=[call_id],
                payload_raw_path=f"$.output[{item_idx}]",
                tool_name="file_search",
                tool_input={"queries": queries},
                payload_preview={"queries": queries[:3] if len(queries) > 3 else queries},
            )
            steps.append(step)
            step_index += 1

            # Tool result step
            if results or status == "completed":
                result_summary = f"{len(results)} resultados" if results else "Sin resultados"
                step = TraceStep(
                    id=_generate_step_id(step_index, "tool_result", call_id),
                    index=step_index,
                    kind=StepKind.TOOL_RESULT,
                    title="file_search results",
                    summary=result_summary,
                    source_node_refs=[call_id],
                    payload_raw_path=f"$.output[{item_idx}].results",
                    tool_name="file_search",
                    tool_output=results,
                    payload_preview={"count": len(results), "preview": results[:3] if results else []},
                )
                steps.append(step)
                step_index += 1

        elif item_type == "web_search_call":
            meta.has_tools = True
            call_id = _safe_get(item, "id", default=item_id)
            status = _safe_get(item, "status", default="completed")

            step = TraceStep(
                id=_generate_step_id(step_index, "tool_call", call_id),
                index=step_index,
                kind=StepKind.TOOL_CALL,
                title="Tool: web_search",
                summary="Web search",
                source_node_refs=[call_id],
                payload_raw_path=f"$.output[{item_idx}]",
                tool_name="web_search",
                tool_input={},
                payload_preview={"status": status},
            )
            steps.append(step)
            step_index += 1

        elif item_type == "function_call_output":
            # Tool result from a function call
            call_id = _safe_get(item, "call_id", default=item_id)
            output = _safe_get(item, "output", default="")
            
            # Try to parse output as JSON
            output_parsed = output
            if isinstance(output, str):
                try:
                    output_parsed = json.loads(output)
                except json.JSONDecodeError:
                    pass

            summary = str(output)[:100] + "..." if len(str(output)) > 100 else str(output)
            step = TraceStep(
                id=_generate_step_id(step_index, "tool_result", call_id),
                index=step_index,
                kind=StepKind.TOOL_RESULT,
                title="Tool result",
                summary=summary,
                source_node_refs=[call_id],
                payload_raw_path=f"$.output[{item_idx}]",
                tool_output=output_parsed,
            )
            steps.append(step)
            step_index += 1

        else:
            # Unknown item type
            diagnostics.warnings.append(f"Unknown output item type: {item_type}")
            step = TraceStep(
                id=_generate_step_id(step_index, "unknown", item_id),
                index=step_index,
                kind=StepKind.UNKNOWN,
                title=f"Unknown: {item_type}",
                summary=f"Unrecognized item type: {item_type}",
                source_node_refs=[item_id],
                payload_raw_path=f"$.output[{item_idx}]",
                payload_preview=item,
            )
            steps.append(step)
            step_index += 1

    # Check for error in response
    error = _safe_get(data, "error")
    if error:
        error_msg = _safe_get(error, "message", default=str(error))
        error_code = _safe_get(error, "code", default="unknown")
        diagnostics.errors.append(f"API Error ({error_code}): {error_msg}")
        
        summary = error_msg[:100] + "..." if len(error_msg) > 100 else error_msg
        step = TraceStep(
            id=_generate_step_id(step_index, "error", error_code),
            index=step_index,
            kind=StepKind.ERROR,
            title="API Error",
            summary=summary,
            error_message=error_msg,
            error_code=error_code,
            payload_raw_path="$.error",
        )
        steps.append(step)

    return meta, steps, final_outputs


def _parse_chat_completions_api(data: dict, diagnostics: Diagnostics) -> tuple[TraceMeta, list[TraceStep], list[FinalOutput]]:
    """
    Parse OpenAI Chat Completions API format.
    
    This handles the classic chat completions structure with:
    - choices array containing message objects
    - tool_calls in messages
    """
    meta = TraceMeta()
    steps: list[TraceStep] = []
    final_outputs: list[FinalOutput] = []
    step_index = 0

    # Extract metadata
    meta.model = _safe_get(data, "model")
    meta.response_id = _safe_get(data, "id")
    meta.object_type = _safe_get(data, "object")
    meta.created_at = _format_timestamp(_safe_get(data, "created"))
    meta.system_fingerprint = _safe_get(data, "system_fingerprint")
    meta.status = "completed"

    # Token usage
    usage = _safe_get(data, "usage")
    if usage:
        diagnostics.token_usage = TokenUsage(
            input_tokens=_safe_get(usage, "prompt_tokens"),
            output_tokens=_safe_get(usage, "completion_tokens"),
            total_tokens=_safe_get(usage, "total_tokens"),
        )

    # Parse choices
    choices = _safe_get(data, "choices", default=[])
    if not isinstance(choices, list):
        choices = [choices] if choices else []

    for choice_idx, choice in enumerate(choices):
        if not isinstance(choice, dict):
            continue

        finish_reason = _safe_get(choice, "finish_reason")
        if finish_reason:
            diagnostics.finish_reason = finish_reason
            if finish_reason in ("length", "content_filter"):
                diagnostics.incomplete = True
                diagnostics.warnings.append(f"Response truncated: {finish_reason}")

        message = _safe_get(choice, "message", default={})
        role = _safe_get(message, "role", default="assistant")

        # Check for tool calls
        tool_calls = _safe_get(message, "tool_calls", default=[])
        if tool_calls:
            meta.has_tools = True
            for tc_idx, tc in enumerate(tool_calls):
                tc_id = _safe_get(tc, "id", default=f"tc_{tc_idx}")
                tc_type = _safe_get(tc, "type", default="function")
                
                if tc_type == "function":
                    func = _safe_get(tc, "function", default={})
                    func_name = _safe_get(func, "name", default="unknown")
                    func_args = _safe_get(func, "arguments", default="{}")
                    
                    # Parse arguments
                    if isinstance(func_args, str):
                        try:
                            func_args = json.loads(func_args)
                        except json.JSONDecodeError:
                            func_args = {"raw": func_args}

                    step = TraceStep(
                        id=_generate_step_id(step_index, "tool_call", tc_id),
                        index=step_index,
                        kind=StepKind.TOOL_CALL,
                        title=f"Tool: {func_name}",
                        summary=f"Calling {func_name}",
                        source_node_refs=[tc_id],
                        payload_raw_path=f"$.choices[{choice_idx}].message.tool_calls[{tc_idx}]",
                        tool_name=func_name,
                        tool_input=func_args,
                        payload_preview={"tool": func_name, "call_id": tc_id},
                    )
                    steps.append(step)
                    step_index += 1

        # Check for content
        content = _safe_get(message, "content")
        if content:
            final_outputs.append(FinalOutput(
                text=content,
                format="text",
                output_index=choice_idx,
                role=role,
            ))

            summary = content[:100] + "..." if len(content) > 100 else content
            step = TraceStep(
                id=_generate_step_id(step_index, "output", content[:50]),
                index=step_index,
                kind=StepKind.OUTPUT,
                title=f"Output ({role})",
                summary=summary,
                source_node_refs=[f"choice_{choice_idx}"],
                payload_raw_path=f"$.choices[{choice_idx}].message.content",
                output_text=content,
                output_format="text",
            )
            steps.append(step)
            step_index += 1

        # Check for refusal
        refusal = _safe_get(message, "refusal")
        if refusal:
            diagnostics.warnings.append(f"Model refusal: {refusal[:100]}..." if len(refusal) > 100 else f"Model refusal: {refusal}")
            summary = refusal[:100] + "..." if len(refusal) > 100 else refusal
            step = TraceStep(
                id=_generate_step_id(step_index, "error", "refusal"),
                index=step_index,
                kind=StepKind.ERROR,
                title="Refusal",
                summary=summary,
                error_message=refusal,
                error_code="refusal",
                payload_raw_path=f"$.choices[{choice_idx}].message.refusal",
            )
            steps.append(step)
            step_index += 1

    return meta, steps, final_outputs


def _parse_custom_rag_response(data: dict, diagnostics: Diagnostics) -> tuple[TraceMeta, list[TraceStep], list[FinalOutput]]:
    """
    Parse custom RAG response format used in this dashboard.
    
    Handles structures like:
    - razonamiento field
    - codigos array
    - conceptos array
    """
    meta = TraceMeta()
    steps: list[TraceStep] = []
    final_outputs: list[FinalOutput] = []
    step_index = 0

    meta.status = "completed"
    meta.object_type = "custom_rag_response"

    # Check for razonamiento (reasoning)
    razonamiento = _safe_get(data, "razonamiento")
    if razonamiento:
        meta.has_reasoning = True
        # Clean up escaped newlines
        if isinstance(razonamiento, str):
            razonamiento_clean = razonamiento.replace("\\n", "\n")
        else:
            razonamiento_clean = str(razonamiento)

        step = TraceStep(
            id=_generate_step_id(step_index, "reasoning", "razonamiento"),
            index=step_index,
            kind=StepKind.REASONING,
            title="Razonamiento",
            summary=_extract_reasoning_summary(razonamiento_clean),
            payload_raw_path="$.razonamiento",
            raw_reasoning=razonamiento_clean,
        )
        steps.append(step)
        step_index += 1

    # Check for codigos (codes output)
    codigos = _safe_get(data, "codigos")
    if codigos:
        if isinstance(codigos, list):
            output_text = ", ".join(str(c) for c in codigos)
        else:
            output_text = str(codigos)

        final_outputs.append(FinalOutput(
            text=output_text,
            format="codes",
            output_index=0,
        ))

        step = TraceStep(
            id=_generate_step_id(step_index, "output", "codigos"),
            index=step_index,
            kind=StepKind.OUTPUT,
            title="Códigos",
            summary=f"{len(codigos) if isinstance(codigos, list) else 1} códigos",
            payload_raw_path="$.codigos",
            output_text=output_text,
            output_format="codes",
            payload_preview={"codigos": codigos},
        )
        steps.append(step)
        step_index += 1

    # Check for conceptos
    conceptos = _safe_get(data, "conceptos")
    if conceptos:
        if isinstance(conceptos, list):
            output_text = "\n".join(str(c) for c in conceptos)
        else:
            output_text = str(conceptos)

        final_outputs.append(FinalOutput(
            text=output_text,
            format="concepts",
            output_index=len(final_outputs),
        ))

        step = TraceStep(
            id=_generate_step_id(step_index, "output", "conceptos"),
            index=step_index,
            kind=StepKind.OUTPUT,
            title="Conceptos",
            summary=f"{len(conceptos) if isinstance(conceptos, list) else 1} conceptos",
            payload_raw_path="$.conceptos",
            output_text=output_text,
            output_format="concepts",
            payload_preview={"conceptos": conceptos[:5] if isinstance(conceptos, list) and len(conceptos) > 5 else conceptos},
        )
        steps.append(step)
        step_index += 1

    # Check for error field
    error = _safe_get(data, "error")
    if error:
        diagnostics.errors.append(str(error))
        error_str = str(error)
        summary = error_str[:100] + "..." if len(error_str) > 100 else error_str
        step = TraceStep(
            id=_generate_step_id(step_index, "error", "error"),
            index=step_index,
            kind=StepKind.ERROR,
            title="Error",
            summary=summary,
            error_message=error_str,
            payload_raw_path="$.error",
        )
        steps.append(step)
        step_index += 1

    return meta, steps, final_outputs


def _detect_response_format(data: dict) -> str:
    """
    Detect the format of the response data.
    
    Returns one of:
    - 'responses_api': OpenAI Responses API format
    - 'chat_completions': OpenAI Chat Completions API format
    - 'custom_rag': Custom RAG response format
    - 'unknown': Unknown format
    """
    if not isinstance(data, dict):
        return "unknown"

    # Check for Responses API indicators
    if _safe_get(data, "object") == "response":
        return "responses_api"
    if "output" in data and isinstance(data.get("output"), list):
        # Check if output items have type field (Responses API style)
        output = data.get("output", [])
        if output and isinstance(output[0], dict) and "type" in output[0]:
            return "responses_api"

    # Check for Chat Completions API indicators
    if _safe_get(data, "object") == "chat.completion":
        return "chat_completions"
    if "choices" in data and isinstance(data.get("choices"), list):
        return "chat_completions"

    # Check for custom RAG format
    if "razonamiento" in data or "codigos" in data or "conceptos" in data:
        return "custom_rag"

    return "unknown"


def parse_response(data: Any) -> TraceModel:
    """
    Parse any supported response format into a TraceModel.
    
    This is the main entry point for the parser. It:
    1. Detects the response format
    2. Delegates to the appropriate parser
    3. Returns a TraceModel with diagnostics
    
    Never throws - returns a TraceModel with error diagnostics instead.
    
    Args:
        data: Raw response data (dict, JSON string, or any)
        
    Returns:
        TraceModel with parsed trace or error diagnostics
    """
    diagnostics = Diagnostics()
    
    # Handle None
    if data is None:
        diagnostics.errors.append("Response data is None")
        return TraceModel(
            meta=TraceMeta(status="error"),
            steps=[],
            final_outputs=[],
            diagnostics=diagnostics,
            raw=None,
        )

    # Parse JSON string if needed
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError as e:
            diagnostics.errors.append(f"Invalid JSON: {str(e)}")
            return TraceModel(
                meta=TraceMeta(status="error"),
                steps=[],
                final_outputs=[],
                diagnostics=diagnostics,
                raw=data,
            )

    # Ensure we have a dict
    if not isinstance(data, dict):
        diagnostics.errors.append(f"Expected dict, got {type(data).__name__}")
        return TraceModel(
            meta=TraceMeta(status="error"),
            steps=[],
            final_outputs=[],
            diagnostics=diagnostics,
            raw=data,
        )

    # Detect format and parse
    format_type = _detect_response_format(data)
    
    try:
        if format_type == "responses_api":
            meta, steps, final_outputs = _parse_openai_responses_api(data, diagnostics)
        elif format_type == "chat_completions":
            meta, steps, final_outputs = _parse_chat_completions_api(data, diagnostics)
        elif format_type == "custom_rag":
            meta, steps, final_outputs = _parse_custom_rag_response(data, diagnostics)
        else:
            diagnostics.warnings.append(f"Unknown response format, attempting best-effort parse")
            # Try all parsers and use the one that produces the most steps
            results = []
            for parser in [_parse_openai_responses_api, _parse_chat_completions_api, _parse_custom_rag_response]:
                try:
                    m, s, f = parser(data, Diagnostics())
                    results.append((m, s, f, len(s) + len(f)))
                except Exception:
                    pass
            
            if results:
                # Use the result with most content
                results.sort(key=lambda x: x[3], reverse=True)
                meta, steps, final_outputs, _ = results[0]
            else:
                meta = TraceMeta(status="unknown")
                steps = []
                final_outputs = []

    except Exception as e:
        diagnostics.errors.append(f"Parse error: {str(e)}")
        meta = TraceMeta(status="error")
        steps = []
        final_outputs = []

    return TraceModel(
        meta=meta,
        steps=steps,
        final_outputs=final_outputs,
        diagnostics=diagnostics,
        raw=data,
    )


def trace_to_dict(trace: TraceModel) -> dict:
    """
    Convert a TraceModel to a dictionary for serialization.
    Useful for caching or debugging.
    """
    return {
        "meta": {
            "model": trace.meta.model,
            "created_at": trace.meta.created_at,
            "response_id": trace.meta.response_id,
            "status": trace.meta.status,
            "has_tools": trace.meta.has_tools,
            "has_reasoning": trace.meta.has_reasoning,
            "object_type": trace.meta.object_type,
        },
        "steps": [
            {
                "id": s.id,
                "index": s.index,
                "kind": s.kind.value,
                "title": s.title,
                "summary": s.summary,
            }
            for s in trace.steps
        ],
        "final_outputs": [
            {
                "text": o.text[:100] + "..." if len(o.text) > 100 else o.text,
                "format": o.format,
                "output_index": o.output_index,
            }
            for o in trace.final_outputs
        ],
        "diagnostics": {
            "errors": trace.diagnostics.errors,
            "warnings": trace.diagnostics.warnings,
            "incomplete": trace.diagnostics.incomplete,
        },
    }
