"""
Unit Tests for Trace Parser
============================

Tests the trace parser module for various response formats and edge cases.
Run with: pytest tests/test_trace_parser.py -v
"""

import json
import pytest

from trace_parser import (
    parse_response,
    TraceModel,
    TraceStep,
    StepKind,
    TraceMeta,
    Diagnostics,
    TokenUsage,
    _detect_response_format,
    _truncate_text,
    _extract_reasoning_summary,
    _safe_get,
)


class TestHelperFunctions:
    """Test helper functions."""

    def test_safe_get_basic(self):
        data = {"a": {"b": {"c": 123}}}
        assert _safe_get(data, "a", "b", "c") == 123
        assert _safe_get(data, "a", "b") == {"c": 123}
        assert _safe_get(data, "x", default="missing") == "missing"
        assert _safe_get(data, "a", "x", "y", default=None) is None

    def test_safe_get_with_none(self):
        data = {"a": None}
        assert _safe_get(data, "a", default="default") == "default"
        assert _safe_get(None, "a", default="default") == "default"

    def test_truncate_text_short(self):
        text = "Hello world"
        assert _truncate_text(text, 50) == "Hello world"

    def test_truncate_text_long(self):
        text = "A" * 300
        result = _truncate_text(text, 100)
        assert len(result) == 100
        assert result.endswith("...")

    def test_extract_reasoning_summary(self):
        reasoning = "This is a test reasoning\nwith multiple lines\nand some content"
        summary = _extract_reasoning_summary(reasoning)
        assert "lines" in summary.lower()
        assert "words" in summary.lower()

    def test_extract_reasoning_summary_empty(self):
        summary = _extract_reasoning_summary("")
        assert "available" in summary.lower()


class TestFormatDetection:
    """Test response format detection."""

    def test_detect_responses_api(self):
        data = {"object": "response", "output": []}
        assert _detect_response_format(data) == "responses_api"

    def test_detect_responses_api_by_output(self):
        data = {"output": [{"type": "message", "content": []}]}
        assert _detect_response_format(data) == "responses_api"

    def test_detect_chat_completions(self):
        data = {"object": "chat.completion", "choices": []}
        assert _detect_response_format(data) == "chat_completions"

    def test_detect_chat_completions_by_choices(self):
        data = {"choices": [{"message": {}}]}
        assert _detect_response_format(data) == "chat_completions"

    def test_detect_custom_rag(self):
        data = {"razonamiento": "some text", "codigos": []}
        assert _detect_response_format(data) == "custom_rag"

    def test_detect_unknown(self):
        data = {"random_field": 123}
        assert _detect_response_format(data) == "unknown"

    def test_detect_non_dict(self):
        assert _detect_response_format([1, 2, 3]) == "unknown"
        assert _detect_response_format("string") == "unknown"


class TestParseResponseBasics:
    """Test basic parse_response functionality."""

    def test_parse_none(self):
        trace = parse_response(None)
        assert isinstance(trace, TraceModel)
        assert len(trace.diagnostics.errors) > 0
        assert trace.meta.status == "error"

    def test_parse_invalid_json_string(self):
        trace = parse_response("not valid json {{{")
        assert len(trace.diagnostics.errors) > 0
        assert "JSON" in trace.diagnostics.errors[0]

    def test_parse_non_dict(self):
        trace = parse_response([1, 2, 3])
        assert len(trace.diagnostics.errors) > 0

    def test_parse_empty_dict(self):
        trace = parse_response({})
        assert isinstance(trace, TraceModel)
        # Should handle gracefully with warnings

    def test_parse_json_string(self):
        data = json.dumps({"razonamiento": "test", "codigos": ["A", "B"]})
        trace = parse_response(data)
        assert len(trace.diagnostics.errors) == 0
        assert trace.meta.has_reasoning


class TestParseOpenAIResponsesAPI:
    """Test parsing OpenAI Responses API format."""

    def test_parse_basic_response(self):
        data = {
            "object": "response",
            "id": "resp_123",
            "model": "gpt-4o",
            "status": "completed",
            "created_at": 1706745600,
            "output": [
                {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "Hello, world!"}
                    ]
                }
            ],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15
            }
        }
        trace = parse_response(data)
        
        assert trace.meta.model == "gpt-4o"
        assert trace.meta.response_id == "resp_123"
        assert trace.meta.status == "completed"
        assert len(trace.final_outputs) == 1
        assert trace.final_outputs[0].text == "Hello, world!"
        assert trace.diagnostics.token_usage.input_tokens == 10

    def test_parse_response_with_reasoning(self):
        data = {
            "object": "response",
            "model": "o1-preview",
            "status": "completed",
            "output": [
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "content": [
                        {"type": "reasoning_text", "text": "Let me think about this..."}
                    ],
                    "summary": [{"text": "Analyzed the problem"}]
                },
                {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "The answer is 42."}
                    ]
                }
            ]
        }
        trace = parse_response(data)
        
        assert trace.meta.has_reasoning
        assert len(trace.steps) == 2
        
        reasoning_step = trace.steps[0]
        assert reasoning_step.kind == StepKind.REASONING
        assert reasoning_step.raw_reasoning == "Let me think about this..."
        
        output_step = trace.steps[1]
        assert output_step.kind == StepKind.OUTPUT

    def test_parse_response_with_file_search(self):
        data = {
            "object": "response",
            "model": "gpt-4o",
            "status": "completed",
            "output": [
                {
                    "type": "file_search_call",
                    "id": "fs_1",
                    "queries": ["query1", "query2"],
                    "results": [
                        {"file_id": "file_1", "score": 0.95},
                        {"file_id": "file_2", "score": 0.85}
                    ],
                    "status": "completed"
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Found results."}]
                }
            ]
        }
        trace = parse_response(data)
        
        assert trace.meta.has_tools
        # Should have tool call + tool result + output
        assert len(trace.steps) >= 3
        
        tool_call = next(s for s in trace.steps if s.kind == StepKind.TOOL_CALL)
        assert tool_call.tool_name == "file_search"
        
        tool_result = next(s for s in trace.steps if s.kind == StepKind.TOOL_RESULT)
        assert len(tool_result.tool_output) == 2

    def test_parse_response_with_function_call(self):
        data = {
            "object": "response",
            "model": "gpt-4o",
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_123",
                    "name": "get_weather",
                    "arguments": '{"location": "Madrid"}'
                }
            ]
        }
        trace = parse_response(data)
        
        assert trace.meta.has_tools
        tool_step = trace.steps[0]
        assert tool_step.kind == StepKind.TOOL_CALL
        assert tool_step.tool_name == "get_weather"
        assert tool_step.tool_input == {"location": "Madrid"}

    def test_parse_incomplete_response(self):
        data = {
            "object": "response",
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": []
        }
        trace = parse_response(data)
        
        assert trace.diagnostics.incomplete
        assert len(trace.diagnostics.warnings) > 0

    def test_parse_response_with_error(self):
        data = {
            "object": "response",
            "status": "failed",
            "error": {
                "code": "rate_limit_exceeded",
                "message": "Rate limit exceeded"
            },
            "output": []
        }
        trace = parse_response(data)
        
        assert len(trace.diagnostics.errors) > 0
        error_step = next((s for s in trace.steps if s.kind == StepKind.ERROR), None)
        assert error_step is not None


class TestParseChatCompletionsAPI:
    """Test parsing OpenAI Chat Completions API format."""

    def test_parse_basic_completion(self):
        data = {
            "object": "chat.completion",
            "id": "chatcmpl_123",
            "model": "gpt-4",
            "created": 1706745600,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello! How can I help?"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 8,
                "total_tokens": 18
            }
        }
        trace = parse_response(data)
        
        assert trace.meta.model == "gpt-4"
        assert trace.meta.response_id == "chatcmpl_123"
        assert len(trace.final_outputs) == 1
        assert trace.final_outputs[0].text == "Hello! How can I help?"
        assert trace.diagnostics.token_usage.input_tokens == 10

    def test_parse_completion_with_tool_calls(self):
        data = {
            "object": "chat.completion",
            "model": "gpt-4",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "search_database",
                                    "arguments": '{"query": "test"}'
                                }
                            }
                        ]
                    },
                    "finish_reason": "tool_calls"
                }
            ]
        }
        trace = parse_response(data)
        
        assert trace.meta.has_tools
        tool_step = trace.steps[0]
        assert tool_step.kind == StepKind.TOOL_CALL
        assert tool_step.tool_name == "search_database"

    def test_parse_completion_truncated(self):
        data = {
            "object": "chat.completion",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Partial..."},
                    "finish_reason": "length"
                }
            ]
        }
        trace = parse_response(data)
        
        assert trace.diagnostics.incomplete
        assert trace.diagnostics.finish_reason == "length"


class TestParseCustomRAGFormat:
    """Test parsing custom RAG response format."""

    def test_parse_basic_rag_response(self):
        data = {
            "razonamiento": "Based on the documents...",
            "codigos": ["ABC123", "DEF456"],
            "conceptos": ["Concept A", "Concept B"]
        }
        trace = parse_response(data)
        
        assert trace.meta.has_reasoning
        assert trace.meta.object_type == "custom_rag_response"
        
        # Should have reasoning + codigos + conceptos steps
        assert len(trace.steps) == 3
        
        reasoning_step = next(s for s in trace.steps if s.kind == StepKind.REASONING)
        assert reasoning_step.raw_reasoning == "Based on the documents..."

    def test_parse_rag_response_with_escaped_newlines(self):
        data = {
            "razonamiento": "Line 1\\nLine 2\\nLine 3",
            "codigos": ["A"]
        }
        trace = parse_response(data)
        
        reasoning_step = next(s for s in trace.steps if s.kind == StepKind.REASONING)
        assert "\n" in reasoning_step.raw_reasoning  # Should be unescaped

    def test_parse_rag_response_with_error(self):
        data = {
            "error": "Failed to process query",
            "codigos": []
        }
        trace = parse_response(data)
        
        assert len(trace.diagnostics.errors) > 0
        error_step = next((s for s in trace.steps if s.kind == StepKind.ERROR), None)
        assert error_step is not None


class TestTraceModelMethods:
    """Test TraceModel helper methods."""

    def test_has_errors_with_diagnostics(self):
        data = {"error": {"message": "Test error"}, "object": "response", "output": []}
        trace = parse_response(data)
        assert trace.has_errors()

    def test_has_errors_with_error_steps(self):
        # Response with a refusal
        data = {
            "object": "response",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "refusal", "refusal": "Cannot help"}]
                }
            ]
        }
        trace = parse_response(data)
        assert trace.has_errors()

    def test_get_steps_by_kind(self):
        data = {
            "object": "response",
            "output": [
                {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "..."}]},
                {"type": "file_search_call", "queries": [], "results": [], "status": "completed"},
                {"type": "message", "content": [{"type": "output_text", "text": "Done"}]}
            ]
        }
        trace = parse_response(data)
        
        reasoning = trace.get_reasoning_steps()
        assert len(reasoning) == 1
        
        tools = trace.get_tool_steps()
        assert len(tools) >= 1


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_malformed_tool_arguments(self):
        data = {
            "object": "response",
            "output": [
                {
                    "type": "function_call",
                    "name": "test",
                    "arguments": "not valid json"
                }
            ]
        }
        trace = parse_response(data)
        # Should not crash, should handle gracefully
        assert len(trace.steps) > 0

    def test_missing_content_arrays(self):
        data = {
            "object": "response",
            "output": [
                {"type": "message", "content": None},
                {"type": "reasoning", "content": None}
            ]
        }
        trace = parse_response(data)
        # Should handle None content gracefully
        assert len(trace.diagnostics.errors) == 0

    def test_unknown_output_type(self):
        data = {
            "object": "response",
            "output": [
                {"type": "future_type_xyz", "data": "unknown"}
            ]
        }
        trace = parse_response(data)
        
        assert len(trace.diagnostics.warnings) > 0
        unknown_step = next(s for s in trace.steps if s.kind == StepKind.UNKNOWN)
        assert unknown_step is not None

    def test_empty_output_array(self):
        data = {"object": "response", "output": [], "status": "completed"}
        trace = parse_response(data)
        assert len(trace.steps) == 0
        assert len(trace.final_outputs) == 0

    def test_deeply_nested_structure(self):
        data = {
            "razonamiento": "Test",
            "codigos": ["A"],
            "nested": {"deep": {"very": {"deep": "value"}}}
        }
        trace = parse_response(data)
        # Should parse without issues, extra fields ignored
        assert trace.meta.has_reasoning


class TestLargePayloads:
    """Test handling of large payloads."""

    def test_large_reasoning_text(self):
        large_text = "A" * 100000  # 100KB
        data = {
            "razonamiento": large_text,
            "codigos": ["TEST"]
        }
        trace = parse_response(data)
        
        reasoning_step = next(s for s in trace.steps if s.kind == StepKind.REASONING)
        assert reasoning_step.raw_reasoning == large_text
        # Summary should be truncated
        assert len(reasoning_step.summary) < 1000

    def test_many_tool_results(self):
        results = [{"id": f"result_{i}", "score": 0.9 - i * 0.01} for i in range(100)]
        data = {
            "object": "response",
            "output": [
                {
                    "type": "file_search_call",
                    "queries": ["test"],
                    "results": results,
                    "status": "completed"
                }
            ]
        }
        trace = parse_response(data)
        
        tool_result = next(s for s in trace.steps if s.kind == StepKind.TOOL_RESULT)
        assert tool_result.tool_output_truncated  # Should indicate truncation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
