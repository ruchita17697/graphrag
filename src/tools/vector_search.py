"""Adapter for TigerGraph's existing similarity-search tool."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


ToolRunner = Callable[[str, dict, Any], dict]


@dataclass
class SearchResult:
    question: str
    tool_name: str
    evidence: list[Any] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    summary: str = ""
    ok: bool = False
    raw_result: dict = field(default_factory=dict)


class VectorSearchError(RuntimeError):
    """Raised when TigerGraph similarity search fails."""


def _extract_evidence(context: Any) -> list[Any]:
    """Extract retrieved items without assuming one fixed response shape."""

    if context is None:
        return []

    if isinstance(context, list):
        return context

    if not isinstance(context, dict):
        return [context]

    result = context.get("result")

    if result is None:
        return []

    if isinstance(result, list):
        return result

    if isinstance(result, dict):
        for key in ("results", "chunks", "documents", "items"):
            value = result.get(key)
            if isinstance(value, list):
                return value

        return [result]

    return [result]


def vector_search(
    ctx: Any,
    question: str,
    *,
    top_k: int = 5,
    tool_runner: ToolRunner,
) -> SearchResult:
    """Run TigerGraph's registered similarity-search tool.

    The function does not create a database connection, embedding model,
    or vector store. Those dependencies are already contained in ``ctx``.
    """

    cleaned_question = question.strip()

    if not cleaned_question:
        raise ValueError("question must not be empty")

    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    output = tool_runner(
        "graphrag__similarity_search",
        {
            "question": cleaned_question,
            "top_k": top_k,
        },
        ctx,
    )

    if not isinstance(output, dict):
        raise VectorSearchError(
            "TigerGraph similarity search returned an invalid response"
        )

    if not output.get("ok", False):
        raise VectorSearchError(
            output.get("summary", "TigerGraph similarity search failed")
        )

    context = output.get("context")
    evidence = _extract_evidence(context)

    return SearchResult(
        question=cleaned_question,
        tool_name="graphrag__similarity_search",
        evidence=evidence,
        citations=list(output.get("citations") or []),
        summary=str(output.get("summary") or ""),
        ok=True,
        raw_result=output,
    )
