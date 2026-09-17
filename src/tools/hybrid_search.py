"""Adapter for TigerGraph's hybrid vector and graph search tool."""

from __future__ import annotations

from typing import Any

from src.tools.vector_search import (
    SearchResult,
    ToolRunner,
    VectorSearchError,
    _extract_evidence,
)


class HybridSearchError(VectorSearchError):
    """Raised when TigerGraph hybrid search fails."""


def hybrid_search(
    ctx: Any,
    question: str,
    *,
    top_k: int = 10,
    num_hops: int = 2,
    chunk_only: bool = True,
    tool_runner: ToolRunner,
) -> SearchResult:
    """Run TigerGraph's registered hybrid-search tool."""

    cleaned_question = question.strip()

    if not cleaned_question:
        raise ValueError("question must not be empty")

    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    if num_hops < 0:
        raise ValueError("num_hops must not be negative")

    output = tool_runner(
        "graphrag__hybrid_search",
        {
            "question": cleaned_question,
            "top_k": top_k,
            "num_hops": num_hops,
            "chunk_only": chunk_only,
        },
        ctx,
    )

    if not isinstance(output, dict):
        raise HybridSearchError(
            "TigerGraph hybrid search returned an invalid response"
        )

    if not output.get("ok", False):
        raise HybridSearchError(
            output.get("summary", "TigerGraph hybrid search failed")
        )

    context = output.get("context")

    return SearchResult(
        question=cleaned_question,
        tool_name="graphrag__hybrid_search",
        evidence=_extract_evidence(context),
        citations=list(output.get("citations") or []),
        summary=str(output.get("summary") or ""),
        ok=True,
        raw_result=output,
    )
