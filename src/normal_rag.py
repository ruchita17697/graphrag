"""Normal RAG evaluation adapter."""

from __future__ import annotations

import asyncio
from typing import Any

from src.agentic_rag import run_chat


def run_normal_rag(
    question: str,
    *,
    graphname: str = "Olympicgraphrag2",
    token: str | None = None,
) -> dict[str, Any]:
    return asyncio.run(
        run_chat(
            question,
            rag_pattern="Similarity Search",
            mode="classic",
            graphname=graphname,
            token=token,
        )
    )
