"""WebSocket client for TigerGraph GraphRAG chat evaluation."""

from __future__ import annotations

import asyncio
import json
import subprocess
import time
from typing import Any
from urllib.parse import quote

import websockets


def load_tigergraph_token() -> str:
    """Read the configured token from the running container without printing it."""
    command = [
        "docker",
        "compose",
        "exec",
        "-T",
        "graphrag",
        "python",
        "-c",
        (
            "from common.config import db_config; "
            "print(db_config.get('apiToken', ''))"
        ),
    ]
    process = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line.strip() for line in process.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("TigerGraph API token was not available")
    token = lines[-1]
    if not token:
        raise RuntimeError("TigerGraph API token is empty")
    return token


def _decode_event(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
        return {"content": parsed, "response_type": "progress"}
    except json.JSONDecodeError:
        return {"content": raw, "response_type": "progress"}


async def run_chat(
    question: str,
    *,
    rag_pattern: str,
    mode: str,
    graphname: str = "Olympicgraphrag2",
    base_ws_url: str = "ws://localhost:8000",
    timeout_seconds: float = 600.0,
    token: str | None = None,
) -> dict[str, Any]:
    """Run one isolated chat turn and retain all progress/trace events."""
    auth_token = token or load_tigergraph_token()
    url = (
        f"{base_ws_url}/ui/{quote(graphname)}/chat"
        f"?rag_pattern={quote(rag_pattern)}&mode={quote(mode)}"
    )
    events: list[dict[str, Any]] = []
    started = time.perf_counter()

    async with websockets.connect(
        url,
        open_timeout=30,
        ping_interval=20,
        ping_timeout=30,
        max_size=None,
    ) as websocket:
        await websocket.send(f"Bearer {auth_token}")
        await websocket.send("new")

        first = _decode_event(await asyncio.wait_for(websocket.recv(), timeout=30))
        events.append(first)
        conversation_id = first.get("conversation_id")
        if not conversation_id:
            raise RuntimeError(f"Expected conversation_id; received: {first}")

        await websocket.send(question)
        final: dict[str, Any] | None = None

        while final is None:
            raw = await asyncio.wait_for(
                websocket.recv(),
                timeout=timeout_seconds,
            )
            event = _decode_event(raw)
            events.append(event)

            if event.get("notice") == "vector_search_unavailable":
                raise RuntimeError(event.get("message", "Vector search unavailable"))

            if event.get("content") is not None and event.get("response_type") != "progress":
                final = event

    elapsed = time.perf_counter() - started
    assert final is not None
    return {
        "answer": final.get("content", ""),
        "citations": final.get("query_sources") or {},
        "answered_question": final.get("answered_question"),
        "response_type": final.get("response_type"),
        "message_id": final.get("message_id"),
        "conversation_id": conversation_id,
        "latency_seconds": round(elapsed, 6),
        "backend_response_time": final.get("response_time"),
        "trace_events": events,
        "raw_response": final,
    }


def run_agentic_rag(
    question: str,
    *,
    graphname: str = "Olympicgraphrag2",
    token: str | None = None,
) -> dict[str, Any]:
    """Run Planned Agentic GraphRAG."""
    return asyncio.run(
        run_chat(
            question,
            rag_pattern="planned",
            mode="agentic",
            graphname=graphname,
            token=token,
        )
    )
