"""Checkpointed evaluation runner for the three GraphRAG pipelines."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.agentic_rag import load_tigergraph_token, run_agentic_rag
from src.graph_rag import run_graph_rag
from src.normal_rag import run_normal_rag


Pipeline = Callable[..., dict[str, Any]]
PIPELINES: dict[str, Pipeline] = {
    "normal_rag": run_normal_rag,
    "graphrag": run_graph_rag,
    "agentic_rag": run_agentic_rag,
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
            if isinstance(value, dict):
                records.append(value)
    return records


def question_text(record: dict[str, Any]) -> str:
    for key in ("question", "query", "text"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(f"Question text missing from record: {record}")


def question_id(record: dict[str, Any], index: int) -> str:
    for key in ("question_id", "id", "qid"):
        value = record.get(key)
        if value is not None:
            return str(value)
    return f"question_{index:03d}"


def reference_answer(record: dict[str, Any]) -> Any:
    for key in ("reference_answer", "gold_answer", "expected_answer", "answer"):
        if key in record:
            return record[key]
    return None


def select_pilot(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    categories = ["lookup", "aggregation", "multi_hop", "temporal", "superlative"]
    selected: dict[str, dict[str, Any]] = {}
    for record in records:
        category = record.get("qtype", record.get("category"))
        if category in categories and category not in selected:
            selected[category] = record
    missing = [category for category in categories if category not in selected]
    if missing:
        raise ValueError(f"Pilot categories missing: {missing}")
    return [selected[category] for category in categories]


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    for record in load_jsonl(path):
        if record.get("status") == "success":
            completed.add(str(record.get("question_id")))
    return completed


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        handle.flush()


def build_result(
    source: dict[str, Any],
    index: int,
    pipeline: str,
    response: dict[str, Any] | None,
    error: Exception | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "question_id": question_id(source, index),
        "category": source.get("qtype", source.get("category")),
        "pipeline": pipeline,
        "question": question_text(source),
        "reference_answer": reference_answer(source),
        "answer": None,
        "citations": [],
        "latency_seconds": None,
        "backend_response_time": None,
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "message_id": None,
        "conversation_id": None,
        "trace_events": [],
        "status": "error" if error else "success",
        "error": str(error) if error else None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if response:
        result.update(
            answer=response.get("answer"),
            citations=response.get("citations") or [],
            latency_seconds=response.get("latency_seconds"),
            backend_response_time=response.get("backend_response_time"),
            message_id=response.get("message_id"),
            conversation_id=response.get("conversation_id"),
            trace_events=response.get("trace_events") or [],
            input_tokens=((response.get("citations") or {}).get("token_usage") or {}).get("input_tokens"),
            output_tokens=((response.get("citations") or {}).get("token_usage") or {}).get("output_tokens"),
            total_tokens=((response.get("citations") or {}).get("token_usage") or {}).get("total_tokens"),
            raw_response=response.get("raw_response"),
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--graphname", default="Olympicgraphrag2")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument(
        "--pipelines",
        nargs="+",
        choices=sorted(PIPELINES),
        default=list(PIPELINES),
    )
    args = parser.parse_args()

    records = load_jsonl(args.questions)
    if args.pilot:
        records = select_pilot(records)
    print(f"Questions selected: {len(records)}")

    token = load_tigergraph_token()
    print("Authentication loaded securely from the running container")

    for pipeline_name in args.pipelines:
        runner = PIPELINES[pipeline_name]
        output_path = args.output_dir / f"{pipeline_name}.jsonl"
        done = completed_ids(output_path)
        print(f"\nPipeline: {pipeline_name}; already completed: {len(done)}")

        for index, source in enumerate(records, start=1):
            qid = question_id(source, index)
            if qid in done:
                print(f"  SKIP {qid}")
                continue
            print(f"  RUN  {qid} ({source.get('qtype', source.get('category', 'unknown'))})")
            response = None
            error = None
            try:
                response = runner(
                    question_text(source),
                    graphname=args.graphname,
                    token=token,
                )
            except Exception as exc:  # preserve the failure and continue
                error = exc
                print(f"  ERROR {qid}: {exc}")
            result = build_result(source, index, pipeline_name, response, error)
            append_jsonl(output_path, result)
            if error and any(marker in str(error) for marker in ("429", "RESOURCE_EXHAUSTED")):
                raise SystemExit("Quota exhausted; checkpoint saved. Rerun later to resume.")


if __name__ == "__main__":
    main()
