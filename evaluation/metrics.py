"""Basic deterministic summary for saved evaluation JSONL files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize(path: Path) -> dict[str, Any]:
    records = read_jsonl(path)
    successes = [r for r in records if r.get("status") == "success"]
    latencies = [r["latency_seconds"] for r in successes if isinstance(r.get("latency_seconds"), (int, float))]
    citations = [r for r in successes if r.get("citations")]
    input_tokens = [
        r["input_tokens"] for r in successes
        if isinstance(r.get("input_tokens"), (int, float))
    ]
    output_tokens = [
        r["output_tokens"] for r in successes
        if isinstance(r.get("output_tokens"), (int, float))
    ]
    total_tokens = [
        r["total_tokens"] for r in successes
        if isinstance(r.get("total_tokens"), (int, float))
    ]
    return {
        "pipeline": path.stem,
        "records": len(records),
        "successful": len(successes),
        "failed": len(records) - len(successes),
        "success_rate": round(len(successes) / len(records), 4) if records else 0,
        "citation_coverage": round(len(citations) / len(successes), 4) if successes else 0,
        "average_latency_seconds": round(mean(latencies), 4) if latencies else None,
        "input_tokens": sum(input_tokens),
        "output_tokens": sum(output_tokens),
        "total_tokens": sum(total_tokens),
        "average_tokens_per_question": round(mean(total_tokens), 2) if total_tokens else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    summaries = [summarize(path) for path in sorted(args.result_dir.glob("*.jsonl"))]
    output = args.result_dir / "summary.csv"
    if summaries:
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
    print(json.dumps(summaries, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
