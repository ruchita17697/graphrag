from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path


NON_ANSWER = re.compile(
    r"couldn't finalize|wasn't able to generate|unable to generate|"
    r"iteration budget|try asking again|contact your administrator",
    re.IGNORECASE,
)


def read_jsonl(path: Path):
    records = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def repair_text(value) -> str:
    text = str(value or "")

    # Repair common UTF-8 text incorrectly decoded as Windows/Latin-1.
    for _ in range(2):
        if not any(marker in text for marker in ("Ã", "â", "ð")):
            break
        try:
            repaired = text.encode("latin-1").decode("utf-8")
            if repaired == text:
                break
            text = repaired
        except (UnicodeEncodeError, UnicodeDecodeError):
            break

    return text


def normalize(value) -> str:
    text = repair_text(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def gold_values(value):
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None:
        return []
    return [str(value)]


def matches_gold(answer: str, gold: str) -> bool:
    answer_norm = normalize(answer)
    gold_norm = normalize(gold)

    if not answer_norm or not gold_norm:
        return False

    # Prevent gold answer 5 from matching numbers such as 50 or 15.
    if re.fullmatch(r"-?\d+(?:\.\d+)?", gold_norm):
        return bool(
            re.search(
                rf"(?<!\d){re.escape(gold_norm)}(?!\d)",
                answer_norm,
            )
        )

    if gold_norm in answer_norm:
        return True

    # Accept shortened event names, e.g. full Olympic title vs "Soling".
    repaired_gold = repair_text(gold)
    parts = re.split(r"\s+[–—-]\s+", repaired_gold)
    if len(parts) > 1:
        short_form = normalize(parts[-1])
        if len(short_form) >= 4 and short_form in answer_norm:
            return True

    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    args = parser.parse_args()

    detail_rows = []
    summary_rows = []
    category_rows = []

    for path in sorted(args.results_dir.glob("*.jsonl")):
        pipeline = path.stem
        records = read_jsonl(path)
        category_counts = defaultdict(
            lambda: {"total": 0, "correct": 0, "non_answer": 0, "review": 0}
        )

        correct = 0
        non_answers = 0
        review = 0

        for record in records:
            answer = str(record.get("answer") or "")
            golds = gold_values(record.get("reference_answer"))
            category = record.get("category") or "unknown"

            if not answer.strip() or NON_ANSWER.search(answer):
                verdict = "incorrect_non_answer"
                non_answers += 1
            elif any(matches_gold(answer, gold) for gold in golds):
                verdict = "correct"
                correct += 1
            else:
                verdict = "needs_manual_review"
                review += 1

            category_counts[category]["total"] += 1
            if verdict == "correct":
                category_counts[category]["correct"] += 1
            elif verdict == "incorrect_non_answer":
                category_counts[category]["non_answer"] += 1
            else:
                category_counts[category]["review"] += 1

            detail_rows.append(
                {
                    "pipeline": pipeline,
                    "question_id": record.get("question_id"),
                    "category": category,
                    "question": repair_text(record.get("question")),
                    "reference_answer": " | ".join(
                        repair_text(gold) for gold in golds
                    ),
                    "answer": repair_text(answer),
                    "automatic_verdict": verdict,
                    "manual_correct": "",
                    "notes": "",
                }
            )

        total = len(records)
        summary_rows.append(
            {
                "pipeline": pipeline,
                "questions": total,
                "automatic_correct": correct,
                "non_answers": non_answers,
                "needs_manual_review": review,
                "confirmed_accuracy_lower_bound": (
                    round(correct / total, 4) if total else 0
                ),
            }
        )

        for category, counts in sorted(category_counts.items()):
            category_rows.append(
                {
                    "pipeline": pipeline,
                    "category": category,
                    **counts,
                }
            )

    detail_path = args.results_dir / "accuracy_details.csv"
    summary_path = args.results_dir / "accuracy_summary_preliminary.csv"
    category_path = args.results_dir / "accuracy_by_category_preliminary.csv"

    with detail_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=detail_rows[0].keys())
        writer.writeheader()
        writer.writerows(detail_rows)

    with summary_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    with category_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=category_rows[0].keys())
        writer.writeheader()
        writer.writerows(category_rows)

    print("\nPreliminary accuracy")
    for row in summary_rows:
        print(
            f"{row['pipeline']}: "
            f"automatic_correct={row['automatic_correct']}, "
            f"non_answers={row['non_answers']}, "
            f"manual_review={row['needs_manual_review']}, "
            f"lower_bound={row['confirmed_accuracy_lower_bound']:.2%}"
        )

    print(f"\nSaved: {detail_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {category_path}")


if __name__ == "__main__":
    main()
