"""
Deterministic aggregation tools for Agentic GraphRAG.

The LLM should not calculate counts, filters, minimums or maximums
directly from long retrieved contexts. This module converts retrieved
documents into structured records and performs those operations using
ordinary Python.
"""

from __future__ import annotations

import operator
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable, Literal

from src.tools.document_fetch import DocumentRecord


@dataclass
class NumericRecord:
    """One entity/event and its extracted numeric value."""

    entity: str
    value: float
    field_name: str
    document_id: str
    citation_id: str
    source_text: str = ""


@dataclass
class AggregationResult:
    """Result of a deterministic numeric aggregation."""

    operation: str
    field_name: str
    comparison: str
    threshold: float
    count: int

    all_records: list[NumericRecord] = field(
        default_factory=list
    )

    qualifying_records: list[NumericRecord] = field(
        default_factory=list
    )

    excluded_records: list[NumericRecord] = field(
        default_factory=list
    )

    missing_documents: list[str] = field(
        default_factory=list
    )

    warnings: list[str] = field(
        default_factory=list
    )

    @property
    def citations(self) -> list[str]:
        """Return citations for qualifying records."""

        return [
            record.citation_id
            for record in self.qualifying_records
        ]


COMPARISON_FUNCTIONS: dict[
    str,
    Callable[[float, float], bool],
] = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}


@dataclass(frozen=True)
class QuestionConstraints:
    """Strict corpus filters and numeric operation parsed from a question."""

    year: int | None = None
    season: str | None = None
    sport: str | None = None
    venue: str | None = None
    date_text: str | None = None
    field_name: str | None = None
    comparison: str | None = None
    threshold: float | None = None
    operation: Literal["count", "lookup", "max", "min"] = "count"


def parse_question_constraints(question: str) -> QuestionConstraints:
    """Parse benchmark-safe constraints without using an LLM.

    The parser is deliberately conservative: a supplied year, season,
    sport, venue, date, comparison or threshold is never discarded.
    """

    cleaned = " ".join(question.strip().split())
    lowered = cleaned.lower()

    year_match = re.search(r"\b((?:18|19|20)\d{2})\b", cleaned)
    year = int(year_match.group(1)) if year_match else None

    season_match = re.search(r"\b(summer|winter)\s+olympics\b", cleaned, re.I)
    season = season_match.group(1).title() if season_match else None

    sport = None
    sport_patterns = [
        r"how many\s+(.+?)\s+events?\s+at\s+the\b",
        r"which\s+(.+?)\s+event\s+at\s+the\b",
        r"\bin\s+(.+?)\s+at\s+the\s+(?:18|19|20)\d{2}\b",
    ]
    for pattern in sport_patterns:
        match = re.search(pattern, cleaned, re.I)
        if match:
            candidate = match.group(1).strip(" ?.,")
            if 1 <= len(candidate.split()) <= 4:
                sport = candidate
                break

    venue_match = re.search(
        r"\bheld\s+at\s+(.+?)(?=\s+on\s+\d{1,2}\s+[A-Za-z]+\s+\d{4}|[?.]|$)",
        cleaned,
        re.I,
    )
    venue = venue_match.group(1).strip() if venue_match else None

    date_match = re.search(
        r"\b(\d{1,2}\s+[A-Za-z]+\s+(?:18|19|20)\d{2})\b",
        cleaned,
        re.I,
    )
    date_text = date_match.group(1) if date_match else None

    field_name = None
    for field in ("competitors", "nations", "participants", "athletes", "teams"):
        if re.search(rf"\b{field}\b", cleaned, re.I):
            field_name = field
            break

    comparison = None
    threshold = None
    comparisons = [
        (r"\bmore than\s+([\d,]+(?:\.\d+)?)", ">"),
        (r"\bat least\s+([\d,]+(?:\.\d+)?)", ">="),
        (r"\bless than\s+([\d,]+(?:\.\d+)?)", "<"),
        (r"\bat most\s+([\d,]+(?:\.\d+)?)", "<="),
        (r"\bexactly\s+([\d,]+(?:\.\d+)?)", "=="),
    ]
    for pattern, operator_name in comparisons:
        match = re.search(pattern, cleaned, re.I)
        if match:
            comparison = operator_name
            threshold = float(match.group(1).replace(",", ""))
            break


    if re.search(r"\b(highest|largest|most|maximum)\b", lowered):
        operation: Literal["count", "lookup", "max", "min"] = "max"
    elif re.search(r"\b(lowest|smallest|least|minimum|fewest)\b", lowered):
        operation = "min"
    elif re.search(r"\bhow many\s+(nations|competitors|participants|athletes|teams)\b", lowered):
        operation = "lookup"
    else:
        operation = "count"

    return QuestionConstraints(
        year=year,
        season=season,
        sport=sport,
        venue=venue,
        date_text=date_text,
        field_name=field_name,
        comparison=comparison,
        threshold=threshold,
        operation=operation,
    )


def filter_documents_by_constraints(
    documents: Iterable[DocumentRecord],
    constraints: QuestionConstraints,
) -> list[DocumentRecord]:
    """Apply title identity constraints before inspecting document text."""

    filtered: list[DocumentRecord] = []

    for document in documents:
        title = extract_title(document).lower()
        full_text = " ".join(
            (document.title, document.text)
        ).lower()

        # Event identity belongs in the title. Article bodies frequently
        # mention other Olympic years and sports, which must not cause a
        # document to enter the calculation.
        if (
            constraints.year is not None
            and str(constraints.year) not in title
        ):
            continue

        if (
            constraints.season
            and constraints.season.lower() not in title
        ):
            continue

        if (
            constraints.sport
            and constraints.sport.lower() not in title
        ):
            continue

        # Venue and exact date may occur only in the article body.
        if (
            constraints.venue
            and constraints.venue.lower() not in full_text
        ):
            continue

        if (
            constraints.date_text
            and constraints.date_text.lower() not in full_text
        ):
            continue

        filtered.append(document)

    return filtered

def answer_numeric_question(
    documents: Iterable[DocumentRecord],
    question: str,
    *,
    field_name: str | None = None,
    comparison: str | None = None,
    threshold: float | None = None,
) -> dict:
    """Answer count, numeric lookup, maximum or minimum deterministically."""

    constraints = parse_question_constraints(question)
    effective_field = field_name or constraints.field_name
    effective_comparison = comparison or constraints.comparison
    effective_threshold = threshold if threshold is not None else constraints.threshold
    candidates = filter_documents_by_constraints(documents, constraints)

    if constraints.operation == "count" and effective_comparison is None:
        unique = {document.document_id: document for document in candidates}
        return {
            "operation": "count",
            "answer_value": len(unique),
            "records": [
                {"entity": extract_title(doc), "document_id": doc.document_id,
                 "citation_id": doc.document_id}
                for doc in unique.values()
            ],
            "citations": list(unique),
            "constraints": constraints,
        }

    if not effective_field:
        raise ValueError("No numeric field could be inferred from the question.")

    records, missing = documents_to_numeric_records(candidates, effective_field)
    records, warnings = deduplicate_records(records)

    if not records:
        raise ValueError("No records matched all question constraints with the requested field.")

    if constraints.operation == "lookup":
        chosen = records[0]
        answer_value = chosen.value
        selected = [chosen]
    elif constraints.operation in ("max", "min"):
        chooser = max if constraints.operation == "max" else min
        extreme = chooser(record.value for record in records)
        selected = [record for record in records if record.value == extreme]
        answer_value = selected[0].entity if len(selected) == 1 else [r.entity for r in selected]
    else:
        if effective_comparison is None or effective_threshold is None:
            raise ValueError("Filtered count requires a comparison and threshold.")
        compare = COMPARISON_FUNCTIONS[effective_comparison]
        selected = [record for record in records if compare(record.value, effective_threshold)]
        answer_value = len(selected)

    return {
        "operation": constraints.operation,
        "answer_value": answer_value,
        "field_name": effective_field,
        "comparison": effective_comparison,
        "threshold": effective_threshold,
        "records": [
            {"entity": record.entity, "value": record.value,
             "document_id": record.document_id, "citation_id": record.citation_id}
            for record in selected
        ],
        "citations": [record.citation_id for record in selected],
        "missing_documents": missing,
        "warnings": warnings,
        "constraints": constraints,
    }


def _extract_first_match(
    text: str,
    patterns: list[str],
) -> str | None:
    """Return the first captured value from matching patterns."""

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        )

        if match:
            return match.group(1).strip()

    return None


def extract_title(document: DocumentRecord) -> str:
    """Extract an event/entity name from a source document."""

    if document.title.strip():
        return document.title.strip()

    title = _extract_first_match(
        document.text,
        [
            r"^\s*TITLE\s*:\s*(.+?)\s*$",
            r"^\s*event\s*:\s*(.+?)\s*$",
        ],
    )

    if title:
        return title

    return document.document_id


def extract_numeric_field(
    document: DocumentRecord,
    field_name: str,
) -> float | None:
    """
    Extract a numeric field from a document.

    Example:
        field_name = "competitors"
        text contains "competitors: 80"
        result = 80.0
    """

    escaped_field = re.escape(field_name)

    raw_value = _extract_first_match(
        document.text,
        [
            rf"^\s*{escaped_field}\s*:\s*"
            rf"([+-]?\d+(?:,\d{{3}})*(?:\.\d+)?)\s*$",

            rf"\b{escaped_field}\b\s*(?:was|were|is|=|:)\s*"
            rf"([+-]?\d+(?:,\d{{3}})*(?:\.\d+)?)",
        ],
    )

    if raw_value is None:
        return None

    normalized_value = raw_value.replace(",", "")

    try:
        return float(normalized_value)
    except ValueError:
        return None


def documents_to_numeric_records(
    documents: Iterable[DocumentRecord],
    field_name: str,
) -> tuple[list[NumericRecord], list[str]]:
    """
    Convert documents into numeric records.

    Returns:
        records
        document IDs where the required field was missing
    """

    records: list[NumericRecord] = []
    missing_documents: list[str] = []

    for document in documents:
        value = extract_numeric_field(
            document=document,
            field_name=field_name,
        )

        if value is None:
            missing_documents.append(document.document_id)
            continue

        records.append(
            NumericRecord(
                entity=extract_title(document),
                value=value,
                field_name=field_name,
                document_id=document.document_id,
                citation_id=document.document_id,
                source_text=document.text,
            )
        )

    return records, missing_documents


def deduplicate_records(
    records: Iterable[NumericRecord],
) -> tuple[list[NumericRecord], list[str]]:
    """
    Deduplicate records using normalized entity names.

    If duplicate entities contain conflicting values, the first value
    is preserved and a warning is returned.
    """

    unique_records: dict[str, NumericRecord] = {}
    warnings: list[str] = []

    for record in records:
        normalized_entity = re.sub(
            r"\s+",
            " ",
            record.entity.strip().lower(),
        )

        existing = unique_records.get(normalized_entity)

        if existing is None:
            unique_records[normalized_entity] = record
            continue

        if existing.value != record.value:
            warnings.append(
                f"Conflicting values for '{record.entity}': "
                f"{existing.value} from {existing.document_id} "
                f"and {record.value} from {record.document_id}."
            )

    return list(unique_records.values()), warnings


def filter_and_count(
    documents: Iterable[DocumentRecord],
    field_name: str,
    comparison: str,
    threshold: float,
) -> AggregationResult:
    """
    Extract, deduplicate, filter and count numeric records.

    Example:
        filter_and_count(
            documents=biathlon_documents,
            field_name="competitors",
            comparison=">",
            threshold=73,
        )
    """

    if comparison not in COMPARISON_FUNCTIONS:
        raise ValueError(
            f"Unsupported comparison '{comparison}'. "
            f"Supported values: "
            f"{sorted(COMPARISON_FUNCTIONS.keys())}"
        )

    extracted_records, missing_documents = (
        documents_to_numeric_records(
            documents=documents,
            field_name=field_name,
        )
    )

    unique_records, warnings = deduplicate_records(
        extracted_records
    )

    compare = COMPARISON_FUNCTIONS[comparison]

    qualifying_records = [
        record
        for record in unique_records
        if compare(record.value, threshold)
    ]

    excluded_records = [
        record
        for record in unique_records
        if not compare(record.value, threshold)
    ]

    return AggregationResult(
        operation="filter_and_count",
        field_name=field_name,
        comparison=comparison,
        threshold=threshold,
        count=len(qualifying_records),
        all_records=unique_records,
        qualifying_records=qualifying_records,
        excluded_records=excluded_records,
        missing_documents=missing_documents,
        warnings=warnings,
    )


