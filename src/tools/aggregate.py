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
from typing import Callable, Iterable

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
