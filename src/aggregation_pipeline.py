"""Deterministic aggregation pipeline for Agentic GraphRAG."""

from __future__ import annotations

import re
import time
from typing import Any

from src.state import (
    AgentState,
    EvidenceItem,
    PipelineType,
    QuestionType,
    ToolCallRecord,
)
from src.tools.aggregate import filter_and_count
from src.tools.document_fetch import LocalDocumentStore
from src.tools.vector_search import SearchResult


def _document_id_from_item(item: Any) -> str | None:
    """Extract a document ID from one retrieved evidence item."""

    if not isinstance(item, dict):
        return None

    for key in (
        "document_id",
        "doc_id",
        "documentId",
        "wikidata_qid",
    ):
        value = item.get(key)

        if value is not None and str(value).strip():
            return str(value).strip().upper()

    # Fall back to extracting Q-ID from a chunk identifier.
    for key in ("chunk_id", "id", "vertex_id"):
        value = item.get(key)

        if value is None:
            continue

        match = re.search(
            r"\b(Q\d+)(?:_chunk_\d+)?\b",
            str(value),
            flags=re.IGNORECASE,
        )

        if match:
            return match.group(1).upper()

    return None


def _chunk_id_from_item(item: Any) -> str | None:
    """Extract a chunk identifier when one is available."""

    if not isinstance(item, dict):
        return None

    for key in ("chunk_id", "chunkId"):
        value = item.get(key)

        if value is not None and str(value).strip():
            return str(value).strip()

    raw_id = item.get("id")

    if raw_id and "_chunk_" in str(raw_id).lower():
        return str(raw_id)

    return None


def _content_from_item(item: Any) -> str:
    """Extract readable content from a retrieved item."""

    if not isinstance(item, dict):
        return str(item)

    for key in (
        "text",
        "content",
        "page_content",
        "document",
        "body",
    ):
        value = item.get(key)

        if value is not None and str(value).strip():
            return str(value).strip()

    return str(item)


def extract_document_ids(
    search_result: SearchResult,
) -> list[str]:
    """Extract unique document IDs while preserving retrieval order."""

    document_ids: list[str] = []
    seen: set[str] = set()

    for item in search_result.evidence:
        document_id = _document_id_from_item(item)

        if document_id is None or document_id in seen:
            continue

        seen.add(document_id)
        document_ids.append(document_id)

    return document_ids


def run_aggregation_pipeline(
    question: str,
    search_result: SearchResult,
    document_store: LocalDocumentStore,
    *,
    field_name: str,
    comparison: str,
    threshold: float,
    expected_document_ids: list[str] | None = None,
    question_id: str | None = None,
) -> AgentState:
    """Run retrieval completion and deterministic aggregation.

    ``expected_document_ids`` is optional. During smoke testing it
    represents the known benchmark scope. In live operation it will be
    omitted and IDs will come from TigerGraph retrieval.
    """

    state = AgentState(
        question=question,
        question_id=question_id,
        pipeline=PipelineType.AGENTIC_RAG,
        question_type=QuestionType.AGGREGATION,
    )

    # Record the retrieval result in our trace.
    state.add_tool_call(
        ToolCallRecord(
            tool_name=search_result.tool_name,
            arguments={
                "question": question,
            },
            success=search_result.ok,
            result_count=len(search_result.evidence),
        )
    )

    # Preserve retrieved chunks as evidence.
    for item in search_result.evidence:
        document_id = _document_id_from_item(item)

        if document_id is None:
            state.add_warning(
                "A retrieved item did not contain a document ID."
            )
            continue

        state.add_evidence(
            EvidenceItem(
                content=_content_from_item(item),
                document_id=document_id,
                chunk_id=_chunk_id_from_item(item),
                source_tool=search_result.tool_name,
                metadata={
                    "retrieval_summary": search_result.summary,
                },
            )
        )

    retrieved_document_ids = extract_document_ids(search_result)

    # In the smoke benchmark, use the known question scope to verify
    # deterministic counting over every gold document.
    document_ids = (
        expected_document_ids
        if expected_document_ids is not None
        else retrieved_document_ids
    )

    if not document_ids:
        state.add_error(
            "No document IDs were available for aggregation."
        )
        state.stop_reason = "no_documents"
        return state

    fetch_started = time.perf_counter()

    documents = document_store.fetch_many(
        document_ids=document_ids,
        required=False,
    )

    fetch_duration = time.perf_counter() - fetch_started
    fetched_ids = {document.document_id for document in documents}

    missing_ids = [
        document_id
        for document_id in document_ids
        if document_id.upper() not in fetched_ids
    ]

    state.add_tool_call(
        ToolCallRecord(
            tool_name="document_fetch",
            arguments={
                "document_ids": document_ids,
            },
            success=len(documents) > 0,
            result_count=len(documents),
            duration_seconds=fetch_duration,
            error=(
                None
                if documents
                else "No complete documents were fetched."
            ),
        )
    )

    for document_id in missing_ids:
        state.add_warning(
            f"Document '{document_id}' could not be fetched."
        )

    aggregate_started = time.perf_counter()

    result = filter_and_count(
        documents=documents,
        field_name=field_name,
        comparison=comparison,
        threshold=threshold,
    )

    aggregate_duration = (
        time.perf_counter() - aggregate_started
    )

    state.add_tool_call(
        ToolCallRecord(
            tool_name="aggregate.filter_and_count",
            arguments={
                "field_name": field_name,
                "comparison": comparison,
                "threshold": threshold,
            },
            success=True,
            result_count=len(result.all_records),
            duration_seconds=aggregate_duration,
        )
    )

    for warning in result.warnings:
        state.add_warning(warning)

    for document_id in result.missing_documents:
        state.add_warning(
            f"Field '{field_name}' was missing from "
            f"document '{document_id}'."
        )

    state.computed_result = {
        "operation": result.operation,
        "field_name": result.field_name,
        "comparison": result.comparison,
        "threshold": result.threshold,
        "count": result.count,
        "documents_requested": len(document_ids),
        "documents_fetched": len(documents),
        "records_extracted": len(result.all_records),
        "qualifying_records": [
            {
                "entity": record.entity,
                "value": record.value,
                "document_id": record.document_id,
                "citation_id": record.citation_id,
            }
            for record in result.qualifying_records
        ],
        "excluded_records": [
            {
                "entity": record.entity,
                "value": record.value,
                "document_id": record.document_id,
            }
            for record in result.excluded_records
        ],
        "missing_documents": (
            missing_ids + result.missing_documents
        ),
    }

    # Only qualifying documents become final citations.
    state.citations = result.citations

    qualifier = (
        f"{comparison} {threshold:g}"
    )

    state.mark_completed(
        final_answer=(
            f"{result.count} events had "
            f"{field_name} {qualifier}."
        ),
        stop_reason="deterministic_aggregation_completed",
    )

    return state
