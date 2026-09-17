"""
Local document retrieval for the GraphRAG project.

This module retrieves complete source documents by document ID.

It supports:
1. The complete JSONL corpus.
2. Individual smoke-test TXT documents.

The local implementation gives us a deterministic source of truth.
Later, we can add a TigerGraph-backed document fetcher using the
same DocumentRecord output structure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class DocumentRecord:
    """Normalized representation of one corpus document."""

    document_id: str
    title: str
    text: str
    source_path: str
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentNotFoundError(LookupError):
    """Raised when a requested document ID cannot be found."""


class LocalDocumentStore:
    """Retrieve documents from the local hackathon dataset."""

    DEFAULT_CORPUS_CANDIDATES = [
        Path("hackathon_data/corpus/corpus.jsonl"),
        Path("hackathon_data/corpus.jsonl"),
        Path("corpus.jsonl"),
    ]

    DEFAULT_SMOKE_DIRECTORY = Path(
        "hackathon_data/smoke_test_documents"
    )

    def __init__(
        self,
        corpus_path: str | Path | None = None,
        smoke_directory: str | Path | None = None,
    ) -> None:
        self.corpus_path = (
            Path(corpus_path)
            if corpus_path is not None
            else self._find_default_corpus()
        )

        self.smoke_directory = (
            Path(smoke_directory)
            if smoke_directory is not None
            else self.DEFAULT_SMOKE_DIRECTORY
        )

        self._index: dict[str, DocumentRecord] | None = None

    @classmethod
    def _find_default_corpus(cls) -> Path | None:
        """Return the first existing default corpus path."""

        for candidate in cls.DEFAULT_CORPUS_CANDIDATES:
            if candidate.exists():
                return candidate

        return None

    @staticmethod
    def _normalize_id(document_id: str) -> str:
        """Normalize document IDs for case-insensitive lookup."""

        normalized = document_id.strip()

        if normalized.lower().endswith(".txt"):
            normalized = normalized[:-4]

        return normalized.upper()

    @staticmethod
    def _extract_document_id(data: dict[str, Any]) -> str | None:
        """Read a document ID from common corpus field names."""

        for key in (
            "document_id",
            "doc_id",
            "id",
            "qid",
            "wikidata_qid",
        ):
            value = data.get(key)

            if value is not None and str(value).strip():
                return str(value).strip()

        return None

    @staticmethod
    def _extract_title(data: dict[str, Any]) -> str:
        """Read a title from common corpus field names."""

        for key in ("title", "name", "heading"):
            value = data.get(key)

            if value is not None and str(value).strip():
                return str(value).strip()

        return ""

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        """Read document content from common corpus field names."""

        for key in (
            "text",
            "content",
            "body",
            "document",
            "page_content",
        ):
            value = data.get(key)

            if value is not None and str(value).strip():
                return str(value).strip()

        return ""

    def _record_from_json(
        self,
        data: dict[str, Any],
        source_path: Path,
    ) -> DocumentRecord | None:
        """Convert one JSON object into a normalized document."""

        raw_id = self._extract_document_id(data)

        if raw_id is None:
            return None

        document_id = self._normalize_id(raw_id)
        title = self._extract_title(data)
        text = self._extract_text(data)

        excluded_keys = {
            "document_id",
            "doc_id",
            "id",
            "qid",
            "wikidata_qid",
            "title",
            "name",
            "heading",
            "text",
            "content",
            "body",
            "document",
            "page_content",
        }

        metadata = {
            key: value
            for key, value in data.items()
            if key not in excluded_keys
        }

        return DocumentRecord(
            document_id=document_id,
            title=title,
            text=text,
            source_path=str(source_path),
            metadata=metadata,
        )

    def _read_jsonl(
        self,
        corpus_path: Path,
    ) -> Iterable[DocumentRecord]:
        """Yield normalized documents from a JSONL file."""

        with corpus_path.open(
            "r",
            encoding="utf-8-sig",
        ) as corpus_file:
            for line_number, line in enumerate(
                corpus_file,
                start=1,
            ):
                line = line.strip()

                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON on line {line_number} "
                        f"of {corpus_path}: {error}"
                    ) from error

                if not isinstance(data, dict):
                    continue

                record = self._record_from_json(
                    data=data,
                    source_path=corpus_path,
                )

                if record is not None:
                    yield record

    def _read_smoke_documents(
        self,
        directory: Path,
    ) -> Iterable[DocumentRecord]:
        """Yield documents from the smoke-test TXT directory."""

        if not directory.exists():
            return

        for path in directory.glob("Q*.txt"):
            document_id = self._normalize_id(path.stem)

            text = path.read_text(
                encoding="utf-8-sig",
                errors="replace",
            ).strip()

            yield DocumentRecord(
                document_id=document_id,
                title="",
                text=text,
                source_path=str(path),
                metadata={
                    "source_type": "smoke_test_txt",
                },
            )

    def build_index(self, force: bool = False) -> int:
        """
        Build an in-memory document index.

        Full corpus records are loaded first. Smoke documents then
        overwrite matching entries because the smoke files contain
        exactly what was uploaded during the smoke test.
        """

        if self._index is not None and not force:
            return len(self._index)

        index: dict[str, DocumentRecord] = {}

        if self.corpus_path is not None:
            if not self.corpus_path.exists():
                raise FileNotFoundError(
                    f"Corpus file does not exist: "
                    f"{self.corpus_path}"
                )

            for record in self._read_jsonl(self.corpus_path):
                index[record.document_id] = record

        for record in self._read_smoke_documents(
            self.smoke_directory
        ):
            index[record.document_id] = record

        self._index = index
        return len(index)

    def fetch(
        self,
        document_id: str,
        required: bool = True,
    ) -> DocumentRecord | None:
        """
        Fetch one document by ID.

        IDs are case-insensitive. The '.txt' suffix is optional.
        """

        if self._index is None:
            self.build_index()

        normalized_id = self._normalize_id(document_id)
        record = self._index.get(normalized_id)

        if record is None and required:
            raise DocumentNotFoundError(
                f"Document '{document_id}' was not found."
            )

        return record

    def fetch_many(
        self,
        document_ids: Iterable[str],
        required: bool = True,
    ) -> list[DocumentRecord]:
        """Fetch several documents while preserving input order."""

        documents: list[DocumentRecord] = []

        for document_id in document_ids:
            record = self.fetch(
                document_id=document_id,
                required=required,
            )

            if record is not None:
                documents.append(record)

        return documents
