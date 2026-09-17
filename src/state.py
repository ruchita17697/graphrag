"""
Shared state models for the three RAG pipelines.

The state records:
- the user's question,
- detected question type,
- retrieved evidence,
- tool execution history,
- computed results,
- citations,
- final answer,
- errors and stopping reason.

Python version: 3.11+
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class QuestionType(str, Enum):
    """Question categories supported by the agent."""

    LOOKUP = "lookup"
    AGGREGATION = "aggregation"
    MULTI_HOP = "multi_hop"
    TEMPORAL = "temporal"
    SUPERLATIVE = "superlative"
    UNKNOWN = "unknown"


class PipelineType(str, Enum):
    """Available RAG pipelines."""

    NORMAL_RAG = "normal_rag"
    GRAPH_RAG = "graph_rag"
    AGENTIC_RAG = "agentic_rag"


@dataclass
class EvidenceItem:
    """One piece of evidence returned by a retrieval tool."""

    content: str
    document_id: str
    chunk_id: str | None = None
    score: float | None = None
    source_tool: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def citation_id(self) -> str:
        """Return the most specific available citation identifier."""

        return self.chunk_id or self.document_id


@dataclass
class ToolCallRecord:
    """Trace of one tool execution."""

    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    result_count: int = 0
    duration_seconds: float = 0.0
    error: str | None = None


@dataclass
class AgentState:
    """Complete state for one question-processing run."""

    question: str
    pipeline: PipelineType

    question_id: str | None = None
    question_type: QuestionType = QuestionType.UNKNOWN

    plan: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[EvidenceItem] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)

    computed_result: Any = None
    final_answer: str | None = None
    citations: list[str] = field(default_factory=list)

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    total_tokens: int = 0
    total_duration_seconds: float = 0.0
    iteration_count: int = 0

    evidence_verified: bool = False
    citations_verified: bool = False
    completed: bool = False
    stop_reason: str | None = None

    def add_evidence(self, item: EvidenceItem) -> bool:
        """
        Add evidence unless the same document/chunk already exists.

        Returns True when added and False when it was a duplicate.
        """

        new_key = (item.document_id.lower(), (item.chunk_id or "").lower())

        existing_keys = {
            (
                evidence.document_id.lower(),
                (evidence.chunk_id or "").lower(),
            )
            for evidence in self.evidence
        }

        if new_key in existing_keys:
            return False

        self.evidence.append(item)

        if item.citation_id not in self.citations:
            self.citations.append(item.citation_id)

        return True

    def add_tool_call(self, call: ToolCallRecord) -> None:
        """Add a tool execution record to the trace."""

        self.tool_calls.append(call)
        self.total_duration_seconds += call.duration_seconds

        if not call.success and call.error:
            self.add_error(
                f"Tool '{call.tool_name}' failed: {call.error}"
            )

    def add_error(self, message: str) -> None:
        """Record an error without adding it twice."""

        if message not in self.errors:
            self.errors.append(message)

    def add_warning(self, message: str) -> None:
        """Record a non-fatal warning."""

        if message not in self.warnings:
            self.warnings.append(message)

    def increment_iteration(self) -> None:
        """Increment the agent iteration counter."""

        self.iteration_count += 1

    def mark_completed(
        self,
        final_answer: str,
        stop_reason: str = "answer_generated",
    ) -> None:
        """Store the final answer and mark the run as complete."""

        self.final_answer = final_answer
        self.completed = True
        self.stop_reason = stop_reason

    def has_evidence(self) -> bool:
        """Return True when at least one evidence item exists."""

        return len(self.evidence) > 0

    def successful_tool_calls(self) -> list[ToolCallRecord]:
        """Return only successful tool calls."""

        return [call for call in self.tool_calls if call.success]

    def failed_tool_calls(self) -> list[ToolCallRecord]:
        """Return only failed tool calls."""

        return [call for call in self.tool_calls if not call.success]

    def to_dict(self) -> dict[str, Any]:
        """Convert the complete state to a JSON-serializable dictionary."""

        return asdict(self)
