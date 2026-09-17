"""
Question router for Agentic GraphRAG.

The router classifies questions into:
- lookup
- aggregation
- multi_hop
- temporal
- superlative
- unknown

The initial version is deterministic and does not call an LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.state import QuestionType


@dataclass
class RouterDecision:
    """Result returned by the question router."""

    question_type: QuestionType
    confidence: float
    matched_rules: list[str] = field(default_factory=list)
    required_tools: list[str] = field(default_factory=list)


# Patterns strongly associated with aggregation.
AGGREGATION_PATTERNS = [
    r"\bhow many\b",
    r"\bnumber of\b",
    r"\bcount\b",
    r"\btotal number\b",
    r"\bmore than\b",
    r"\bless than\b",
    r"\bat least\b",
    r"\bat most\b",
    r"\baverage\b",
    r"\bmean\b",
    r"\bsum of\b",
]

# Patterns strongly associated with time or date reasoning.
TEMPORAL_PATTERNS = [
    r"\bwhen\b",
    r"\bwhat year\b",
    r"\bwhich year\b",
    r"\bwhat date\b",
    r"\bwhich date\b",
    r"\bbefore\b",
    r"\bafter\b",
    r"\bearlier\b",
    r"\blater\b",
    r"\bbetween\s+\d{4}\s+and\s+\d{4}\b",
]

# Patterns associated with maximum/minimum comparisons.
SUPERLATIVE_PATTERNS = [
    r"\bmost\b",
    r"\bleast\b",
    r"\bhighest\b",
    r"\blowest\b",
    r"\blargest\b",
    r"\bsmallest\b",
    r"\bmaximum\b",
    r"\bminimum\b",
    r"\bbest\b",
    r"\bworst\b",
    r"\blongest\b",
    r"\bshortest\b",
]

# A single multi-hop keyword is not enough.
# Multiple relationship cues should occur together.
MULTI_HOP_CUES = [
    r"\bheld at\b",
    r"\brepresenting\b",
    r"\bvenue\b",
    r"\bon\s+\d{1,2}\s+[a-z]+\s+\d{4}\b",
    r"\bwon\b",
    r"\bwinner\b",
    r"\bgold medal\b",
    r"\bbelongs to\b",
    r"\bpart of\b",
    r"\bwhose\b",
    r"\bthat competed in\b",
    r"\bwho represented\b",
]

# Direct fact retrieval patterns.
LOOKUP_PATTERNS = [
    r"^\s*who\b",
    r"^\s*what\b",
    r"^\s*which\b",
    r"^\s*where\b",
    r"\bname the\b",
    r"\bidentify\b",
]


TOOLS_BY_TYPE: dict[QuestionType, list[str]] = {
    QuestionType.LOOKUP: [
        "vector_search",
        "document_fetch",
    ],
    QuestionType.AGGREGATION: [
        "hybrid_search",
        "document_fetch",
        "aggregate",
    ],
    QuestionType.MULTI_HOP: [
        "hybrid_search",
        "graph_traversal",
        "document_fetch",
    ],
    QuestionType.TEMPORAL: [
        "hybrid_search",
        "document_fetch",
        "temporal_filter",
    ],
    QuestionType.SUPERLATIVE: [
        "hybrid_search",
        "document_fetch",
        "aggregate",
    ],
    QuestionType.UNKNOWN: [
        "hybrid_search",
        "document_fetch",
    ],
}


def _matched_patterns(
    question: str,
    patterns: list[str],
) -> list[str]:
    """Return all regular expressions that match the question."""

    return [
        pattern
        for pattern in patterns
        if re.search(pattern, question, flags=re.IGNORECASE)
    ]


def classify_question(question: str) -> RouterDecision:
    """
    Classify a question and return routing information.

    Routing priority:
    1. Aggregation
    2. Superlative
    3. Multi-hop
    4. Temporal
    5. Lookup
    6. Unknown

    Aggregation and superlative checks have higher priority because
    questions beginning with "which" or "what" can still require
    counting or comparison.
    """

    cleaned_question = " ".join(question.strip().split())

    if not cleaned_question:
        return RouterDecision(
            question_type=QuestionType.UNKNOWN,
            confidence=0.0,
            matched_rules=[],
            required_tools=TOOLS_BY_TYPE[QuestionType.UNKNOWN],
        )

    aggregation_matches = _matched_patterns(
        cleaned_question,
        AGGREGATION_PATTERNS,
    )

    temporal_matches = _matched_patterns(
        cleaned_question,
        TEMPORAL_PATTERNS,
    )

    superlative_matches = _matched_patterns(
        cleaned_question,
        SUPERLATIVE_PATTERNS,
    )

    multi_hop_matches = _matched_patterns(
        cleaned_question,
        MULTI_HOP_CUES,
    )

    lookup_matches = _matched_patterns(
        cleaned_question,
        LOOKUP_PATTERNS,
    )

    # Questions asking for a count or numeric filtering are aggregation.
    if aggregation_matches:
        question_type = QuestionType.AGGREGATION
        confidence = min(
            0.80 + (0.05 * len(aggregation_matches)),
            0.98,
        )
        matched_rules = aggregation_matches

    # Maximum/minimum questions use candidate collection + comparison.
    elif superlative_matches:
        question_type = QuestionType.SUPERLATIVE
        confidence = min(
            0.80 + (0.05 * len(superlative_matches)),
            0.98,
        )
        matched_rules = superlative_matches

    # Require at least two relationship cues for multi-hop classification.
    elif len(multi_hop_matches) >= 2:
        question_type = QuestionType.MULTI_HOP
        confidence = min(
            0.75 + (0.05 * len(multi_hop_matches)),
            0.98,
        )
        matched_rules = multi_hop_matches

    elif temporal_matches:
        question_type = QuestionType.TEMPORAL
        confidence = min(
            0.75 + (0.05 * len(temporal_matches)),
            0.95,
        )
        matched_rules = temporal_matches

    elif lookup_matches:
        question_type = QuestionType.LOOKUP
        confidence = min(
            0.70 + (0.05 * len(lookup_matches)),
            0.90,
        )
        matched_rules = lookup_matches

    else:
        question_type = QuestionType.UNKNOWN
        confidence = 0.40
        matched_rules = []

    return RouterDecision(
        question_type=question_type,
        confidence=round(confidence, 2),
        matched_rules=matched_rules,
        required_tools=TOOLS_BY_TYPE[question_type],
    )
