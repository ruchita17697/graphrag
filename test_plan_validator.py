"""Tests for the Planned-Agent plan validator."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parent

sys.path.insert(
    0,
    str(PROJECT_ROOT / "graphrag" / "app"),
)

sys.path.insert(
    0,
    str(PROJECT_ROOT / "graphrag"),
)


from common.py_schemas import Plan, PlanStep
from agent.agentic_plan_validator import (
    PlanValidationError,
    require_valid_plan,
    validate_plan,
)


QUESTION = (
    "According to the provided corpus, how many biathlon events "
    "at the 2018 Winter Olympics had more than 73 competitors?"
)

KNOWN_TOOLS = [
    "graphrag__get_schema",
    "graphrag__structural_retrieve",
    "graphrag__deterministic_aggregate",
    "graphrag__hybrid_search",
    "graphrag__similarity_search",
    "graphrag__contextual_search",
    "graphrag__community_search",
]


def build_answer_only_plan() -> Plan:
    return Plan(
        strategy="Answer directly.",
        steps=[
            PlanStep(
                id="A",
                kind="answer",
                tool="",
                args={},
                arg_bindings={},
                depends_on=[],
                rationale="Synthesize the final answer.",
            )
        ],
    )


def build_valid_aggregation_plan() -> Plan:
    return Plan(
        strategy=(
            "Retrieve candidate documents, calculate the threshold "
            "count deterministically, and answer from verified evidence."
        ),
        steps=[
            PlanStep(
                id="S1",
                kind="unstructured",
                tool="graphrag__hybrid_search",
                args={
                    "question": (
                        "biathlon events 2018 Winter Olympics "
                        "competitors"
                    ),
                    "top_k": 20,
                },
                arg_bindings={},
                depends_on=[],
                rationale="Retrieve candidate biathlon documents.",
            ),
            PlanStep(
                id="S2",
                kind="structural",
                tool="graphrag__deterministic_aggregate",
                args={
                    "field_name": "competitors",
                    "comparison": ">",
                    "threshold": 73,
                },
                arg_bindings={
                    "document_ids": "S1.document_ids",
                },
                depends_on=["S1"],
                rationale=(
                    "Count events whose competitor value is greater "
                    "than 73."
                ),
            ),
            PlanStep(
                id="A",
                kind="answer",
                tool="",
                args={},
                arg_bindings={},
                depends_on=["S2"],
                rationale="Answer using the deterministic result.",
            ),
        ],
    )


def build_broken_dependency_plan() -> Plan:
    return Plan(
        strategy="Broken dependency example.",
        steps=[
            PlanStep(
                id="S1",
                kind="unstructured",
                tool="graphrag__hybrid_search",
                args={"question": QUESTION},
                arg_bindings={},
                depends_on=["DOES_NOT_EXIST"],
                rationale="Retrieve evidence.",
            ),
            PlanStep(
                id="A",
                kind="answer",
                tool="",
                args={},
                arg_bindings={},
                depends_on=["S1"],
                rationale="Answer.",
            ),
        ],
    )


def print_result(name: str, result) -> None:
    print(f"\n{name}")
    print("Valid:", result.valid)

    if result.errors:
        print("Errors:")

        for error in result.errors:
            print(f"- {error}")
    else:
        print("Errors: none")


def main() -> None:
    with patch(
        "agent.agentic_plan_validator.registry.tool_names",
        return_value=KNOWN_TOOLS,
    ):
        invalid_plan = build_answer_only_plan()

        invalid_result = validate_plan(
            plan=invalid_plan,
            question=QUESTION,
        )

        print_result(
            "Test 1: answer-only aggregation plan",
            invalid_result,
        )

        assert invalid_result.valid is False

        assert any(
            "answer-only" in error
            for error in invalid_result.errors
        )

        assert any(
            "deterministic_aggregate" in error
            for error in invalid_result.errors
        )

        try:
            require_valid_plan(
                plan=invalid_plan,
                question=QUESTION,
            )
        except PlanValidationError as exc:
            print("\nExpected PlanValidationError:")
            print(exc)
        else:
            raise AssertionError(
                "Answer-only plan was not rejected."
            )

        valid_plan = build_valid_aggregation_plan()

        valid_result = validate_plan(
            plan=valid_plan,
            question=QUESTION,
        )

        print_result(
            "Test 2: valid aggregation plan",
            valid_result,
        )

        assert valid_result.valid is True
        assert valid_result.errors == []

        returned_plan = require_valid_plan(
            plan=valid_plan,
            question=QUESTION,
        )

        assert returned_plan is valid_plan

        broken_plan = build_broken_dependency_plan()

        broken_result = validate_plan(
            plan=broken_plan,
            question="Who won the event?",
        )

        print_result(
            "Test 3: broken dependency",
            broken_result,
        )

        assert broken_result.valid is False

        assert any(
            "DOES_NOT_EXIST" in error
            for error in broken_result.errors
        )

    print("\nplan validator tests passed successfully.")


if __name__ == "__main__":
    main()
