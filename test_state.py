from src.state import (
    AgentState,
    EvidenceItem,
    PipelineType,
    QuestionType,
    ToolCallRecord,
)


state = AgentState(
    question=(
        "How many biathlon events at the 2018 Winter Olympics "
        "had more than 73 competitors?"
    ),
    question_id="pub-001",
    pipeline=PipelineType.AGENTIC_RAG,
    question_type=QuestionType.AGGREGATION,
)

state.add_evidence(
    EvidenceItem(
        content="Mixed relay had 80 competitors.",
        document_id="Q47155555",
        chunk_id="q47155555_chunk_0",
        score=0.95,
        source_tool="hybrid_search",
        metadata={
            "event": "Mixed relay",
            "competitors": 80,
        },
    )
)

state.add_tool_call(
    ToolCallRecord(
        tool_name="hybrid_search",
        arguments={"question": state.question, "top_k": 10},
        success=True,
        result_count=1,
        duration_seconds=1.25,
    )
)

state.computed_result = {"count": 5}
state.evidence_verified = True
state.citations_verified = True

state.mark_completed(
    final_answer="Five biathlon events had more than 73 competitors."
)

print("Question type:", state.question_type.value)
print("Evidence count:", len(state.evidence))
print("Citations:", state.citations)
print("Computed result:", state.computed_result)
print("Completed:", state.completed)
print("Stop reason:", state.stop_reason)

assert state.question_type == QuestionType.AGGREGATION
assert len(state.evidence) == 1
assert state.citations == ["q47155555_chunk_0"]
assert state.computed_result["count"] == 5
assert state.completed is True

print("\nstate.py test passed successfully.")
