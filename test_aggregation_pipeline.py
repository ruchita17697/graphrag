from src.aggregation_pipeline import run_aggregation_pipeline
from src.tools.document_fetch import LocalDocumentStore
from src.tools.vector_search import SearchResult


QUESTION = (
    "According to the provided corpus, how many biathlon "
    "events at the 2018 Winter Olympics had more than "
    "73 competitors?"
)

GOLD_DOCUMENT_IDS = [
    "Q47091419",
    "Q47105341",
    "Q47155365",
    "Q47155371",
    "Q47155408",
    "Q47155425",
    "Q47155467",
    "Q47155505",
    "Q47155541",
    "Q47155555",
    "Q47155815",
]

EXPECTED_QUALIFYING_IDS = {
    "Q47091419",  # Women's sprint: 87
    "Q47105341",  # Men's sprint: 87
    "Q47155408",  # Men's individual: 86
    "Q47155425",  # Women's individual: 87
    "Q47155555",  # Mixed relay: 80
}


# Simulated Hybrid Search output. The pipeline then fetches the
# complete eleven documents from the actual local smoke corpus.
search_result = SearchResult(
    question=QUESTION,
    tool_name="graphrag__hybrid_search",
    evidence=[
        {
            "chunk_id": "q47091419_chunk_0",
            "document_id": "Q47091419",
            "text": "Women's sprint had 87 competitors.",
        },
        {
            "chunk_id": "q47155555_chunk_0",
            "document_id": "Q47155555",
            "text": "Mixed relay had 80 competitors.",
        },
    ],
    citations=[
        "q47091419_chunk_0",
        "q47155555_chunk_0",
    ],
    summary=(
        "GraphRAG_Hybrid_Vector_Search "
        "returned 2 item(s)"
    ),
    ok=True,
)

store = LocalDocumentStore()

state = run_aggregation_pipeline(
    question=QUESTION,
    question_id="pub-001",
    search_result=search_result,
    document_store=store,
    field_name="competitors",
    comparison=">",
    threshold=73,
    expected_document_ids=GOLD_DOCUMENT_IDS,
)

print("Question type:", state.question_type.value)
print(
    "Documents requested:",
    state.computed_result["documents_requested"],
)
print(
    "Documents fetched:",
    state.computed_result["documents_fetched"],
)
print(
    "Records extracted:",
    state.computed_result["records_extracted"],
)
print("Computed count:", state.computed_result["count"])
print("Final answer:", state.final_answer)

print("\nQualifying records:")

for record in state.computed_result["qualifying_records"]:
    print(
        f"- {record['document_id']}: "
        f"{record['entity']} = {record['value']:g}"
    )

print("\nCitations:", state.citations)
print("Warnings:", state.warnings)
print("Errors:", state.errors)
print("Completed:", state.completed)
print("Stop reason:", state.stop_reason)

assert state.question_type.value == "aggregation"
assert state.computed_result["documents_requested"] == 11
assert state.computed_result["documents_fetched"] == 11
assert state.computed_result["records_extracted"] == 11
assert state.computed_result["count"] == 5

assert set(state.citations) == EXPECTED_QUALIFYING_IDS
assert len(state.citations) == 5

assert state.completed is True
assert state.errors == []
assert state.stop_reason == (
    "deterministic_aggregation_completed"
)

print(
    "\naggregation_pipeline.py test "
    "passed successfully."
)
