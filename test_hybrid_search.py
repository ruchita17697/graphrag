from src.tools.hybrid_search import (
    HybridSearchError,
    hybrid_search,
)


QUESTION = (
    "According to the provided corpus, how many biathlon events "
    "at the 2018 Winter Olympics had more than 73 competitors?"
)


def fake_successful_runner(name, args, ctx):
    assert name == "graphrag__hybrid_search"
    assert args["question"] == QUESTION
    assert args["top_k"] == 10
    assert args["num_hops"] == 2
    assert args["chunk_only"] is True
    assert ctx == "fake-context"

    return {
        "ok": True,
        "summary": "GraphRAG_Hybrid_Vector_Search returned 2 item(s)",
        "context": {
            "function_call": "GraphRAG_Hybrid_Vector_Search",
            "result": [
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
        },
        "citations": [
            "q47091419_chunk_0",
            "q47155555_chunk_0",
        ],
    }


def fake_failed_runner(name, args, ctx):
    return {
        "ok": False,
        "summary": "GraphRAG_Hybrid_Vector_Search returned no chunks",
        "context": None,
        "citations": [],
    }


result = hybrid_search(
    "fake-context",
    QUESTION,
    top_k=10,
    num_hops=2,
    chunk_only=True,
    tool_runner=fake_successful_runner,
)

print("Tool:", result.tool_name)
print("Summary:", result.summary)
print("Evidence count:", len(result.evidence))
print("Citations:", result.citations)

assert result.ok is True
assert result.tool_name == "graphrag__hybrid_search"
assert len(result.evidence) == 2
assert result.evidence[1]["document_id"] == "Q47155555"
assert "80 competitors" in result.evidence[1]["text"]

try:
    hybrid_search(
        "fake-context",
        QUESTION,
        tool_runner=fake_failed_runner,
    )
except HybridSearchError as exc:
    print("Expected search error:", exc)
else:
    raise AssertionError("Expected HybridSearchError")

print("\nhybrid_search.py tests passed successfully.")
