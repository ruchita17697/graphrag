from src.tools.vector_search import (
    VectorSearchError,
    vector_search,
)


def fake_successful_runner(name, args, ctx):
    assert name == "graphrag__similarity_search"
    assert args["question"] == "Who won the event?"
    assert args["top_k"] == 5
    assert ctx == "fake-context"

    return {
        "ok": True,
        "summary": "Content_Similarity_Vector_Search returned 1 item(s)",
        "context": {
            "function_call": "Content_Similarity_Vector_Search",
            "result": [
                {
                    "chunk_id": "q25239316_chunk_0",
                    "document_id": "Q25239316",
                    "text": "Naim Süleymanoğlu won the gold medal.",
                }
            ],
        },
        "citations": ["q25239316_chunk_0"],
    }


def fake_failed_runner(name, args, ctx):
    return {
        "ok": False,
        "summary": "Content_Similarity_Vector_Search returned no chunks",
        "context": None,
        "citations": [],
    }


result = vector_search(
    "fake-context",
    "Who won the event?",
    top_k=5,
    tool_runner=fake_successful_runner,
)

print("Tool:", result.tool_name)
print("Summary:", result.summary)
print("Evidence count:", len(result.evidence))
print("Citations:", result.citations)
print("First evidence:", result.evidence[0])

assert result.ok is True
assert len(result.evidence) == 1
assert result.evidence[0]["document_id"] == "Q25239316"
assert result.citations == ["q25239316_chunk_0"]

try:
    vector_search(
        "fake-context",
        "Missing question",
        tool_runner=fake_failed_runner,
    )
except VectorSearchError as exc:
    print("Expected search error:", exc)
else:
    raise AssertionError("Expected VectorSearchError")

print("\nvector_search.py tests passed successfully.")
