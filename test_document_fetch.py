from src.tools.document_fetch import (
    DocumentNotFoundError,
    LocalDocumentStore,
)


store = LocalDocumentStore()

document_count = store.build_index()

print("Indexed documents:", document_count)
print("Corpus path:", store.corpus_path)
print("Smoke directory:", store.smoke_directory)


mixed_relay = store.fetch("Q47155555")

print("\nDocument ID:", mixed_relay.document_id)
print("Source path:", mixed_relay.source_path)
print("Text preview:")
print(mixed_relay.text[:400])

assert mixed_relay.document_id == "Q47155555"
assert "competitors: 80" in mixed_relay.text.lower()


same_document = store.fetch("q47155555.txt")

assert same_document.document_id == "Q47155555"


documents = store.fetch_many(
    [
        "Q47091419",
        "Q47105341",
        "Q47155408",
        "Q47155425",
        "Q47155555",
    ]
)

assert len(documents) == 5


try:
    store.fetch("Q_DOES_NOT_EXIST")
except DocumentNotFoundError as error:
    print("\nExpected missing-document error:")
    print(error)
else:
    raise AssertionError(
        "Missing document should have raised "
        "DocumentNotFoundError."
    )


print("\ndocument_fetch.py tests passed successfully.")
