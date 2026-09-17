from src.tools.aggregate import filter_and_count
from src.tools.document_fetch import LocalDocumentStore


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


store = LocalDocumentStore()

documents = store.fetch_many(GOLD_DOCUMENT_IDS)

result = filter_and_count(
    documents=documents,
    field_name="competitors",
    comparison=">",
    threshold=73,
)


print("Documents examined:", len(documents))
print("Records extracted:", len(result.all_records))
print("Missing fields:", result.missing_documents)
print("Warnings:", result.warnings)

print("\nQualifying events:")

for record in result.qualifying_records:
    print(
        f"- {record.entity}: "
        f"{record.value:g} competitors "
        f"[{record.document_id}]"
    )

print("\nExcluded events:")

for record in result.excluded_records:
    print(
        f"- {record.entity}: "
        f"{record.value:g} competitors "
        f"[{record.document_id}]"
    )

print("\nFinal count:", result.count)
print("Citations:", result.citations)


assert len(documents) == 11
assert len(result.all_records) == 11
assert result.missing_documents == []
assert result.count == 5

qualifying_ids = {
    record.document_id
    for record in result.qualifying_records
}

assert qualifying_ids == {
    "Q47091419",
    "Q47105341",
    "Q47155408",
    "Q47155425",
    "Q47155555",
}

mixed_relay = next(
    record
    for record in result.qualifying_records
    if record.document_id == "Q47155555"
)

assert mixed_relay.value == 80

mens_relay = next(
    record
    for record in result.excluded_records
    if record.document_id == "Q47155541"
)

assert mens_relay.value == 73

print("\naggregate.py tests passed successfully.")
