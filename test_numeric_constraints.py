"""Fast offline regression tests for strict deterministic numeric answers."""

from src.tools.aggregate import answer_numeric_question, parse_question_constraints
from src.tools.document_fetch import DocumentRecord


def document(document_id: str, title: str, **fields) -> DocumentRecord:
    text = "\n".join([f"TITLE: {title}", *[f"{key}: {value}" for key, value in fields.items()]])
    return DocumentRecord(document_id, title, text, "synthetic.jsonl")


def main() -> None:
    documents = [
        document("Q1", "Biathlon at the 2018 Winter Olympics – Women's sprint", competitors=87),
        document("Q2", "Biathlon at the 2018 Winter Olympics – Men's relay", competitors=73),
        document("Q3", "Biathlon at the 2018 Winter Olympics – Mixed relay", competitors=80),
        document("Q4", "Biathlon at the 2014 Winter Olympics – Women's sprint", competitors=90),
        document("Q5", "Athletics at the 2008 Summer Olympics – Men's marathon", competitors=95),
        document("Q6", "Athletics at the 2008 Summer Olympics – Women's marathon", competitors=82),
        document("Q7", "Athletics at the 2012 Summer Olympics – Men's marathon", competitors=105),
        document("Q8", "Sailing at the 2016 Summer Olympics – Women's RS:X", nations=26),
    ]

    count_question = (
        "According to the provided corpus, how many biathlon events at the "
        "2018 Winter Olympics had more than 73 competitors?"
    )
    constraints = parse_question_constraints(count_question)
    assert constraints.year == 2018
    assert constraints.season == "Winter"
    assert constraints.sport.lower() == "biathlon"
    count = answer_numeric_question(documents, count_question)
    assert count["answer_value"] == 2
    assert {record["document_id"] for record in count["records"]} == {"Q1", "Q3"}

    maximum = answer_numeric_question(
        documents,
        "Which athletics event at the 2008 Summer Olympics had the highest number of competitors?",
    )
    assert maximum["answer_value"] == "Athletics at the 2008 Summer Olympics – Men's marathon"
    assert maximum["records"][0]["value"] == 95

    lookup = answer_numeric_question(
        documents,
        "How many nations competed in Sailing at the 2016 Summer Olympics – Women's RS:X?",
    )
    assert lookup["answer_value"] == 26
    assert lookup["records"][0]["document_id"] == "Q8"

    print("numeric constraint regression tests passed")


if __name__ == "__main__":
    main()
