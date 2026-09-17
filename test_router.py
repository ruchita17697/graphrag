from src.router import classify_question
from src.state import QuestionType


test_cases = [
    {
        "question": (
            "How many nations competed in Sailing at the "
            "2016 Summer Olympics - Women's RS:X?"
        ),
        "expected": QuestionType.AGGREGATION,
    },
    {
        "question": (
            "According to the provided corpus, how many biathlon "
            "events at the 2018 Winter Olympics had more than "
            "73 competitors?"
        ),
        "expected": QuestionType.AGGREGATION,
    },
    {
        "question": (
            "Who won the gold medal in the event held at the "
            "Olympic Weightlifting Gymnasium on 20 September 1988?"
        ),
        "expected": QuestionType.MULTI_HOP,
    },
    {
        "question": "When were the 2018 Winter Olympics held?",
        "expected": QuestionType.TEMPORAL,
    },
    {
        "question": "Which athlete won the most gold medals?",
        "expected": QuestionType.SUPERLATIVE,
    },
    {
        "question": "Who won the men's 60 kg weightlifting event?",
        "expected": QuestionType.LOOKUP,
    },
]


for index, case in enumerate(test_cases, start=1):
    decision = classify_question(case["question"])

    print(f"\nTest {index}")
    print("Question:", case["question"])
    print("Detected:", decision.question_type.value)
    print("Expected:", case["expected"].value)
    print("Confidence:", decision.confidence)
    print("Matched rules:", decision.matched_rules)
    print("Required tools:", decision.required_tools)

    assert decision.question_type == case["expected"], (
        f"Test {index} failed: expected "
        f"{case['expected'].value}, but detected "
        f"{decision.question_type.value}"
    )


empty_decision = classify_question("")

assert empty_decision.question_type == QuestionType.UNKNOWN
assert empty_decision.confidence == 0.0

print("\nrouter.py tests passed successfully.")
