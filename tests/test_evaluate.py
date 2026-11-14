"""Tests for the evaluation maths. No database or LLM needed."""

from app.evaluate import first_hit_rank, print_report, retrieval_metrics

RESULTS = [
    {"document": "OSHJ-CoP-04", "clause": "5.4.2", "language": "en"},
    {"document": "OSHJ-CoP-04", "clause": "5.4.1", "language": "ar"},
    {"document": "OSHJ-CoP-02", "clause": "5.8", "language": "en"},
]


def test_rank_of_first_expected_clause_in_any_language():
    expected = [{"document": "OSHJ-CoP-04", "clause": "5.4.1"}]
    assert first_hit_rank(RESULTS, expected) == 2          # found as the Arabic chunk


def test_rank_is_none_when_the_clause_was_not_retrieved():
    assert first_hit_rank(RESULTS, [{"document": "OSHJ-CoP-04", "clause": "7"}]) is None


def test_same_clause_number_in_another_document_does_not_count():
    assert first_hit_rank(RESULTS, [{"document": "OSHJ-CoP-04", "clause": "5.8"}]) is None


def test_hit_rate_and_mrr():
    metrics = retrieval_metrics([1, 2, None, 4])      # four questions
    assert metrics["hit@1"] == 0.25
    assert metrics["hit@3"] == 0.5
    assert metrics["hit@5"] == 0.75
    assert metrics["mrr"] == (1 + 1 / 2 + 0 + 1 / 4) / 4
    assert metrics["questions"] == 4


def test_report_prints_without_error(capsys):
    report = {
        "retrieval": {"all": retrieval_metrics([1, None])},
        "details": [{"id": "q1", "group": "en", "rank": None, "top": ["OSHJ-CoP-04 5.3 (en)"]}],
        "answers": {"cited_correctly": 1, "answerable_asked": 2, "refused_correctly": 1, "unanswerable_asked": 1},
        "answer_model": "test-model",
    }
    print_report(report)
    output = capsys.readouterr().out
    assert "missed q1" in output and "1/2" in output and "test-model" in output
