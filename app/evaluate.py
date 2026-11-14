"""Evaluation: measure how good the system is, with numbers instead of impressions.

    python -m app.evaluate              retrieval only (free, no LLM, about 10 seconds)
    python -m app.evaluate --answers    also check the generated answers (uses the LLM)

The test set is data/eval/questions.json. Each question lists the clause(s) that
contain the answer ("expected"). Questions in the "unanswerable" group have no
expected clause: the right behaviour is to say the documents do not cover it.

Retrieval metrics (is the right clause among the search results?):
  hit@k  share of questions where an expected clause is in the top k results
  MRR    mean reciprocal rank: 1 if the right clause is first, 1/2 if second, ... averaged

Answer metrics (with --answers):
  cited correctly    the answer cites at least one expected clause
  refused correctly  for unanswerable questions, the answer cites nothing
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from app.answer import LLMError, cited_numbers, generate_answer
from app.config import ANTHROPIC_MODEL, GEMINI_MODEL, LLM_PROVIDER
from app.retrieval import retrieve

EVAL_DIR = Path(__file__).resolve().parent.parent / "data" / "eval"
KS = (1, 3, 5, 8)


def first_hit_rank(results: list[dict], expected: list[dict]) -> int | None:
    """Position (1-based) of the first result that is an expected clause, or None.

    A clause counts in either language: the Arabic 5.4.1 is as correct as the English 5.4.1.
    """
    wanted = {(e["document"], e["clause"]) for e in expected}
    for rank, result in enumerate(results, start=1):
        if (result["document"], result["clause"]) in wanted:
            return rank
    return None


def retrieval_metrics(ranks: list[int | None]) -> dict:
    total = len(ranks)
    metrics = {f"hit@{k}": sum(1 for r in ranks if r and r <= k) / total for k in KS}
    metrics["mrr"] = sum(1 / r for r in ranks if r) / total
    metrics["questions"] = total
    return metrics


def evaluate(with_answers: bool) -> dict:
    questions = json.loads((EVAL_DIR / "questions.json").read_text(encoding="utf-8"))
    ranks_by_group: dict[str, list] = defaultdict(list)
    details = []
    answers = {"cited_correctly": 0, "answerable_asked": 0, "refused_correctly": 0, "unanswerable_asked": 0}

    for item in questions:
        results = retrieve(item["question"], top_k=max(KS))
        answerable = bool(item["expected"])
        rank = first_hit_rank(results, item["expected"]) if answerable else None
        if answerable:
            ranks_by_group[item["group"]].append(rank)
        detail = {"id": item["id"], "group": item["group"], "rank": rank,
                  "top": [f"{r['document']} {r['clause']} ({r['language']})" for r in results[:3]]}

        if with_answers:
            try:
                text = generate_answer(item["question"], results)
            except LLMError as error:
                print(f"stopped at {item['id']}: {error}")
                break
            cited = [results[n - 1] for n in cited_numbers(text, len(results))]
            detail["cited"] = [f"{c['document']} {c['clause']}" for c in cited]
            if answerable:
                answers["answerable_asked"] += 1
                detail["answer_ok"] = first_hit_rank(cited, item["expected"]) is not None
                answers["cited_correctly"] += detail["answer_ok"]
            else:
                answers["unanswerable_asked"] += 1
                detail["answer_ok"] = not cited
                answers["refused_correctly"] += detail["answer_ok"]
            print(f"{item['id']}: {'ok' if detail['answer_ok'] else 'WRONG'}", flush=True)
        details.append(detail)

    all_ranks = [rank for ranks in ranks_by_group.values() for rank in ranks]
    report = {
        "retrieval": {"all": retrieval_metrics(all_ranks),
                      **{group: retrieval_metrics(ranks) for group, ranks in ranks_by_group.items()}},
        "details": details,
    }
    if with_answers:
        report["answers"] = answers
        report["answer_model"] = GEMINI_MODEL if LLM_PROVIDER == "gemini" else ANTHROPIC_MODEL
    return report


def print_report(report: dict) -> None:
    print(f"\n{'group':<14}{'questions':>10}" + "".join(f"{'hit@' + str(k):>8}" for k in KS) + f"{'MRR':>8}")
    for group, m in report["retrieval"].items():
        hits = "".join(f"{m[f'hit@{k}']:>8.0%}" for k in KS)
        print(f"{group:<14}{m['questions']:>10}{hits}{m['mrr']:>8.2f}")
    for detail in report["details"]:
        if detail["group"] != "unanswerable" and not (detail["rank"] and detail["rank"] <= max(KS)):
            print(f"  missed {detail['id']}: top results were {detail['top']}")
    if "answers" in report:
        a = report["answers"]
        print(f"\nanswer model: {report['answer_model']}")
        print(f"answers citing an expected clause: {a['cited_correctly']}/{a['answerable_asked']}")
        print(f"unanswerable questions refused:    {a['refused_correctly']}/{a['unanswerable_asked']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval (and optionally answers).")
    parser.add_argument("--answers", action="store_true", help="also generate and check answers (uses the LLM)")
    args = parser.parse_args()
    report = evaluate(args.answers)
    print_report(report)
    name = "results_answers.json" if args.answers else "results_retrieval.json"
    (EVAL_DIR / name).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nsaved to data/eval/{name}")


if __name__ == "__main__":
    main()
