"""Step 4: measure the agent on a test set.  Run with:  python evaluate.py

eval_questions.json format (copy eval_questions.json and write ~20 of your own):
[
  {"question": "...", "expected_keywords": ["30 days"], "answerable": true},
  {"question": "Who won the 2018 World Cup?", "answerable": false}
]
Answerable questions pass if the answer contains ALL expected keywords.
Unanswerable questions pass if the agent declines (fallback) instead of guessing.
"""
import json
import sys
import time

import config
from agent import ask, build_graph, get_retriever

QUESTIONS_FILE = config.BASE_DIR / "eval_questions.json"

# An out-of-scope question counts as handled if the agent falls back OR says the
# documents don't contain the answer. (Still read eval_results.json to confirm.)
DECLINE_PHRASES = [
    "couldn't find", "could not find", "does not specify", "doesn't specify",
    "do not specify", "not specified", "does not contain", "doesn't contain",
    "does not mention", "doesn't mention", "not mentioned", "no information",
    "not provided", "don't know", "do not know",
]


def declined(result: dict) -> bool:
    answer = result["answer"].lower()
    return result["fallback"] or any(phrase in answer for phrase in DECLINE_PHRASES)


def main() -> None:
    if not QUESTIONS_FILE.exists():
        sys.exit("eval_questions.json not found. Copy eval_questions.json and edit it.")

    cases = json.loads(QUESTIONS_FILE.read_text())

    # Fail fast if the vector index is missing or empty (otherwise every question "fails")
    if not config.index_exists() or not get_retriever().invoke("prior authorization"):
        sys.exit(
            "The vector index is empty or missing. Stop the Streamlit app, run "
            "'python ingest.py', check it reports 12 chunks, then run this again."
        )
    agent = build_graph()

    rows, latencies = [], []
    for case in cases:
        start = time.time()
        result = ask(agent, case["question"])
        latencies.append(time.time() - start)

        if case.get("answerable", True):
            text = result["answer"].lower()
            passed = not result["fallback"] and all(
                kw.lower() in text for kw in case.get("expected_keywords", [])
            )
            cited = bool(result["sources"])
        else:
            passed = declined(result)
            cited = None
        rows.append({**case, "passed": passed, "cited": cited, "result": result})
        print(f"[{'PASS' if passed else 'FAIL'}] {case['question']}")

    answerable = [r for r in rows if r.get("answerable", True)]
    unanswerable = [r for r in rows if not r.get("answerable", True)]
    n_pass = sum(r["passed"] for r in rows)

    print("\n=== Results ===")
    print(f"Overall accuracy:        {n_pass}/{len(rows)} ({n_pass / len(rows):.0%})")
    if answerable:
        ok = sum(r["passed"] for r in answerable)
        cited = sum(bool(r["cited"]) for r in answerable)
        print(f"Answerable correct:      {ok}/{len(answerable)}")
        print(f"Answers with citations:  {cited}/{len(answerable)}")
    if unanswerable:
        ok = sum(r["passed"] for r in unanswerable)
        strict = sum(r["result"]["fallback"] for r in unanswerable)
        print(f"Correctly declined:      {ok}/{len(unanswerable)} "
              f"({strict} via fallback, {ok - strict} via 'not specified' answers)")
    print(f"Avg latency:             {sum(latencies) / len(latencies):.1f}s")
    print(f"Queries needing rewrite: {sum(r['result']['retries'] > 0 for r in rows)}/{len(rows)}")

    out = config.BASE_DIR / "eval_results.json"
    out.write_text(json.dumps(
        [{"question": r["question"], "passed": r["passed"], "answer": r["result"]["answer"],
          "sources": r["result"]["sources"], "retries": r["result"]["retries"]} for r in rows],
        indent=2))
    print(f"Details saved to {out.name}")


if __name__ == "__main__":
    main()
