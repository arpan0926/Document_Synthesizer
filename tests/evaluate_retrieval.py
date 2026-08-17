#!/usr/bin/env python3
"""Standalone retrieval evaluation script.

Usage:
  python tests/evaluate_retrieval.py --eval-file eval_set.json

The script compares retrieval with and without cross-encoder reranking.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

# Ensure the project root is on sys.path so `import retrieval` works when
# running this script directly from the repo root as `python tests/...`.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import retrieval


def percentile_ms(values_ms: List[float], pct: float) -> float:
    if not values_ms:
        return 0.0
    s = sorted(values_ms)
    k = math.ceil((pct / 100.0) * len(s)) - 1
    k = max(0, min(k, len(s) - 1))
    return s[k]


def evaluate(eval_set: List[Dict[str, Any]]):
    modes = [("reranked", True), ("raw", False)]
    results: Dict[str, Any] = {m[0]: {"per_query": [], "latencies_ms": []} for m in modes}
    # candidate_k controls how many candidates are fetched from ChromaDB
    candidate_k = globals().get("CANDIDATE_K", 10)

    for mode_name, rerank_flag in modes:
        for item in eval_set:
            query = item.get("query")
            expected_doc = item.get("expected_source_doc")
            expected_page = item.get("expected_page")

            start = time.perf_counter()
            hits = retrieval.retrieve(query, top_k=candidate_k, rerank=rerank_flag)
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            top3 = hits[:3]

            rank = 0
            for idx, chunk in enumerate(top3, start=1):
                md = chunk.get("metadata", {}) if chunk else {}
                if md.get("source_doc") == expected_doc and md.get("page_number") == expected_page:
                    rank = idx
                    break

            hit = rank > 0

            results[mode_name]["per_query"].append(
                {
                    "query": query,
                    "expected": {"source_doc": expected_doc, "page_number": expected_page},
                    "rank": rank,
                    "hit": hit,
                    "latency_ms": elapsed_ms,
                    "top3_metadata": [c.get("metadata") for c in top3],
                }
            )
            results[mode_name]["latencies_ms"].append(elapsed_ms)

    # Aggregate metrics
    summary: Dict[str, Any] = {}
    for mode_name, _ in modes:
        per_query = results[mode_name]["per_query"]
        n = len(per_query)
        hits = [1 if q["hit"] else 0 for q in per_query]
        recall_at_3 = sum(hits) / n if n else 0.0
        mrr = mean([1.0 / q["rank"] if q["rank"] > 0 else 0.0 for q in per_query]) if n else 0.0
        latencies = results[mode_name]["latencies_ms"]
        mean_latency = mean(latencies) if latencies else 0.0
        p95_latency = percentile_ms(latencies, 95.0)

        summary[mode_name] = {
            "recall_at_3": recall_at_3,
            "mrr": mrr,
            "mean_latency_ms": mean_latency,
            "p95_latency_ms": p95_latency,
        }

    # Print comparison table
    a = summary["reranked"]
    b = summary["raw"]

    def pct_diff(x, y):
        try:
            return ((x - y) / y) * 100.0
        except Exception:
            return float("nan")

    print("\nRetrieval Evaluation Summary\n")
    print(f"{'Metric':<20}{'Reranked':>12}{'Raw':>12}{'AbsDiff':>12}{'PctDiff':>12}")
    def fval(v, digits=4):
        return f"{v:.{digits}f}"

    rows = [
        ("Recall@3", a["recall_at_3"], b["recall_at_3"]),
        ("MRR", a["mrr"], b["mrr"]),
        ("Mean latency (ms)", a["mean_latency_ms"], b["mean_latency_ms"]),
        ("P95 latency (ms)", a["p95_latency_ms"], b["p95_latency_ms"]),
    ]

    for name, va, vb in rows:
        absd = va - vb
        pd = pct_diff(va, vb)
        print(f"{name:<20}{va:12.4f}{vb:12.4f}{absd:12.4f}{pd:12.2f}%")

    # Per-query breakdown of failures
    print("\nPer-query failures:\n")
    for mode_name, _ in modes:
        fails = [q for q in results[mode_name]["per_query"] if not q["hit"]]
        print(f"{mode_name}: {len(fails)} failures out of {len(results[mode_name]['per_query'])} queries")
        for q in fails:
            print(
                f"- Query: {q['query']}\n  Expected: {q['expected']}\n  Rank: {q['rank']}\n  Latency(ms): {q['latency_ms']:.2f}\n  Top3 metadata: {q['top3_metadata']}\n"
            )

    # Save full results
    out = {"summary": summary, "results": results}
    with open("eval_results.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    print("\nFull results saved to eval_results.json\n")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval reranking vs. raw retrieval")
    parser.add_argument("--eval-file", required=True, help="Path to eval_set.json")
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=10,
        help="Number of candidate chunks to retrieve from the vector store before reranking",
    )
    args = parser.parse_args(argv)

    with open(args.eval_file, "r", encoding="utf-8") as fh:
        eval_set = json.load(fh)

    # Pass candidate_k into evaluate via a closure-like global var
    global CANDIDATE_K
    CANDIDATE_K = args.candidate_k

    evaluate(eval_set)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
