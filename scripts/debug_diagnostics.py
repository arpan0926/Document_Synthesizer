#!/usr/bin/env python3
"""Quick diagnostics for Chroma, retrieval, and Hugging Face inference."""
from __future__ import annotations

import json
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))

def main():
    try:
        from vector_store import _get_collection
        import retrieval
        import generation
    except Exception:
        print("Import error:")
        traceback.print_exc()
        return 1

    print("Checking Chroma collection...")
    try:
        client, coll = _get_collection()
        print("Collection dir:", dir(coll)[:40])
        try:
            info = coll.get()
            # print summary
            ids = info.get("ids") or []
            print(f"Total items in collection (get() ids length): {len(ids)}")
        except Exception as e:
            print("coll.get() failed:", e)
            try:
                # try count()
                cnt = coll.count()
                print("collection.count():", cnt)
            except Exception:
                print("Could not determine collection size; showing repr:")
                print(repr(coll))
    except Exception:
        print("_get_collection() failed:")
        traceback.print_exc()

    sample_query = "What accuracy did the model achieve on the test set?"
    print(f"\nRunning retrieval.retrieve raw (no rerank) for sample query:\n  {sample_query}\n")
    try:
        raw = retrieval.retrieve(sample_query, top_k=10, rerank=False)
        print("Raw results count:", len(raw))
        print(json.dumps(raw, indent=2, default=str))
    except Exception:
        print("Raw retrieval failed:")
        traceback.print_exc()

    print(f"\nRunning retrieval.retrieve reranked for sample query:\n  {sample_query}\n")
    try:
        rer = retrieval.retrieve(sample_query, top_k=10, rerank=True)
        print("Reranked results count:", len(rer))
        print(json.dumps(rer, indent=2, default=str))
    except Exception:
        print("Reranked retrieval failed:")
        traceback.print_exc()

    print("\nTesting Hugging Face inference call (short prompt)...")
    try:
        out = generation._call_huggingface("Say hello in one sentence.")
        print("HF response:", out)
    except Exception:
        print("HF call raised:")
        traceback.print_exc()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
