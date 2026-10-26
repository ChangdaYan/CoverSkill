"""Freeze train-only, multi-gold prompts before any task-specific training."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import torch

from .common import (
    RAW_BASE_MODEL,
    TRAIN_QRELS,
    TRAIN_QUERIES,
    TRAIN_SKILLS,
    atomic_json,
    iter_jsonl,
    sha256_file,
    write_gzip_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-gold-count", type=int, default=12000)
    parser.add_argument(
        "--focus-policy",
        choices=("random", "weakest_lexical"),
        default="weakest_lexical",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"immutable task file already exists: {args.output}")
    queries = {str(row["id"]): str(row["query"]) for row in iter_jsonl(TRAIN_QUERIES)}
    skills = {str(row["id"]): row for row in iter_jsonl(TRAIN_SKILLS)}
    gold: dict[str, set[str]] = defaultdict(set)
    for row in iter_jsonl(TRAIN_QRELS):
        if int(row.get("relevance", 1)) > 0:
            gold[str(row["query_id"])].add(str(row["skill_id"]))
    payload = torch.load(args.source_cache, map_location="cpu", weights_only=False)
    if not payload.get("complete") or not payload.get("raw_generation_preserved"):
        raise SystemExit(
            "source Qwen3-32B cache is incomplete or lacks immutable raw generations"
        )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(RAW_BASE_MODEL, local_files_only=False)
    filtered_tokens = payload["filtered_subintent_tokens"]
    filtered_alignments = payload["filtered_alignments"]
    candidates = {2: [], 3: []}
    for qid, skill_ids in gold.items():
        if len(skill_ids) not in candidates or qid not in queries:
            continue
        aligned: dict[str, str] = {}
        for tokens, assignments in zip(
            filtered_tokens.get(qid, []), filtered_alignments.get(qid, [])
        ):
            decoded = tokenizer.decode(tokens.tolist(), skip_special_tokens=True)
            # Cached tensors include the retrieval instruction; retain only
            # the query-supported capability phrase for synthesis guidance.
            decoded = decoded.split("\nQuery: ", 1)[-1].strip()
            for sid, weight in assignments:
                if float(weight) >= 0.5 and sid not in aligned:
                    aligned[str(sid)] = decoded
        facets = []
        for sid in sorted(skill_ids):
            text = " ".join(aligned.get(sid, "").split())
            if not text:
                facets = []
                break
            facets.append(
                {
                    "skill_id": sid,
                    "intent": text[:600],
                    "skill_name": str(skills[sid].get("name") or ""),
                }
            )
        if len(facets) != len(skill_ids):
            continue
        digest = hashlib.sha256(f"coverage-v2:{qid}".encode()).hexdigest()
        if args.focus_policy == "weakest_lexical":
            terms = lambda value: set(re.findall(r"[a-z0-9_]+", value.casefold()))
            query_terms = terms(queries[qid])
            focus = min(
                range(len(facets)),
                key=lambda i: (
                    len(terms(facets[i]["intent"]) & query_terms)
                    / max(1, len(terms(facets[i]["intent"]))),
                    facets[i]["skill_id"],
                ),
            )
        else:
            focus = int(digest[:8], 16) % len(facets)
        candidates[len(skill_ids)].append(
            (
                digest,
                {
                    "parent_query_id": qid,
                    "query": queries[qid],
                    "gold_skill_ids": sorted(skill_ids),
                    "facets": facets,
                    "focus_skill_id": facets[focus]["skill_id"],
                },
            )
        )
    rows = []
    for count in (2, 3):
        rows.extend(row for _, row in sorted(candidates[count])[: args.per_gold_count])
    rows.sort(
        key=lambda row: hashlib.sha256(row["parent_query_id"].encode()).hexdigest()
    )
    write_gzip_jsonl(args.output, rows)
    atomic_json(
        args.output.with_suffix(args.output.suffix + ".manifest.json"),
        {
            "complete": True,
            "count": len(rows),
            "gold_count_2": sum(len(x["gold_skill_ids"]) == 2 for x in rows),
            "gold_count_3": sum(len(x["gold_skill_ids"]) == 3 for x in rows),
            "source_cache_sha256": sha256_file(args.source_cache),
            "source_train_queries_sha256": sha256_file(TRAIN_QUERIES),
            "source_train_qrels_sha256": sha256_file(TRAIN_QRELS),
            "focus_policy": args.focus_policy,
            "task_sha256": sha256_file(args.output),
            "test_labels_used": False,
        },
    )
    print(json.dumps({"tasks": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
