"""Create an immutable, stricter training view without changing generated records."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .common import TRAIN_QUERIES, atomic_json, iter_jsonl, sha256_file, utc_now


def quality_reasons(
    original: str, rewrite: str, finish_reason: str | None
) -> list[str]:
    reasons: list[str] = []
    rewrite = rewrite.strip()
    if finish_reason is not None and finish_reason != "stop":
        reasons.append("generation_incomplete")
    if re.search(r"\b(?:a|the|of|to|with|for|like|that|between|named)$", rewrite, re.I):
        reasons.append("sentence_truncated")
    if not re.search(r"[.!?][\"'”’)]?$", rewrite):
        reasons.append("missing_terminal_punctuation")
    numbers = lambda value: set(
        re.findall(r"(?<!\w)\d+(?:[.,]\d+)*(?:%|[A-Za-z]+)?", value)
    )
    if numbers(original) - numbers(rewrite):
        reasons.append("numeric_constraint_changed")
    code_literals = set(re.findall(r"`([^`]+)`", original))
    if any(f"`{literal}`" not in rewrite for literal in code_literals):
        reasons.append("code_identifier_changed")
    return reasons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-accepted", type=int, default=5000)
    args = parser.parse_args()
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    if args.output.exists() or manifest_path.exists():
        raise RuntimeError("curated rewrite output is immutable")
    originals = {str(row["id"]): str(row["query"]) for row in iter_jsonl(TRAIN_QUERIES)}
    rows = []
    rejected_by_quality = 0
    for row in iter_jsonl(args.source):
        record = dict(row)
        qid = str(record["parent_query_id"])
        if qid not in originals:
            raise RuntimeError(f"rewrite parent is outside train: {qid}")
        checks = quality_reasons(
            originals[qid],
            str(record.get("rewrite", "")),
            record.get("generation_finish_reason"),
        )
        if checks and record.get("accepted"):
            rejected_by_quality += 1
        record["accepted"] = bool(record.get("accepted")) and not checks
        record["curation_reasons"] = checks
        rows.append(record)
    if len({str(row["parent_query_id"]) for row in rows}) != len(rows):
        raise RuntimeError("duplicate rewrite parent")
    accepted = sum(bool(row["accepted"]) for row in rows)
    if accepted < args.min_accepted:
        raise RuntimeError(
            f"only {accepted} curated rewrites, below {args.min_accepted}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    atomic_json(
        manifest_path,
        {
            "complete": True,
            "source": str(args.source),
            "source_sha256": sha256_file(args.source),
            "train_queries_sha256": sha256_file(TRAIN_QUERIES),
            "output_sha256": sha256_file(args.output),
            "candidates": len(rows),
            "accepted": accepted,
            "additional_quality_rejections": rejected_by_quality,
            "official_test_used": False,
            "updated_utc": utc_now(),
        },
    )
    print(
        f"curated {accepted}/{len(rows)} accepted; additional quality rejections={rejected_by_quality}"
    )


if __name__ == "__main__":
    main()
