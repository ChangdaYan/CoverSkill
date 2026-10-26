from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import torch

from .common import HARD_ROWS, TRAIN_QRELS, TRAIN_QUERIES, iter_jsonl


def select_decomposition_view(payload: dict, view: str = "filtered") -> dict:
    """Return a reader view without mutating the canonical raw cache layer."""
    if view not in {"filtered", "raw"}:
        raise ValueError(f"unsupported decomposition view: {view}")
    if view == "raw":
        if payload.get("raw_generation_preserved") is not True:
            raise RuntimeError("raw decomposition view requested from a legacy cache")
        token_key, alignment_key = "raw_subintent_tokens", "raw_alignments"
    else:
        token_key = (
            "filtered_subintent_tokens"
            if "filtered_subintent_tokens" in payload
            else "subintent_tokens"
        )
        alignment_key = (
            "filtered_alignments" if "filtered_alignments" in payload else "alignments"
        )
    selected = dict(payload)
    selected["subintent_tokens"] = payload.get(token_key, {})
    selected["alignments"] = payload.get(alignment_key, {})
    selected["active_decomposition_view"] = view
    return selected


def load_hard_records(path: Path = HARD_ROWS) -> list[dict]:
    """Load one frozen hard-8192 membership selected from training data only.

    Membership must have been selected by this run's own Stage-1 parent.
    """
    rows = list(iter_jsonl(path))
    if len(rows) != 8192 or len({str(row["query_id"]) for row in rows}) != 8192:
        raise RuntimeError("hard-8192 domain changed")
    query_ids = {str(row["id"]) for row in iter_jsonl(TRAIN_QUERIES)}
    if not {str(row["query_id"]) for row in rows} <= query_ids:
        raise RuntimeError("hard rows escape official train")
    return rows


class HardSampler:
    def __init__(
        self, records: list[dict], seed: int = 43, logical_batch: int = 192
    ) -> None:
        self.records = records
        self.seed = seed
        self.logical_batch = logical_batch

    def batch(self, step: int) -> list[dict]:
        if step < 1:
            raise ValueError("steps are one-based")
        rng = random.Random(self.seed + step * 1_000_003)
        return [
            self.records[index]
            for index in rng.sample(range(len(self.records)), self.logical_batch)
        ]


def load_full_records(*, require_decomposition: dict | None = None) -> list[dict]:
    """Build one canonical row per official-train query from qrels."""
    qrels = load_qrels(TRAIN_QRELS)
    result = []
    subintents = (
        require_decomposition.get("subintent_tokens", {})
        if require_decomposition
        else None
    )
    alignments = (
        require_decomposition.get("alignments", {}) if require_decomposition else None
    )
    for row in iter_jsonl(TRAIN_QUERIES):
        query_id = str(row["id"])
        gold = sorted(qrels.get(query_id, ()))
        if not gold:
            continue
        if subintents is not None:
            branches = subintents.get(query_id, [])[:3]
            branch_alignments = alignments.get(query_id, [])[:3]
            if not branches or not any(branch_alignments):
                continue
        result.append(
            {
                "query_id": query_id,
                "gold_skill_ids": gold,
                "gold_count": len(gold),
                "source": "broad",
            }
        )
    if not result:
        raise RuntimeError("full-train sampler domain is empty")
    return result


class MixedSampler:
    """Deterministic hard/broad mixture with no cross-stage state."""

    def __init__(
        self,
        hard_records: list[dict],
        broad_records: list[dict],
        *,
        hard_fraction: float,
        multi_gold_fraction: float = 0.0,
        seed: int = 43,
        logical_batch: int = 192,
    ) -> None:
        if not 0.0 <= hard_fraction <= 1.0:
            raise ValueError("hard_fraction must be in [0, 1]")
        if not 0.0 <= multi_gold_fraction <= 1.0:
            raise ValueError("multi_gold_fraction must be in [0, 1]")
        self.hard_records = hard_records
        self.broad_records = broad_records
        self.hard_count = round(logical_batch * hard_fraction)
        self.broad_count = logical_batch - self.hard_count
        if self.hard_count and len(hard_records) < self.hard_count:
            raise ValueError("hard pool smaller than requested batch quota")
        if self.broad_count and len(broad_records) < self.broad_count:
            raise ValueError("broad pool smaller than requested batch quota")
        self.seed = seed
        self.multi_gold_fraction = multi_gold_fraction
        self.broad_multi = [
            row
            for row in broad_records
            if int(row.get("gold_count", len(row["gold_skill_ids"]))) >= 2
        ]

    def batch(self, step: int) -> list[dict]:
        if step < 1:
            raise ValueError("steps are one-based")
        rng = random.Random(self.seed + step * 1_000_003)
        rows = []
        if self.hard_count:
            rows.extend(rng.sample(self.hard_records, self.hard_count))
        if self.broad_count:
            multi_count = min(
                len(self.broad_multi),
                round(self.broad_count * self.multi_gold_fraction),
            )
            prioritized = rng.sample(self.broad_multi, multi_count)
            chosen_ids = {str(row["query_id"]) for row in prioritized}
            remaining = [
                row
                for row in self.broad_records
                if str(row["query_id"]) not in chosen_ids
            ]
            rows.extend(prioritized)
            rows.extend(rng.sample(remaining, self.broad_count - multi_count))
        rng.shuffle(rows)
        return rows


def skill_gold_indices(records: list[dict], skill_ids: list[str]) -> list[list[int]]:
    index = {str(value): position for position, value in enumerate(skill_ids)}
    return [
        [index[str(skill_id)] for skill_id in row["gold_skill_ids"]] for row in records
    ]


def load_qrels(path) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for row in iter_jsonl(path):
        if int(row.get("relevance", 1)) > 0:
            result[str(row["query_id"])].add(str(row["skill_id"]))
    return dict(result)


def skill_text(row: dict) -> str:
    name = str(row.get("name") or "").strip()
    description = str(row.get("description") or "").strip()
    body = str(row.get("skill_md") or "").strip() or description
    return f"{name} | {description} | {body}"


def aligned_intents(
    local: list[dict], decomposition: dict, skill_index: dict[str, int]
):
    sequences: list[torch.Tensor] = []
    owners: list[int] = []
    aligned: list[list[tuple[int, float]]] = []
    per_query: list[list[list[tuple[int, float]]]] = [[] for _ in local]
    for owner, row in enumerate(local):
        query_id = str(row["query_id"])
        source_alignments = decomposition.get("alignments", {}).get(query_id, [])[:3]
        for branch, sequence in enumerate(
            decomposition.get("subintent_tokens", {}).get(query_id, [])[:3]
        ):
            sequences.append(sequence)
            owners.append(owner)
            values = (
                source_alignments[branch] if branch < len(source_alignments) else []
            )
            branch_targets = [
                (skill_index[str(skill_id)], float(weight))
                for skill_id, weight in values
                if str(skill_id) in row["gold_skill_ids"] and float(weight) > 0
            ]
            aligned.append(branch_targets)
            per_query[owner].append(branch_targets)
    return sequences, owners, aligned, per_query
