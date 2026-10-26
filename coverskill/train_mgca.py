#!/usr/bin/env python3
"""Stage 1: Multi-Granularity Coverage Alignment (MGCA)."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import math
import os
import random
import re
import shutil
import socket
import sys
import time
import datetime as dt
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path

_CLEAN_SCRIPTS = Path(__file__).resolve().parent
if str(_CLEAN_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_CLEAN_SCRIPTS))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import torch.distributed as dist
from datasets import Dataset, load_from_disk
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.sentence_transformer.losses.cached_multiple_negatives_ranking import (
    CachedMultipleNegativesRankingLoss,
    _backward_hook,
)
from sentence_transformers.util import all_gather_with_grad
from torch.utils.data import BatchSampler, Sampler

from .common import RAW_BASE_MODEL, QUERY_PROMPT as SKILL_QUERY_PROMPT
from .views import clean

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)
RANK = int(os.environ.get("RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
MAX_GOLDS = 3


_FACET_KEYWORDS = {
    "interface": (
        "input",
        "output",
        "parameter",
        "argument",
        "api",
        "endpoint",
        "schema",
        "config",
        "auth",
    ),
    "procedure": (
        "usage",
        "workflow",
        "step",
        "example",
        "how to",
        "process",
        "pipeline",
        "implement",
    ),
    "constraints": (
        "limit",
        "constraint",
        "security",
        "guardrail",
        "error",
        "warning",
        "require",
        "must",
        "avoid",
    ),
}


def _terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9_+.#/-]+", text.casefold())
        if len(token) > 2
    }


def semantic_facets(skill: dict) -> dict[str, str]:
    """Build interpretable Markdown facets instead of arbitrary token prefixes."""
    name = clean(skill.get("name"))
    description = clean(skill.get("description"))
    body = str(skill.get("skill_md") or skill.get("body") or "")
    identity = clean(f"{name} | {description}")
    buckets: dict[str, list[str]] = {key: [] for key in _FACET_KEYWORDS}
    current = "procedure"
    for raw in body.splitlines():
        line = clean(raw)
        if not line:
            continue
        lowered = line.casefold()
        if raw.lstrip().startswith("#"):
            matches = [
                facet
                for facet, words in _FACET_KEYWORDS.items()
                if any(word in lowered for word in words)
            ]
            current = matches[0] if matches else "procedure"
        direct = [
            facet
            for facet, words in _FACET_KEYWORDS.items()
            if any(word in lowered for word in words)
        ]
        bucket = direct[0] if direct else current
        if len(" ".join(buckets[bucket]).split()) < 640:
            buckets[bucket].append(line)
    facets = {"identity": identity}
    for facet, lines in buckets.items():
        text = clean(f"{name} | {description} | {' '.join(lines)}")
        if text and text != identity:
            facets[facet] = text
    return facets


def routed_semantic_facet(skill: dict, routing_text: str) -> tuple[str, str]:
    return route_semantic_facets(semantic_facets(skill), routing_text)


def route_semantic_facets(facets: dict[str, str], routing_text: str) -> tuple[str, str]:
    candidates = [(name, text) for name, text in facets.items() if name != "identity"]
    if not candidates:
        return "identity", facets["identity"]
    route_terms = _terms(routing_text)
    ranked = []
    for name, text in candidates:
        facet_terms = _terms(text)
        overlap = len(route_terms & facet_terms) / max(1, len(route_terms))
        ranked.append((overlap, name, text))
    _, name, text = max(ranked, key=lambda row: (row[0], row[1]))
    return name, text


def read_jsonl(path: str | Path) -> list[dict]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_train_from_paths(
    queries_path: Path, qrels_path: Path, skills_path: Path
) -> tuple[list[dict], dict[str, dict]]:
    query_rows = read_jsonl(queries_path)
    skills = {str(row["id"]): row for row in read_jsonl(skills_path)}
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in read_jsonl(qrels_path):
        if int(row.get("relevance", 1)) > 0:
            grouped[str(row["query_id"])].append(str(row["skill_id"]))
    queries = []
    for row in query_rows:
        qid = str(row["id"])
        gold = list(dict.fromkeys(grouped.get(qid, [])))
        if gold:
            queries.append({"id": qid, "query": str(row["query"]), "skill_ids": gold})
    return queries, skills


def select_decomposition_view(payload: dict, view: str) -> dict:
    """Select raw or filtered tensors without changing the raw generation records."""
    if view not in {"filtered", "raw"}:
        raise ValueError(f"unsupported decomposition view: {view}")
    if view == "raw":
        if payload.get("raw_generation_preserved") is not True:
            raise RuntimeError(
                "raw decomposition view requested from a unsupported cache"
            )
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
    return selected


def load_aligned_subintents(
    path: Path, tokenizer, view: str
) -> tuple[dict[str, dict[str, str]], dict]:
    payload = select_decomposition_view(
        torch.load(path, map_location="cpu", weights_only=False), view
    )
    output: dict[str, dict[str, str]] = {}
    active = 0
    for qid, token_rows in payload["subintent_tokens"].items():
        alignments = payload.get("alignments", {}).get(qid, [])
        mapped: dict[str, str] = {}
        for index, tokens in enumerate(token_rows):
            aligned = alignments[index] if index < len(alignments) else []
            text = tokenizer.decode(tokens.tolist(), skip_special_tokens=True)
            for sid, weight in aligned:
                if float(weight) >= 0.5 and str(sid) not in mapped:
                    mapped[str(sid)] = text
                    active += 1
        if mapped:
            output[str(qid)] = mapped
    return output, {
        "queries_with_aligned_subintents": len(output),
        "aligned_query_gold_slots": active,
        "decomposition_view": view,
    }


def stable_ids(values: list[str]) -> dict[str, int]:
    return {value: index + 1 for index, value in enumerate(sorted(set(values)))}


def mgca_options(
    query: dict,
    skills: dict[str, dict],
    seed: int,
    subintents: dict[str, dict[str, str]],
    positive_views: str,
    semantic_facet_cache: dict[str, dict[str, str]],
    routed_view_cache: dict[tuple[str, str], tuple[str, str]],
) -> list[tuple[str, str, str]]:
    qid = query["id"]
    gold_ids = [sid for sid in query.get("skill_ids", []) if sid in skills]
    options: list[tuple[str, str, str]] = []
    for positive_slot, sid in enumerate(gold_ids):
        if positive_views == "semantic_routed":
            facets = semantic_facet_cache[sid]
            facet_name, facet_text = routed_view_cache[(qid, sid)]
            # Identity preserves the deployment geometry; the routed facet
            # supplies a query-supported, interpretable completion view.
            options.append((sid, "semantic_identity", facets["identity"]))
            if facet_name != "identity":
                options.append((sid, f"semantic_{facet_name}", facet_text))
            continue
    if not options:
        raise ValueError(f"Query {qid} has no valid Multi-granularity options")
    return options


def make_row(
    query: dict,
    selected: tuple[str, str, str],
    cycle: int,
    query_ids: dict[str, int],
    skill_ids: dict[str, int],
    skills: dict[str, dict],
    group_aux: bool,
    family_hn_ratio: float,
    by_family: dict[str, list[str]],
    by_family_capability: dict[tuple[str, str], list[str]],
    seed: int,
    subintents: dict[str, dict[str, str]],
    routed_view_cache: dict[tuple[str, str], tuple[str, str]],
) -> tuple[dict, dict]:
    qid = query["id"]
    gold_sids = [sid for sid in query.get("skill_ids", []) if sid in skills]
    if not 1 <= len(gold_sids) <= MAX_GOLDS:
        raise ValueError(f"Unexpected gold count for {qid}: {len(gold_sids)}")
    selected_sid, view_name, positive_text = selected
    padded = gold_sids + [gold_sids[0]] * (MAX_GOLDS - len(gold_sids))
    label = [query_ids[qid], skill_ids[selected_sid], len(gold_sids)]
    label.extend(skill_ids[sid] for sid in padded)
    row: dict = {
        "anchor": query["query"],
        "positive": positive_text,
    }
    subintent_text = subintents.get(qid, {}).get(selected_sid)
    row["subintent"] = subintent_text or query["query"]
    stats = {
        "view": view_name,
        "hard_negative": False,
        "hard_negative_exact_capability": False,
    }
    label.extend([0, skill_ids[selected_sid]])
    label.append(int(subintent_text is not None))
    if group_aux:
        for slot, sid in enumerate(padded, start=1):
            _, routed = routed_view_cache[(qid, sid)]
            row[f"group_positive_{slot}"] = routed
    row["label"] = label
    return row, stats


def build_dataset(
    queries: list[dict],
    skills: dict[str, dict],
    query_unique: bool,
    group_aux: bool,
    family_hn_ratio: float,
    seed: int,
    max_pairs: int,
    global_batch: int,
    subintents: dict[str, dict[str, str]],
    positive_views: str,
    coverage_rewrites: dict[str, str] | None = None,
    coverage_rewrite_cycles: int = 1,
    progress_callback=None,
) -> tuple[Dataset, dict, list[int] | None]:
    coverage_rewrites = coverage_rewrites or {}
    usable_queries = [
        query
        for query in queries
        if query.get("query")
        and all(sid in skills for sid in query.get("skill_ids", []))
        and query.get("skill_ids")
    ]
    query_ids = stable_ids([query["id"] for query in usable_queries])
    skill_ids = stable_ids(list(skills))
    needs_semantic_routing = positive_views == "semantic_routed" or group_aux
    semantic_facet_cache = {}
    if needs_semantic_routing:
        for index, (sid, skill) in enumerate(skills.items(), 1):
            semantic_facet_cache[sid] = semantic_facets(skill)
            if progress_callback and (index % 1000 == 0 or index == len(skills)):
                progress_callback("semantic_facets", index, len(skills))
    routed_view_cache: dict[tuple[str, str], tuple[str, str]] = {}
    if needs_semantic_routing:
        for query_index, query in enumerate(usable_queries, 1):
            qid = query["id"]
            for sid in query.get("skill_ids", []):
                routing_text = subintents.get(qid, {}).get(sid) or str(
                    query.get("query", "")
                )
                routed_view_cache[(qid, sid)] = route_semantic_facets(
                    semantic_facet_cache[sid], routing_text
                )
            if progress_callback and (
                query_index % 5000 == 0 or query_index == len(usable_queries)
            ):
                progress_callback(
                    "query_gold_routing", query_index, len(usable_queries)
                )
    options = {
        query["id"]: mgca_options(
            query,
            skills,
            seed,
            subintents,
            positive_views,
            semantic_facet_cache,
            routed_view_cache,
        )
        for query in usable_queries
    }
    by_family, by_family_capability = {}, {}
    rows: list[dict] = []
    counts: Counter[str] = Counter()
    sampler_order: list[int] | None = [] if query_unique else None

    if query_unique:
        cycles = max(1, math.ceil(max_pairs / len(usable_queries)))
        if coverage_rewrites and cycles < coverage_rewrite_cycles + 1:
            raise ValueError(
                "coverage augmentation needs an original cycle and the requested rewrite cycles"
            )
        for cycle in range(cycles):
            cycle_indices: list[int] = []
            for query in usable_queries:
                qid = query["id"]
                offset = query_ids[qid] + seed
                selected = options[qid][(cycle + offset) % len(options[qid])]
                # Use exactly one view of each original query in each cycle.
                # This preserves global query-unique batching, complete gold
                # labels, sibling masking, and the original per-gold subintent.
                row_query = query
                if 1 <= cycle <= coverage_rewrite_cycles and qid in coverage_rewrites:
                    row_query = {**query, "query": coverage_rewrites[qid]}
                row, row_stats = make_row(
                    row_query,
                    selected,
                    cycle,
                    query_ids,
                    skill_ids,
                    skills,
                    group_aux,
                    family_hn_ratio,
                    by_family,
                    by_family_capability,
                    seed,
                    subintents,
                    routed_view_cache,
                )
                cycle_indices.append(len(rows))
                rows.append(row)
                counts[f"view_{row_stats['view']}"] += 1
                counts["family_hard_negative_slots"] += int(row_stats["hard_negative"])
                counts["family_hard_negative_exact_capability"] += int(
                    row_stats["hard_negative_exact_capability"]
                )
                if progress_callback and len(rows) % 10000 == 0:
                    progress_callback(
                        "training_row_build", min(len(rows), max_pairs), max_pairs
                    )
            rng = random.Random(seed * 1_000_003 + cycle)
            rng.shuffle(cycle_indices)
            usable = len(cycle_indices) - (len(cycle_indices) % global_batch)
            sampler_order.extend(cycle_indices[:usable])
        if not sampler_order:
            raise ValueError(
                "No complete query-unique global batch could be constructed"
            )
    else:
        all_options: list[tuple[dict, tuple[str, str, str]]] = []
        for query in usable_queries:
            all_options.extend((query, selected) for selected in options[query["id"]])
        random.Random(seed).shuffle(all_options)
        cycle = 0
        while len(rows) < max_pairs:
            for query, selected in all_options:
                row, row_stats = make_row(
                    query,
                    selected,
                    cycle,
                    query_ids,
                    skill_ids,
                    skills,
                    group_aux,
                    family_hn_ratio,
                    by_family,
                    by_family_capability,
                    seed,
                    subintents,
                    routed_view_cache,
                )
                rows.append(row)
                counts[f"view_{row_stats['view']}"] += 1
                counts["family_hard_negative_slots"] += int(row_stats["hard_negative"])
                counts["family_hard_negative_exact_capability"] += int(
                    row_stats["hard_negative_exact_capability"]
                )
                if len(rows) >= max_pairs:
                    break
            cycle += 1

    gold_counts = [len(query["skill_ids"]) for query in usable_queries]
    multi = [count for count in gold_counts if count >= 2]
    stats = {
        "rows": len(rows),
        "train_queries": len(usable_queries),
        "query_unique_global_batch": query_unique,
        "planned_unique_queries_per_global_batch": global_batch
        if query_unique
        else None,
        "query_unique_sampler_rows": len(sampler_order)
        if sampler_order is not None
        else None,
        "positive_slots": {
            "standard": 1,
            "group_fixed_slots": MAX_GOLDS if group_aux else 0,
        },
        "gold_count_distribution": dict(Counter(gold_counts)),
        "average_positives_per_multi_gold_query": sum(multi) / len(multi),
        "group_score_normalization": "logsumexp(scores)-log(number_of_golds)"
        if group_aux
        else "disabled",
        "family_hard_negative_target_ratio": family_hn_ratio,
        "family_hard_negative_realized_ratio": counts["family_hard_negative_slots"]
        / max(1, len(rows)),
        "counts": dict(counts),
        "rows_with_aligned_subintent": sum(int(row["label"][-1]) for row in rows),
        "coverage_rewrite_parent_count": len(coverage_rewrites),
        "coverage_rewrite_training_rows": sum(
            len(coverage_rewrites)
            for cycle in range(cycles)
            if 1 <= cycle <= coverage_rewrite_cycles
        )
        if query_unique
        else 0,
        "coverage_rewrite_cycles": coverage_rewrite_cycles,
        "positive_view_policy": positive_views,
        "semantic_router": (
            "fixed lexical routing from per-gold query-supported subintent to Markdown facet"
            if positive_views == "semantic_routed"
            else "round-robin skill views"
        ),
        "semantic_skill_facet_cache_entries": len(semantic_facet_cache),
        "semantic_query_gold_route_cache_entries": len(routed_view_cache),
    }
    if progress_callback:
        progress_callback("arrow_materialization", len(rows), len(rows))
    return Dataset.from_list(rows), stats, sampler_order


def shared_dataset_cache_path(
    args: argparse.Namespace, global_batch: int, out_dir: Path
) -> Path:
    def source_identity(path: Path) -> dict:
        stat = path.stat()
        return {
            "path": str(path.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }

    specification = {
        "schema_version": 1,
        "queries": source_identity(args.queries),
        "qrels": source_identity(args.qrels),
        "skills": source_identity(args.skills),
        "subintent_cache": source_identity(args.subintent_cache),
        "query_unique": bool(args.query_unique),
        "group_aux": bool(args.group_lambda > 0 or args.weakest_gold_lambda > 0),
        "family_hn_ratio": args.family_hn_ratio,
        "seed": args.seed,
        "max_pairs": args.max_pairs,
        "global_batch": global_batch,
        "subintent_view": args.subintent_view,
        "positive_views": args.positive_views,
        "coverage_rewrites": source_identity(args.coverage_rewrites)
        if args.coverage_rewrites
        else None,
        "coverage_rewrite_cycles": args.coverage_rewrite_cycles,
    }
    key = hashlib.sha256(
        json.dumps(specification, sort_keys=True).encode()
    ).hexdigest()[:20]
    return Path(os.environ.get("COVERSKILL_WORK", "work")) / "stage1_dataset" / key


def load_shared_dataset(
    args: argparse.Namespace,
    out_dir: Path,
    queries: list[dict],
    skills: dict[str, dict],
    subintents: dict[str, dict[str, str]],
    global_batch: int,
    progress_callback,
) -> tuple[Dataset, dict, list[int] | None, Path]:
    cache_dir = shared_dataset_cache_path(args, global_batch, out_dir)
    manifest_path = cache_dir / "manifest.json"
    if RANK == 0 and not manifest_path.is_file():
        temporary = cache_dir.with_name(f".{cache_dir.name}.tmp.{os.getpid()}")
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True, exist_ok=True)
        coverage_rewrites = {}
        if args.coverage_rewrites:
            known = {row["id"]: set(row["skill_ids"]) for row in queries}
            for item in read_jsonl(args.coverage_rewrites):
                if item.get("accepted") is not True:
                    continue
                qid = str(item["parent_query_id"])
                if qid not in known or set(item["gold_skill_ids"]) != known[qid]:
                    raise ValueError(f"coverage rewrite has mismatched gold set: {qid}")
                if qid in coverage_rewrites:
                    raise ValueError(f"duplicate coverage rewrite parent: {qid}")
                coverage_rewrites[qid] = str(item["rewrite"])
        dataset, stats, sampler_order = build_dataset(
            queries,
            skills,
            args.query_unique,
            args.group_lambda > 0 or args.weakest_gold_lambda > 0,
            args.family_hn_ratio,
            args.seed,
            args.max_pairs,
            global_batch,
            subintents,
            args.positive_views,
            coverage_rewrites=coverage_rewrites,
            coverage_rewrite_cycles=args.coverage_rewrite_cycles,
            progress_callback=progress_callback,
        )
        progress_callback("dataset_cache_write", len(dataset), len(dataset))
        dataset.save_to_disk(str(temporary / "dataset"))
        np.save(
            temporary / "sampler_order.npy",
            np.asarray(
                sampler_order if sampler_order is not None else [], dtype=np.int64
            ),
            allow_pickle=False,
        )
        (temporary / "stats.json").write_text(
            json.dumps(stats, indent=2) + "\n", encoding="utf-8"
        )
        (temporary / "manifest.json").write_text(
            json.dumps({"complete": True, "rows": len(dataset)}, indent=2) + "\n",
            encoding="utf-8",
        )
        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        os.replace(temporary, cache_dir)
        del dataset
    elif RANK != 0:
        progress_callback("waiting_for_rank0_dataset", 0, 1)
        deadline = time.time() + 1800
        while not manifest_path.is_file():
            if time.time() >= deadline:
                raise TimeoutError(
                    f"timed out waiting for shared dataset cache: {manifest_path}"
                )
            time.sleep(1)

    progress_callback("dataset_cache_load", 0, 1)
    dataset = load_from_disk(str(cache_dir / "dataset"))
    stats = json.loads((cache_dir / "stats.json").read_text(encoding="utf-8"))
    saved_order = np.load(cache_dir / "sampler_order.npy", allow_pickle=False).tolist()
    sampler_order = saved_order if args.query_unique else None
    progress_callback("dataset_cache_load", 1, 1)
    return dataset, stats, sampler_order, cache_dir


class FixedOrderSampler(Sampler[int]):
    def __init__(self, indices: list[int]) -> None:
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


class MGCATrainer(SentenceTransformerTrainer):
    def __init__(
        self,
        *args,
        fixed_sampler_order: list[int] | None = None,
        scheduler_horizon: int | None = None,
        **kwargs,
    ) -> None:
        self.scheduler_horizon = scheduler_horizon
        self.fixed_sampler_order = fixed_sampler_order
        super().__init__(*args, **kwargs)

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        horizon = (
            self.scheduler_horizon
            if self.scheduler_horizon is not None
            else num_training_steps
        )
        if horizon < num_training_steps:
            raise ValueError(
                "scheduler horizon cannot be shorter than the fixed training run"
            )
        scheduler = super().create_scheduler(
            num_training_steps=horizon, optimizer=optimizer
        )
        if RANK == 0:
            contract = {
                "actual_training_steps": num_training_steps,
                "trainer_max_steps": self.args.max_steps,
                "scheduler_horizon": horizon,
                "warmup_steps": self.args.get_warmup_steps(horizon),
                "scheduler": "linear",
                "peak_lr": self.args.learning_rate,
                "final_selection": "fixed checkpoint-492",
            }
            Path(self.args.output_dir, "ACTUAL_LR_SCHEDULER.json").write_text(
                json.dumps(contract, indent=2) + "\n"
            )
        return scheduler

    def get_batch_sampler(
        self,
        dataset,
        batch_size: int,
        drop_last: bool,
        valid_label_columns=None,
        generator=None,
        seed: int = 0,
    ):
        if self.fixed_sampler_order is not None:
            # SentenceTransformerTrainer builds DataLoaders from a batch
            # sampler and does not consult Transformers' _get_train_sampler.
            # Accelerate shards consecutive local batches across ranks, so
            # this shared deterministic order is arranged in groups of
            # WORLD_SIZE local batches to form one query-unique global batch.
            return BatchSampler(
                FixedOrderSampler(self.fixed_sampler_order), batch_size, drop_last
            )
        return super().get_batch_sampler(
            dataset,
            batch_size,
            drop_last,
            valid_label_columns=valid_label_columns,
            generator=generator,
            seed=seed,
        )


class CachedMGCALoss(CachedMultipleNegativesRankingLoss):
    """Standard single-positive CachedMNRL plus optional corrected components."""

    def __init__(
        self,
        *args,
        gold_denominator_mask: bool,
        group_lambda: float,
        weakest_gold_lambda: float,
        weakest_gold_temperature: float,
        family_hn_ratio: float,
        subintent_weight: float,
        stats_path: Path,
        max_steps: int,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.gold_denominator_mask = gold_denominator_mask
        self.group_lambda = group_lambda
        self.weakest_gold_lambda = weakest_gold_lambda
        self.weakest_gold_temperature = weakest_gold_temperature
        self.family_hn_ratio = family_hn_ratio
        self.subintent_weight = subintent_weight
        self.stats_path = stats_path
        self._labels: torch.Tensor | None = None
        self._step = 0
        self.max_steps = int(max_steps)
        self.started = time.time()

    def forward(self, sentence_features, labels: torch.Tensor) -> torch.Tensor:
        if labels is None or labels.ndim != 2 or labels.size(1) != 9:
            raise ValueError(
                "Expected labels [query_id, selected_skill_id, gold_count, gold_1..3, hn_valid, hn_skill_id, subintent_valid]"
            )
        group_enabled = self.group_lambda > 0 or self.weakest_gold_lambda > 0
        expected = (
            3 + int(self.family_hn_ratio > 0) + (MAX_GOLDS if group_enabled else 0)
        )
        if len(sentence_features) != expected:
            raise ValueError(
                f"Expected {expected} sentence feature groups, got {len(sentence_features)}"
            )
        self._labels = labels.long()
        features = list(sentence_features)
        reps, self.random_states = [], []
        for feature in features:
            chunks, states = [], []
            for rep, state in self.embed_minibatch_iter(
                feature, with_grad=False, copy_random_state=True
            ):
                chunks.append(rep.detach().requires_grad_())
                states.append(state)
            reps.append(chunks)
            self.random_states.append(states)
        if torch.is_grad_enabled():
            loss = self.calculate_loss_and_cache_gradients(reps)
            loss.register_hook(
                partial(_backward_hook, sentence_features=features, loss_obj=self)
            )
            return loss
        return self.calculate_loss(reps)

    def _gather_labels(self, labels: torch.Tensor) -> torch.Tensor:
        if (
            not self.gather_across_devices
            or not dist.is_available()
            or not dist.is_initialized()
        ):
            return labels
        chunks = [torch.empty_like(labels) for _ in range(dist.get_world_size())]
        dist.all_gather(chunks, labels)
        return torch.cat(chunks, dim=0)

    def calculate_loss(
        self, reps: list[list[torch.Tensor]], with_backward: bool = False
    ) -> torch.Tensor:
        if self._labels is None:
            raise RuntimeError("Missing Multi-granularity labels")
        cursor = 0
        local_queries = torch.cat(reps[cursor])
        cursor += 1
        local_selected = torch.cat(reps[cursor])
        cursor += 1
        local_subintent = torch.cat(reps[cursor])
        cursor += 1
        local_hard_negative = None
        if self.family_hn_ratio > 0:
            local_hard_negative = torch.cat(reps[cursor])
            cursor += 1
        local_group = []
        if self.group_lambda > 0 or self.weakest_gold_lambda > 0:
            local_group = [torch.cat(reps[cursor + slot]) for slot in range(MAX_GOLDS)]
            cursor += MAX_GOLDS
        if cursor != len(reps):
            raise RuntimeError("Multi-granularity feature cursor mismatch")

        local_labels = self._labels.to(local_queries.device)
        local_batch = local_queries.size(0)
        queries, selected, subintents, labels = (
            local_queries,
            local_selected,
            local_subintent,
            local_labels,
        )
        hard_negative, group_docs = local_hard_negative, local_group
        offset = 0
        if self.gather_across_devices:
            queries = all_gather_with_grad(queries)
            selected = all_gather_with_grad(selected)
            subintents = all_gather_with_grad(subintents)
            if hard_negative is not None:
                hard_negative = all_gather_with_grad(hard_negative)
            if group_docs:
                group_docs = [all_gather_with_grad(slot) for slot in group_docs]
            labels = self._gather_labels(local_labels)
            if dist.is_available() and dist.is_initialized():
                offset = dist.get_rank() * local_batch

        world_batch = queries.size(0)
        if labels.size(0) != world_batch:
            raise RuntimeError("Global labels and embeddings are misaligned")
        local_indices = torch.arange(
            offset, offset + local_batch, device=queries.device
        )
        candidate_docs = selected
        candidate_valid = torch.ones(
            world_batch, dtype=torch.bool, device=queries.device
        )
        if hard_negative is not None:
            candidate_docs = torch.cat([candidate_docs, hard_negative], dim=0)
            candidate_valid = torch.cat([candidate_valid, labels[:, 6].bool()], dim=0)

        group_all = torch.cat(group_docs, dim=0) if group_docs else None
        group_valid = (
            torch.cat([labels[:, 2] > slot for slot in range(MAX_GOLDS)], dim=0)
            if group_docs
            else None
        )
        chunks = []
        standard_detached = []
        group_detached = []
        weakest_detached = []
        subintent_detached = []
        mask_trigger_count = 0
        local_active_subintents = int(labels[local_indices, 8].sum().item())

        for begin in range(0, local_batch, self.mini_batch_size):
            end = min(begin + self.mini_batch_size, local_batch)
            indices = local_indices[begin:end]
            rows = torch.arange(end - begin, device=queries.device)
            scores = self.similarity_fct(queries[indices], candidate_docs) * self.scale
            scores = scores.masked_fill(~candidate_valid.unsqueeze(0), -torch.inf)
            if self.gold_denominator_mask:
                selected_is_gold = (
                    labels[:, 1].unsqueeze(0).unsqueeze(0)
                    == labels[indices, 3:6].unsqueeze(2)
                ).any(dim=1)
                # A candidate selected for any other query is still a false
                # negative when its skill id belongs to this row's gold set.
                # Query-unique batches intentionally contain no duplicate
                # query rows, so conditioning this mask on equal query ids
                # would make the denominator correction a no-op.
                mask = selected_is_gold
                mask[rows, indices] = False
                mask_trigger_count += int(mask.sum().item())
                scores[:, :world_batch] = scores[:, :world_batch].masked_fill(
                    mask, -torch.inf
                )
                if hard_negative is not None:
                    hard_negative_is_gold = (
                        labels[:, 7].unsqueeze(0).unsqueeze(0)
                        == labels[indices, 3:6].unsqueeze(2)
                    ).any(dim=1)
                    scores[:, world_batch:] = scores[:, world_batch:].masked_fill(
                        hard_negative_is_gold, -torch.inf
                    )
            standard = torch.logsumexp(scores, dim=1) - scores[rows, indices]

            sub_scores = (
                self.similarity_fct(subintents[indices], candidate_docs) * self.scale
            )
            sub_scores = sub_scores.masked_fill(
                ~candidate_valid.unsqueeze(0), -torch.inf
            )
            if self.gold_denominator_mask:
                sub_scores[:, :world_batch] = sub_scores[:, :world_batch].masked_fill(
                    mask, -torch.inf
                )
                if hard_negative is not None:
                    sub_scores[:, world_batch:] = sub_scores[
                        :, world_batch:
                    ].masked_fill(hard_negative_is_gold, -torch.inf)
            subintent = torch.logsumexp(sub_scores, dim=1) - sub_scores[rows, indices]
            subintent = subintent * labels[indices, 8].to(subintent.dtype)
            subintent = subintent * (local_batch / max(1, local_active_subintents))

            group = torch.zeros_like(standard)
            weakest = torch.zeros_like(standard)
            if group_all is not None and group_valid is not None:
                group_scores = (
                    self.similarity_fct(queries[indices], group_all) * self.scale
                )
                group_scores = group_scores.masked_fill(
                    ~group_valid.unsqueeze(0), -torch.inf
                )
                same_query_group = labels[:, 0].unsqueeze(0) == labels[
                    indices, 0
                ].unsqueeze(1)
                for slot in range(MAX_GOLDS):
                    other_same_query = same_query_group.clone()
                    other_same_query[rows, indices] = False
                    start = slot * world_batch
                    group_scores[:, start : start + world_batch] = group_scores[
                        :, start : start + world_batch
                    ].masked_fill(other_same_query, -torch.inf)
                pos_mask = torch.zeros_like(group_scores, dtype=torch.bool)
                own_counts = labels[indices, 2]
                for slot in range(MAX_GOLDS):
                    active = own_counts > slot
                    pos_mask[rows[active], slot * world_batch + indices[active]] = True
                positive_scores = group_scores.masked_fill(~pos_mask, -torch.inf)
                normalized_group_score = torch.logsumexp(
                    positive_scores, dim=1
                ) - torch.log(own_counts.to(scores.dtype))
                group = torch.logsumexp(group_scores, dim=1) - normalized_group_score

                for local_row, global_row in enumerate(indices.tolist()):
                    count = int(labels[global_row, 2].item())
                    per_gold = []
                    for slot in range(count):
                        target = slot * world_batch + global_row
                        logits = group_scores[local_row].clone()
                        for sibling in range(count):
                            if sibling != slot:
                                logits[sibling * world_batch + global_row] = -torch.inf
                        per_gold.append(torch.logsumexp(logits, dim=0) - logits[target])
                    values = torch.stack(per_gold)
                    tau = self.weakest_gold_temperature
                    weakest[local_row] = tau * (
                        torch.logsumexp(values / tau, dim=0) - math.log(count)
                    )

            weight = (end - begin) / local_batch
            value = (
                standard
                + self.subintent_weight * subintent
                + self.group_lambda * group
                + self.weakest_gold_lambda * weakest
            ).mean() * weight
            if with_backward:
                # Fixed dense slots share cached embedding leaves across chunks;
                # retain the tiny embedding graph until every cached gradient is collected.
                value.backward(retain_graph=True)
                value = value.detach()
            chunks.append(value)
            standard_detached.append(standard.detach())
            group_detached.append(group.detach())
            weakest_detached.append(weakest.detach())
            subintent_detached.append(subintent.detach())

        total = sum(chunks)
        if with_backward and RANK == 0:
            self._step += 1
            unique_queries = int(torch.unique(labels[:, 0]).numel())
            multi = labels[:, 2] >= 2
            record = {
                "step": self._step,
                "standard_loss": float(torch.cat(standard_detached).mean().item()),
                "group_loss": float(torch.cat(group_detached).mean().item())
                if self.group_lambda > 0
                else 0.0,
                "weakest_gold_loss": (
                    float(torch.cat(weakest_detached).mean().item())
                    if self.weakest_gold_lambda > 0
                    else 0.0
                ),
                "subintent_loss_active_row_mean": float(
                    torch.cat(subintent_detached).sum().item() / max(1, local_batch)
                ),
                "subintent_weight": self.subintent_weight,
                "subintent_active_rows_global": int(labels[:, 8].sum().item()),
                "subintent_active_rows_local": local_active_subintents,
                "group_lambda": self.group_lambda,
                "weakest_gold_lambda": self.weakest_gold_lambda,
                "total_loss": float(total.item()),
                "positive_slots": {
                    "standard": 1,
                    "group_fixed": MAX_GOLDS
                    if (self.group_lambda > 0 or self.weakest_gold_lambda > 0)
                    else 0,
                },
                "unique_queries_global_batch": unique_queries,
                "global_batch_rows": world_batch,
                "same_query_gold_mask_trigger_count_local": mask_trigger_count,
                "average_positives_per_multi_gold_query": (
                    float(labels[multi, 2].float().mean().item())
                    if multi.any()
                    else 0.0
                ),
                "group_score_normalized": self.group_lambda > 0,
                "family_hard_negative_ratio_global": float(
                    labels[:, 6].float().mean().item()
                ),
                "max_steps": self.max_steps,
                "elapsed_seconds": time.time() - self.started,
                "estimated_remaining_seconds": (self.max_steps - self._step)
                * (time.time() - self.started)
                / max(1, self._step),
                "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            }
            self.stats_path.parent.mkdir(parents=True, exist_ok=True)
            with self.stats_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            status_path = self.stats_path.parent / "training_status.json"
            temporary = status_path.with_name(f".{status_path.name}.{os.getpid()}.tmp")
            temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary, status_path)
            logger.info("mgca_components=%s", json.dumps(record, sort_keys=True))
        return total

    def get_config_dict(self) -> dict:
        return {
            **super().get_config_dict(),
            "objective": "standard CachedMNRL + optional denominator mask + normalized complete-gold group auxiliary",
            "gold_denominator_mask": self.gold_denominator_mask,
            "group_lambda": self.group_lambda,
            "weakest_gold_lambda": self.weakest_gold_lambda,
            "weakest_gold_temperature": self.weakest_gold_temperature,
            "group_score": "logsumexp(scores over complete gold set) - log(number of golds)",
            "family_hard_negative_ratio": self.family_hn_ratio,
            "subintent_weight": self.subintent_weight,
            "fixed_slot_dense_gradcache": True,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MGCA from an unmodified pretrained backbone."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/mgca.json"))
    parser.add_argument("--base-model", default=RAW_BASE_MODEL)
    parser.add_argument("--output", type=Path, default=Path("work/stage1"))
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(os.environ.get("COVERSKILL_DATA", "data/skillret")),
    )
    parser.add_argument(
        "--work", type=Path, default=Path(os.environ.get("COVERSKILL_WORK", "work"))
    )
    parser.add_argument("--micro-batch", type=int, default=24)
    a = parser.parse_args()
    c = json.loads(a.config.read_text())
    w = a.work
    from .common import resolve_backbone

    a.base_model = str(resolve_backbone(a.base_model))
    return argparse.Namespace(
        method_name="MGCA",
        output_dir=str(a.output),
        query_unique=True,
        gold_denominator_mask=True,
        group_lambda=c["group_lambda"],
        weakest_gold_lambda=c["weakest_gold_lambda"],
        weakest_gold_temperature=c["weakest_gold_temperature"],
        positive_views="semantic_routed",
        family_hn_ratio=0.0,
        per_device_batch=c["per_device_batch"],
        mini_batch_size=a.micro_batch,
        lr=c["lr"],
        warmup_ratio=0.1,
        warmup_steps=c["warmup_steps"],
        scheduler_horizon=c["scheduler_horizon"],
        max_length=c["max_length"],
        max_steps=c["max_steps"],
        save_steps=c["max_steps"],
        save_total_limit=1,
        max_pairs=c["max_pairs"],
        seed=c["seed"],
        queries=a.data / "queries/train.jsonl",
        qrels=a.data / "qrels/train.jsonl",
        skills=a.data / "skills/train.jsonl",
        subintent_cache=w / "subintents.pt",
        subintent_view="filtered",
        coverage_rewrites=w / "rewrites.jsonl",
        coverage_rewrite_cycles=2,
        subintent_weight=c["subintent_weight"],
        checkpoint=a.base_model,
        recipe=a.config,
        view_manifest=w / "subintents.pt.manifest.json",
    )


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    preparation_started = time.time()

    def preparation_status(
        phase: str,
        detail: str = "",
        completed: int | None = None,
        total: int | None = None,
    ) -> None:
        if RANK != 0:
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        status_path = out_dir / "training_status.json"
        value = {
            "phase": phase,
            "detail": detail,
            "step": 0,
            "max_steps": args.max_steps,
            "elapsed_seconds": time.time() - preparation_started,
            "estimated_remaining_seconds": None,
            "preparation_completed": completed,
            "preparation_total": total,
            "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        temporary = status_path.with_name(f".{status_path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, status_path)
        logger.info("preparation_status=%s", json.dumps(value, sort_keys=True))

    from .common import stage1_preflight

    stage1_preflight(args)
    if (
        args.group_lambda < 0
        or args.weakest_gold_lambda < 0
        or not 0 <= args.family_hn_ratio <= 1
    ):
        raise SystemExit("Invalid group lambda or family hard-negative ratio")
    random.seed(args.seed + RANK)
    np.random.seed(args.seed + RANK)
    torch.manual_seed(args.seed + RANK)

    preparation_status("model_loading", "load Qwen3-Embedding-0.6B on four ranks")
    model = SentenceTransformer(
        str(args.checkpoint),
        trust_remote_code=True,
        model_kwargs={"attn_implementation": "sdpa"},
    )
    model.max_seq_length = args.max_length
    model.prompts = {"query": SKILL_QUERY_PROMPT, "document": ""}
    if hasattr(model[0], "auto_model"):
        model[0].auto_model.config.use_cache = False
        model[0].auto_model.gradient_checkpointing_enable()

    preparation_status("data_loading", "load train queries, qrels, and skills")
    queries, skills = load_train_from_paths(args.queries, args.qrels, args.skills)
    preparation_status("subintent_loading", "decode aligned training-only subintents")
    subintents, subintent_stats = load_aligned_subintents(
        args.subintent_cache, model.tokenizer, args.subintent_view
    )
    global_batch = WORLD_SIZE * args.per_device_batch
    preparation_status(
        "semantic_dataset_build",
        "precompute skill facets and query-gold routes, then materialize training rows",
    )
    dataset, stats, sampler_order, dataset_cache = load_shared_dataset(
        args,
        out_dir,
        queries,
        skills,
        subintents,
        global_batch,
        lambda phase, completed, total: preparation_status(
            phase,
            "shared rank-0 build with memory-mapped loading on all ranks",
            completed,
            total,
        ),
    )
    if sampler_order is None or len(sampler_order) // global_batch != args.max_steps:
        raise RuntimeError(
            "fixed one-pass sampler length does not equal declared max_steps"
        )
    preparation_status(
        "trainer_initialization", "construct loss, sampler, and four-rank trainer"
    )
    if RANK == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        config = {
            **{
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "init": "raw_base_only",
            "base_model": str(args.checkpoint),
            "queries": str(args.queries),
            "qrels": str(args.qrels),
            "skills": str(args.skills),
            "subintent_cache": str(args.subintent_cache),
            "train_only": True,
            "official_test_in_training": False,
            "world_size": WORLD_SIZE,
            "logical_cross_device_batch": global_batch,
            "fixed_slot_dense_gradcache": True,
            "packed_rank_divergent_slots": False,
            "checkpoint_state_policy": "model_only_all_checkpoints_no_optimizer_state",
            "shared_dataset_cache": str(dataset_cache),
            "hostname": socket.gethostname(),
            **stats,
            **subintent_stats,
        }
        (out_dir / "train_config.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )
        logger.info("Multi-granularity config=%s", json.dumps(config, sort_keys=True))

    train_args = SentenceTransformerTrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=1.0,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=1,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps
        if args.warmup_steps is not None
        else max(1, int(args.max_steps * args.warmup_ratio)),
        fp16=False,
        bf16=True,
        gradient_checkpointing=False,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_only_model=True,
        logging_steps=1,
        dataloader_num_workers=4,
        dataloader_drop_last=True,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        report_to="none",
        run_name=args.method_name,
        prompts={"anchor": SKILL_QUERY_PROMPT},
        lr_scheduler_type="linear",
        max_steps=args.max_steps,
        seed=args.seed,
        data_seed=args.seed,
        optim="adamw_torch",
        ddp_timeout=1800,
    )
    stats_path = out_dir / "training_component_stats.jsonl"
    loss = CachedMGCALoss(
        model,
        scale=20.0,
        mini_batch_size=args.mini_batch_size,
        gather_across_devices=True,
        gold_denominator_mask=args.gold_denominator_mask,
        group_lambda=args.group_lambda,
        weakest_gold_lambda=args.weakest_gold_lambda,
        weakest_gold_temperature=args.weakest_gold_temperature,
        family_hn_ratio=args.family_hn_ratio,
        subintent_weight=args.subintent_weight,
        stats_path=stats_path,
        max_steps=args.max_steps,
    )
    trainer = MGCATrainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        loss=loss,
        fixed_sampler_order=sampler_order,
        scheduler_horizon=args.scheduler_horizon,
    )
    if not hasattr(trainer, "_nested_gather"):
        trainer._nested_gather = lambda tensors, name=None: tensors
    if not hasattr(trainer.args, "save_safetensors"):
        trainer.args.save_safetensors = False
    preparation_status(
        "training_start", "dataset ready; entering the first GPU optimization step"
    )
    trainer.train()
    if RANK == 0:
        final_dir = out_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(final_dir))
        recipe = json.loads(args.recipe.read_text(encoding="utf-8"))

        def file_sha(path: Path) -> str:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()

        ancestry = {
            "recipe_id": recipe["recipe_id"],
            "training_stage_id": "MGCA",
            "training_stage": 1,
            "parent_path": str(args.checkpoint),
            "parent_model_sha256": file_sha(
                Path(args.checkpoint) / "model.safetensors"
            ),
            "serial_training_depth": 1,
            "uses_historical_targeted_checkpoint": False,
            "shared_trainable_query_document_encoder": True,
        }
        manifest = {
            "complete": True,
            "recipe_id": recipe["recipe_id"],
            "recipe_sha256": file_sha(args.recipe),
            "parent": ancestry,
            "queries_sha256": file_sha(args.queries),
            "qrels_sha256": file_sha(args.qrels),
            "skills_sha256": file_sha(args.skills),
            "subintent_cache_sha256": file_sha(args.subintent_cache),
            "view_manifest_sha256": file_sha(args.view_manifest),
            "training_domain": "full official-train",
            "official_test_qrels_used_for_training": False,
            "leaderboard_overfitting_allowed": False,
            "checkpoint_selection": "fixed final step; no test selection",
            "document_embeddings_frozen_or_cached": False,
            **stats,
            **subintent_stats,
        }
        for directory in [out_dir, *sorted(out_dir.glob("checkpoint-*")), final_dir]:
            if directory.is_dir():
                (directory / "TRAINING_ANCESTRY.json").write_text(
                    json.dumps(ancestry, indent=2) + "\n"
                )
                (directory / "method_config.json").write_text(
                    json.dumps(recipe, indent=2) + "\n"
                )
                (directory / "training_manifest.json").write_text(
                    json.dumps(manifest, indent=2) + "\n"
                )
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        logger.info("Saved final to %s", final_dir)


if __name__ == "__main__":
    main()
