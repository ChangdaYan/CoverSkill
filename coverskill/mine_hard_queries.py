from __future__ import annotations

import argparse
import hashlib
import os
import time
from collections import Counter
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .common import (
    FULL_QUERY_TOKENS,
    TRAIN_QRELS,
    TRAIN_QUERIES,
    artifact,
    atomic_json,
    atomic_torch_save,
    iter_jsonl,
    read_json,
    sha256_file,
    utc_now,
    write_gzip_jsonl,
)
from .data import load_qrels
from .modeling import Retriever, cached_features, similarity_scores


HARD_TARGETS = {1: 2048, 2: 4096, 3: 2048}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine a deterministic hard-8192 from one experiment's own Stage-1 retrieval geometry"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--document-cache", type=Path, required=True)
    parser.add_argument("--document-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-id", required=True)
    parser.add_argument("--micro-batch", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--seed", type=int, default=43)
    return parser.parse_args()


def stable_tie(selection_id: str, seed: int, query_id: str) -> str:
    return hashlib.sha256(f"{selection_id}:{seed}:{query_id}".encode()).hexdigest()


def risk_row(
    query_id: str,
    gold_skill_ids: list[str],
    top_indices: torch.Tensor,
    top_scores: torch.Tensor,
    skill_ids: list[str],
    selection_id: str,
    seed: int,
) -> dict:
    gold = set(gold_skill_ids)
    top = [
        {
            "skill_id": skill_ids[int(index)],
            "rank": rank,
            "score": float(score),
        }
        for rank, (index, score) in enumerate(
            zip(top_indices.tolist(), top_scores.tolist()), start=1
        )
    ]
    rank_by_skill = {item["skill_id"]: item["rank"] for item in top}
    score_by_skill = {item["skill_id"]: item["score"] for item in top}
    floor = min(item["score"] for item in top) - 1.0
    gold_ranks = [rank_by_skill.get(skill_id, 31) for skill_id in gold]
    gold_scores = [score_by_skill.get(skill_id, floor) for skill_id in gold]
    non_gold = [item for item in top if item["skill_id"] not in gold]
    if not non_gold:
        raise RuntimeError(
            f"Stage-1 Top-{len(top)} contains no non-gold for query {query_id}"
        )
    confuser = non_gold[0]
    weakest_gold = min(
        gold, key=lambda value: (score_by_skill.get(value, floor), value)
    )
    missing10 = sum(value > 10 for value in gold_ranks)
    missing30 = sum(value > 30 for value in gold_ranks)
    weak_rank = max(gold_ranks)
    gap = float(confuser["score"]) - min(gold_scores)
    return {
        "query_id": query_id,
        "gold_skill_ids": sorted(gold),
        "gold_count": len(gold),
        "missing_gold_top10": missing10,
        "missing_gold_top30": missing30,
        "weakest_gold_rank_capped31": weak_rank,
        "weakest_gold_skill_id": weakest_gold,
        "top_non_gold_skill_id": confuser["skill_id"],
        "top_non_gold_rank": confuser["rank"],
        "top_non_gold_score": confuser["score"],
        "confuser_minus_weakest_gold_score": gap,
        "selection_priority": [
            missing10,
            missing30,
            int(confuser["rank"] <= 10),
            weak_rank,
            gap,
        ],
        "stable_tie_sha256": stable_tie(selection_id, seed, query_id),
        "stage1_top30": top,
        "official_test_qrels_used": False,
    }


def select_hard(rows: list[dict]) -> list[dict]:
    selected: list[dict] = []
    for gold_count, target in HARD_TARGETS.items():
        group = [row for row in rows if row["gold_count"] == gold_count]
        group.sort(
            key=lambda row: (
                tuple(-float(value) for value in row["selection_priority"])
                + (row["stable_tie_sha256"],)
            )
        )
        if len(group) < target:
            raise RuntimeError(
                f"only {len(group)} {gold_count}-gold rows available for quota {target}"
            )
        selected.extend(group[:target])
    selected.sort(key=lambda row: row["query_id"])
    if len(selected) != 8192 or len({row["query_id"] for row in selected}) != 8192:
        raise RuntimeError("own-Stage-1 hard selection size/uniqueness failure")
    return selected


def main() -> None:
    args = parse_args()
    from .common import train_sources, validate_parent_matrix

    train_sources()
    validate_parent_matrix(args.model, args.document_cache)
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")
    if args.top_k != 30:
        raise RuntimeError("the comparable hard-mining protocol requires Top-30")
    if args.micro_batch < 1:
        raise RuntimeError("micro batch must be positive")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 4:
        raise RuntimeError("own-Stage-1 hard mining requires exactly four ranks")

    model_sha = sha256_file(args.model / "model.safetensors")
    document_manifest = __import__("json").loads(
        args.document_manifest.read_text(encoding="utf-8")
    )
    if document_manifest.get("complete") is not True:
        raise RuntimeError("Stage-1 document manifest is incomplete")
    if document_manifest.get("model_sha256") != model_sha:
        raise RuntimeError(
            "Stage-1 document cache was not encoded by the requested Stage-1 checkpoint"
        )
    if sha256_file(args.document_cache) != document_manifest.get("cache_sha256"):
        raise RuntimeError("Stage-1 document cache hash mismatch")

    qrels = load_qrels(TRAIN_QRELS)
    records = [
        {"query_id": str(row["id"]), "gold_skill_ids": sorted(qrels[str(row["id"])])}
        for row in iter_jsonl(TRAIN_QUERIES)
        if str(row["id"]) in qrels
    ]
    if len(records) != 63259 or len({row["query_id"] for row in records}) != len(
        records
    ):
        raise RuntimeError("official full-train query domain changed")
    if any(len(row["gold_skill_ids"]) not in HARD_TARGETS for row in records):
        raise RuntimeError(
            "hard-mining protocol only supports the official 1/2/3-gold strata"
        )

    query_tokens = torch.load(FULL_QUERY_TOKENS, map_location="cpu", weights_only=True)[
        "queries"
    ]
    document_payload = torch.load(
        args.document_cache, map_location="cpu", weights_only=True
    )["train"]
    skill_ids = [str(value) for value in document_payload["skill_ids"]]
    if len(skill_ids) != len(set(skill_ids)):
        raise RuntimeError("duplicate skill IDs in Stage-1 document cache")
    skill_id_set = set(skill_ids)
    if any(not set(row["gold_skill_ids"]) <= skill_id_set for row in records):
        raise RuntimeError(
            "training qrels contain skills outside the Stage-1 document cache"
        )
    documents = F.normalize(document_payload["embeddings"].float(), dim=-1).to(device)
    query_encoder = Retriever.from_pretrained(args.model).to(device).eval()

    local_records = records[rank::world]
    local_rows: list[dict] = []
    progress_path = args.output.with_suffix(args.output.suffix + ".progress.json")
    started = time.time()
    with torch.inference_mode():
        for begin in range(0, len(local_records), args.micro_batch):
            batch = local_records[begin : begin + args.micro_batch]
            sequences = [query_tokens[row["query_id"]] for row in batch]
            states = query_encoder(cached_features(sequences, device)).states
            scores = similarity_scores(states, documents)
            top_scores, top_indices = torch.topk(scores, k=args.top_k, dim=-1)
            top_scores = top_scores.cpu()
            top_indices = top_indices.cpu()
            for row, indices, scores in zip(batch, top_indices, top_scores):
                local_rows.append(
                    risk_row(
                        row["query_id"],
                        row["gold_skill_ids"],
                        indices,
                        scores,
                        skill_ids,
                        args.selection_id,
                        args.seed,
                    )
                )
            if rank == 0:
                completed = min(len(records), len(local_rows) * world)
                elapsed = max(1e-6, time.time() - started)
                atomic_json(
                    progress_path,
                    {
                        "status": "scoring_full_train",
                        "completed": completed,
                        "total": len(records),
                        "estimated_remaining_seconds": (len(records) - completed)
                        * elapsed
                        / max(1, completed),
                        "selection_id": args.selection_id,
                        "updated_utc": utc_now(),
                    },
                )

    part = args.output.with_name(f".{args.output.name}.rank{rank}.part.pt")
    atomic_torch_save(part, {"rows": local_rows})
    dist.barrier()
    if rank == 0:
        rows: list[dict] = []
        parts: list[Path] = []
        for other in range(world):
            path = args.output.with_name(f".{args.output.name}.rank{other}.part.pt")
            parts.append(path)
            rows.extend(
                torch.load(path, map_location="cpu", weights_only=False)["rows"]
            )
        if len(rows) != len(records) or len({row["query_id"] for row in rows}) != len(
            records
        ):
            raise RuntimeError(
                "distributed hard-mining rows are incomplete or duplicated"
            )
        hard = select_hard(rows)
        risk_output = args.output.with_suffix(
            args.output.suffix + ".fulltrain_risk.jsonl.gz"
        )
        write_gzip_jsonl(risk_output, sorted(rows, key=lambda row: row["query_id"]))
        write_gzip_jsonl(args.output, hard)
        selected_ids = {row["query_id"] for row in hard}
        manifest = {
            "complete": True,
            "selection_id": args.selection_id,
            "selection_model": str(args.model.resolve()),
            "selection_model_sha256": model_sha,
            "document_cache": artifact(
                args.document_cache, "same-Stage-1 train document embeddings"
            ),
            "document_manifest": artifact(
                args.document_manifest, "same-Stage-1 document-cache manifest"
            ),
            "full_train_queries_scored": len(rows),
            "top_k": args.top_k,
            "rows": len(hard),
            "strata": dict(Counter(str(row["gold_count"]) for row in hard)),
            "risk_distribution_full_train": {
                "missing_gold_top10_queries": sum(
                    row["missing_gold_top10"] > 0 for row in rows
                ),
                "missing_gold_top30_queries": sum(
                    row["missing_gold_top30"] > 0 for row in rows
                ),
            },
            "risk_distribution_selected": {
                "missing_gold_top10_queries": sum(
                    row["missing_gold_top10"] > 0 for row in hard
                ),
                "missing_gold_top30_queries": sum(
                    row["missing_gold_top30"] > 0 for row in hard
                ),
            },
            "selection_priority": [
                "missing_gold_top10",
                "missing_gold_top30",
                "top_non_gold_at10",
                "weakest_or_last_gold_rank",
                "confuser_minus_weakest_gold_score",
                "seed43_tie",
            ],
            "selected_query_id_sha256": hashlib.sha256(
                "\n".join(sorted(selected_ids)).encode()
            ).hexdigest(),
            "queries": artifact(TRAIN_QUERIES, "official training queries only"),
            "qrels": artifact(TRAIN_QRELS, "official training qrels only"),
            "official_test_qrels_used": False,
            "fulltrain_risk_rows": artifact(
                risk_output,
                "train-only full-query rankings for targeted data and confusers",
            ),
            "artifact": artifact(
                args.output, "hard-8192 mined from this experiment's own Stage-1 model"
            ),
            "updated_utc": utc_now(),
        }
        atomic_json(
            args.output.with_suffix(args.output.suffix + ".manifest.json"), manifest
        )
        for path in parts:
            path.unlink()
        progress_path.unlink(missing_ok=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
