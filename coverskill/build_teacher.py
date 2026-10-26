from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoTokenizer

from .common import (
    FULL_QUERY_TOKENS,
    TRAIN_DECOMPOSITION as Q32_GOLD_FULLTRAIN,
    ROOT,
    HARD_ROWS as S2_OWN_HARD,
    TRAIN_SKILL_CACHE as S2_TRAIN_SKILL_CACHE,
    artifact,
    atomic_json,
    atomic_torch_save,
    iter_jsonl,
    sha256_file,
    utc_now,
)
from .data import load_full_records, select_decomposition_view
from .modeling import Retriever, cached_features


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the train-only facet-aligned explicit-effect teacher cache"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "teacher.pt")
    parser.add_argument("--hard-rows", type=Path, default=S2_OWN_HARD)
    parser.add_argument(
        "--all-train",
        action="store_true",
        help="Build rows for all official-train queries",
    )
    parser.add_argument("--document-cache", type=Path, default=S2_TRAIN_SKILL_CACHE)
    parser.add_argument("--micro-batch", type=int, default=12)
    parser.add_argument("--candidate-count", type=int, default=256)
    parser.add_argument("--effect-depth", type=int, default=50)
    parser.add_argument("--teacher-model", type=Path, required=True)
    parser.add_argument("--decomposition", type=Path, default=Q32_GOLD_FULLTRAIN)
    parser.add_argument(
        "--decomposition-view", choices=("filtered", "raw"), default="filtered"
    )
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--engineering-only", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def encode(reasoner, sequences, micro_batch, device):
    chunks = []
    for start in range(0, len(sequences), micro_batch):
        chunks.append(
            reasoner.encode_anchor_features(
                cached_features(sequences[start : start + micro_batch], device)
            ).cpu()
        )
    return torch.cat(chunks) if chunks else torch.empty(0, reasoner.hidden_size)


def force_gold_topk(scores: torch.Tensor, gold: list[int], count: int):
    indices = torch.topk(scores, k=count).indices.tolist()
    present = set(indices)
    cursor = len(indices) - 1
    for target in gold:
        if target not in present:
            while indices[cursor] in gold:
                cursor -= 1
            present.remove(indices[cursor])
            indices[cursor] = target
            present.add(target)
            cursor -= 1
    index = torch.tensor(indices, dtype=torch.int64)
    return index, scores[index].to(torch.float16)


def effect_payload(
    full_scores: torch.Tensor,
    explicit_scores: torch.Tensor,
    branch_scores: torch.Tensor,
    gold: list[int],
    depth: int,
) -> dict:
    """Build a compact Top-k correction cache without opening test labels."""
    depth = min(depth, full_scores.numel())
    full_order = torch.topk(full_scores, k=depth).indices.tolist()
    explicit_order = torch.topk(explicit_scores, k=depth).indices.tolist()
    candidates = list(dict.fromkeys(full_order + explicit_order + list(gold)))
    indices = torch.tensor(candidates, dtype=torch.int64)
    teacher = explicit_scores[indices].to(torch.float16)
    anchor = full_scores[indices].to(torch.float16)
    teacher_rank = {value: rank + 1 for rank, value in enumerate(explicit_order)}
    anchor_rank = {value: rank + 1 for rank, value in enumerate(full_order)}
    weights = []
    gold_set = set(gold)

    def dcg(order: list[int]) -> float:
        return sum(
            1.0 / __import__("math").log2(rank + 2)
            for rank, value in enumerate(order[:10])
            if value in gold_set
        )

    full_hits = len(gold_set.intersection(full_order[:10]))
    explicit_hits = len(gold_set.intersection(explicit_order[:10]))
    effect_accept = explicit_hits > full_hits or (
        explicit_hits == full_hits and dcg(explicit_order) > dcg(full_order) + 1e-8
    )
    for value in candidates:
        rank = teacher_rank.get(value, depth + 1)
        weight = 1.0 if rank <= 10 else 0.5 if rank <= 20 else 0.15
        if value in gold_set and teacher_rank.get(value, depth + 1) <= 10:
            weight *= 2.0
        weights.append(weight)

    # Only retain relevance-changing inversions.  Pairing every candidate with
    # every other candidate would mostly duplicate the base contrastive loss.
    positives, negatives, pair_weights = [], [], []
    non_gold = [value for value in candidates if value not in gold_set]
    for positive in gold:
        for negative in non_gold:
            teacher_prefers = float(explicit_scores[positive]) > float(
                explicit_scores[negative]
            )
            anchor_prefers = float(full_scores[positive]) > float(full_scores[negative])
            boundary_relevant = (
                min(
                    teacher_rank.get(positive, depth + 1),
                    anchor_rank.get(positive, depth + 1),
                    teacher_rank.get(negative, depth + 1),
                    anchor_rank.get(negative, depth + 1),
                )
                <= 10
            )
            if teacher_prefers and not anchor_prefers and boundary_relevant:
                positives.append(positive)
                negatives.append(negative)
                promoted = (
                    anchor_rank.get(positive, depth + 1) > 10
                    and teacher_rank.get(positive, depth + 1) <= 10
                )
                pair_weights.append(2.0 if promoted else 1.0)
    # Cap pathological ties while keeping the largest teacher corrections.
    if len(positives) > 64:
        ordering = sorted(
            range(len(positives)),
            key=lambda i: float(
                explicit_scores[positives[i]] - explicit_scores[negatives[i]]
            ),
            reverse=True,
        )[:64]
        positives = [positives[i] for i in ordering]
        negatives = [negatives[i] for i in ordering]
        pair_weights = [pair_weights[i] for i in ordering]

    facet_count = branch_scores.shape[0]
    facet_width = max(depth + len(gold), 1)
    facet_indices = torch.full((3, facet_width), -1, dtype=torch.int64)
    facet_scores = torch.zeros((3, facet_width), dtype=torch.float16)
    facet_anchor = torch.zeros((3, facet_width), dtype=torch.float16)
    for facet in range(min(3, facet_count)):
        selected = list(
            dict.fromkeys(
                torch.topk(branch_scores[facet], k=depth).indices.tolist() + list(gold)
            )
        )
        selected = selected[:facet_width]
        count = len(selected)
        facet_indices[facet, :count] = torch.tensor(selected, dtype=torch.int64)
        facet_scores[facet, :count] = branch_scores[facet, selected].to(torch.float16)
        facet_anchor[facet, :count] = full_scores[selected].to(torch.float16)
    return {
        "effect_indices": indices,
        "effect_scores": teacher,
        "effect_anchor_scores": anchor,
        "effect_weights": torch.tensor(weights, dtype=torch.float16),
        "effect_pair_positive": torch.tensor(positives, dtype=torch.int64),
        "effect_pair_negative": torch.tensor(negatives, dtype=torch.int64),
        "effect_pair_weights": torch.tensor(pair_weights, dtype=torch.float16),
        "effect_accept": effect_accept,
        "teacher_top10_gold_delta": explicit_hits - full_hits,
        "teacher_top10_dcg_delta": dcg(explicit_order) - dcg(full_order),
        "facet_indices": facet_indices,
        "facet_scores": facet_scores,
        "facet_anchor_scores": facet_anchor,
    }


def main() -> None:
    args = parse_args()
    from .common import train_sources, validate_parent_matrix

    train_sources()
    validate_parent_matrix(args.teacher_model, args.document_cache)
    if args.max_queries and not args.engineering_only:
        raise RuntimeError("--max-queries requires --engineering-only")
    output = args.output
    if output.exists():
        raise RuntimeError(f"refusing to overwrite teacher cache: {output}")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 4:
        raise RuntimeError("teacher cache builder requires four ranks")
    parent_sha = sha256_file(args.teacher_model / "model.safetensors")
    decomposition = select_decomposition_view(
        torch.load(args.decomposition, map_location="cpu", weights_only=True),
        args.decomposition_view,
    )
    records = (
        load_full_records() if args.all_train else list(iter_jsonl(args.hard_rows))
    )
    domain = "official_train_all" if args.all_train else "s2_own_hard8192"
    if not args.all_train and len(records) != 8192:
        raise RuntimeError(f"expected S2 own-hard8192, found {len(records)} rows")
    if args.max_queries:
        records = records[: args.max_queries]
    total_records = len(records)
    records = records[rank::world]
    query_tokens = torch.load(FULL_QUERY_TOKENS, map_location="cpu", weights_only=True)[
        "queries"
    ]
    skill_payload = torch.load(
        args.document_cache, map_location="cpu", weights_only=True
    )["train"]
    skill_ids = [str(value) for value in skill_payload["skill_ids"]]
    skill_index = {value: index for index, value in enumerate(skill_ids)}
    documents = F.normalize(skill_payload["embeddings"].float(), dim=-1)
    recipe = {"representation": "anchor", "latent_steps": 0, "slot_count": 0}
    reasoner = Retriever.from_pretrained(args.teacher_model, recipe).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.teacher_model, local_files_only=True, trust_remote_code=True
    )
    rows = {}
    started = time.time()
    progress_path = output.with_suffix(output.suffix + ".progress.json")
    for start in range(0, len(records), args.micro_batch):
        batch = records[start : start + args.micro_batch]
        full = encode(
            reasoner,
            [query_tokens[str(row["query_id"])] for row in batch],
            args.micro_batch,
            device,
        )
        intent_sequences, owners = [], []
        for owner, row in enumerate(batch):
            for sequence in decomposition.get("subintent_tokens", {}).get(
                str(row["query_id"]), []
            )[:3]:
                intent_sequences.append(sequence)
                owners.append(owner)
        intent = encode(reasoner, intent_sequences, args.micro_batch, device)
        for owner, row in enumerate(batch):
            query_id = str(row["query_id"])
            branches = intent[
                [index for index, value in enumerate(owners) if value == owner]
            ]
            full_scores = full[owner] @ documents.T
            explicit_scores = (
                full_scores
                if not len(branches)
                else 0.8 * full_scores + 0.2 * (branches @ documents.T).max(0).values
            )
            gold = [skill_index[str(value)] for value in row["gold_skill_ids"]]
            final_indices, final_scores = force_gold_topk(
                explicit_scores, gold, args.candidate_count
            )
            padded_intents = torch.zeros(3, full.shape[-1], dtype=torch.float16)
            intent_mask = torch.zeros(3, dtype=torch.bool)
            if len(branches):
                retained = min(3, len(branches))
                padded_intents[:retained] = branches[:retained].to(torch.float16)
                intent_mask[:retained] = True
            payload = {
                "final_indices": final_indices,
                "final_scores": final_scores,
                "intent_embeddings": padded_intents,
                "intent_mask": intent_mask,
                "anchor_embedding": full[owner].to(torch.float16),
            }
            branch_scores = (
                branches @ documents.T if len(branches) else full_scores[None]
            )
            payload.update(
                effect_payload(
                    full_scores, explicit_scores, branch_scores, gold, args.effect_depth
                )
            )
            rows[query_id] = payload
        if rank == 0:
            completed = min(total_records, (start + len(batch)) * world)
            elapsed = max(1e-6, time.time() - started)
            atomic_json(
                progress_path,
                {
                    "status": "encoding",
                    "kind": "s2_selective_effect",
                    "domain": domain,
                    "completed": completed,
                    "total": total_records,
                    "estimated_remaining_seconds": (total_records - completed)
                    * elapsed
                    / max(1, completed),
                    "updated_utc": utc_now(),
                },
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    part = output.with_name(f".{output.name}.rank{rank}.part.pt")
    atomic_torch_save(part, rows)
    dist.barrier()
    if rank == 0:
        merged = {}
        parts = []
        for other in range(world):
            path = output.with_name(f".{output.name}.rank{other}.part.pt")
            parts.append(path)
            merged.update(torch.load(path, map_location="cpu", weights_only=True))
        expected_count = args.max_queries or (
            len(load_full_records())
            if args.all_train
            else len(list(iter_jsonl(args.hard_rows)))
        )
        if len(merged) != expected_count:
            raise RuntimeError(f"teacher cache rows {len(merged)} != {expected_count}")
        final = {
            "schema_version": 1,
            "complete": True,
            "kind": "s2_selective_effect",
            "domain": domain,
            "rows": merged,
            "query_count": len(merged),
            "candidate_count": args.candidate_count,
            "effect_depth": args.effect_depth,
            "w0_sha256": parent_sha,
            "document_cache_sha256": sha256_file(args.document_cache),
            "teacher_model": str(args.teacher_model.resolve()),
            "teacher_model_sha256": sha256_file(
                args.teacher_model / "model.safetensors"
            ),
            "decomposition": str(args.decomposition.resolve()),
            "decomposition_sha256": sha256_file(args.decomposition),
            "decomposition_view": args.decomposition_view,
            "accepted_queries": sum(
                bool(row.get("effect_accept")) for row in merged.values()
            ),
            "official_test_used": False,
            "engineering_only": args.engineering_only,
        }
        atomic_torch_save(output, final)
        atomic_json(
            output.with_suffix(output.suffix + ".manifest.json"),
            {
                "complete": True,
                "kind": "s2_selective_effect",
                "domain": domain,
                "query_count": len(merged),
                "w0_sha256": parent_sha,
                "document_cache_sha256": sha256_file(args.document_cache),
                "teacher_model": str(args.teacher_model.resolve()),
                "teacher_model_sha256": sha256_file(
                    args.teacher_model / "model.safetensors"
                ),
                "decomposition": str(args.decomposition.resolve()),
                "decomposition_sha256": sha256_file(args.decomposition),
                "decomposition_view": args.decomposition_view,
                "accepted_queries": sum(
                    bool(row.get("effect_accept")) for row in merged.values()
                ),
                "official_test_used": False,
                "artifact": artifact(output, "frozen train-only local-score cache"),
                "updated_utc": utc_now(),
            },
        )
        for path in parts:
            path.unlink()
        progress_path.unlink(missing_ok=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
