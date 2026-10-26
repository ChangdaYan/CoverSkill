from __future__ import annotations

import argparse
import math
import os
import random
import shutil
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from .common import (
    ROOT,
    FULL_QUERY_TOKENS,
    HARD_ROWS,
    RUNS,
    TRAIN_DECOMPOSITION,
    TRAIN_SKILL_CACHE,
    TRAIN_SKILL_MANIFEST,
    append_jsonl,
    artifact,
    atomic_json,
    read_json,
    sha256_file,
    stable_json_sha,
    utc_now,
)
from .data import (
    HardSampler,
    MixedSampler,
    aligned_intents,
    load_full_records,
    load_hard_records,
    select_decomposition_view,
    skill_gold_indices,
)
from .modeling import Retriever, cached_features
from .objectives import compute_objective
from .training_budget import resolve_budget


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/fatr.json"))
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--query-micro-batch-override", type=int)
    parser.add_argument("--gradient-checkpointing-override", choices=("true", "false"))
    parser.add_argument("--allreduce-bucket-mb-override", type=float)
    parser.add_argument(
        "--length-sorted-micro-batches-override", choices=("true", "false")
    )
    return parser.parse_args()


def setup_distributed() -> tuple[int, int, int, torch.device]:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world = dist.get_world_size()
    return rank, world, local_rank, device


def micro_batch_order(sequences, length_sorted):
    order = list(range(len(sequences)))
    if length_sorted:
        order.sort(key=lambda index: len(sequences[index]))
    return order


def restore_batch_order(values: torch.Tensor, order: list[int]) -> torch.Tensor:
    inverse = [0] * len(order)
    for sorted_index, original_index in enumerate(order):
        inverse[original_index] = sorted_index
    return values.index_select(
        0, torch.tensor(inverse, device=values.device, dtype=torch.long)
    )


def encode_queries_no_grad(
    query_encoder, sequences, micro_batch, device, length_sorted=False
):
    chunks, entropy = [], []
    order = micro_batch_order(sequences, length_sorted)
    with torch.no_grad():
        for start in range(0, len(order), micro_batch):
            indices = order[start : start + micro_batch]
            output = query_encoder(
                cached_features([sequences[index] for index in indices], device)
            )
            chunks.append(output.states)
            if output.token_entropy.numel():
                entropy.append(output.token_entropy)
    states = (
        torch.cat(chunks)
        if chunks
        else torch.empty(0, 1, query_encoder.hidden_size, device=device)
    )
    entropies = torch.cat(entropy) if entropy else states.new_empty(states.shape[0], 0)
    return restore_batch_order(states, order), restore_batch_order(entropies, order)


def encode_anchor_no_grad(
    query_encoder, sequences, micro_batch, device, length_sorted=False
):
    chunks = []
    order = micro_batch_order(sequences, length_sorted)
    with torch.no_grad():
        for start in range(0, len(order), micro_batch):
            indices = order[start : start + micro_batch]
            chunks.append(
                query_encoder.encode_anchor_features(
                    cached_features([sequences[index] for index in indices], device)
                )
            )
    values = (
        torch.cat(chunks)
        if chunks
        else torch.empty(0, query_encoder.hidden_size, device=device)
    )
    return restore_batch_order(values, order)


def backward_queries(
    query_encoder, sequences, gradients, micro_batch, device, length_sorted=False
):
    order = micro_batch_order(sequences, length_sorted)
    for start in range(0, len(order), micro_batch):
        indices = order[start : start + micro_batch]
        values = [sequences[index] for index in indices]
        output = query_encoder(cached_features(values, device)).states
        gradient_indices = torch.tensor(indices, device=gradients.device)
        torch.autograd.backward(output, gradients.index_select(0, gradient_indices))


def backward_anchor(
    query_encoder, sequences, gradients, micro_batch, device, length_sorted=False
):
    order = micro_batch_order(sequences, length_sorted)
    for start in range(0, len(order), micro_batch):
        indices = order[start : start + micro_batch]
        values = [sequences[index] for index in indices]
        output = query_encoder.encode_anchor_features(cached_features(values, device))
        gradient_indices = torch.tensor(indices, device=gradients.device)
        torch.autograd.backward(output, gradients.index_select(0, gradient_indices))


def _allreduce_gradient_bucket(gradients: list[torch.Tensor], world: int) -> None:
    if not gradients:
        return
    flat = torch.cat([gradient.reshape(-1) for gradient in gradients])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat.div_(world)
    offset = 0
    for gradient in gradients:
        count = gradient.numel()
        gradient.copy_(flat[offset : offset + count].view_as(gradient))
        offset += count


def allreduce_gradients(module: nn.Module, world: int, bucket_cap_mb: float) -> None:
    gradients = [
        parameter.grad
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    if bucket_cap_mb <= 0:
        for gradient in gradients:
            dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
            gradient.div_(world)
        return

    cap_bytes = max(1, round(bucket_cap_mb * 2**20))
    by_type: dict[tuple[torch.device, torch.dtype], list[torch.Tensor]] = {}
    for gradient in gradients:
        by_type.setdefault((gradient.device, gradient.dtype), []).append(gradient)
    for typed_gradients in by_type.values():
        bucket: list[torch.Tensor] = []
        bucket_bytes = 0
        for gradient in typed_gradients:
            gradient_bytes = gradient.numel() * gradient.element_size()
            if bucket and bucket_bytes + gradient_bytes > cap_bytes:
                _allreduce_gradient_bucket(bucket, world)
                bucket = []
                bucket_bytes = 0
            bucket.append(gradient)
            bucket_bytes += gradient_bytes
        _allreduce_gradient_bucket(bucket, world)


def validate_recipe(recipe):
    from .common import stage2_preflight

    stage2_preflight(recipe)
    return {"recipe_id": "MGCA"}


def save_checkpoint(query_encoder, tokenizer, target, recipe, ancestry, manifest, step):
    free_gib = shutil.disk_usage(target.parent).free / 2**30
    if free_gib < 6:
        raise RuntimeError(
            f"refusing checkpoint save: output disk has only {free_gib:.2f} GiB free"
        )
    if target.exists():
        raise RuntimeError(f"refusing to overwrite inference checkpoint {target}")
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    query_encoder.save_inference(temporary, tokenizer)
    atomic_json(temporary / "method_config.json", {**recipe, "step": step})
    atomic_json(
        temporary / "TRAINING_ANCESTRY.json",
        {**ancestry, "step": step, "output_checkpoint": str(target)},
    )
    atomic_json(
        temporary / "training_manifest.json",
        {**manifest, "complete": True, "step": step},
    )
    os.replace(temporary, target)


def main() -> None:
    args = parse_args()
    recipe = read_json(args.config)
    recipe.update(
        parent_model=str(args.parent.resolve()),
        parent_recipe_id="MGCA",
        train_skill_cache=str(ROOT / "train_documents.pt"),
        train_skill_manifest=str(ROOT / "train_documents.manifest.json"),
        hard_rows=str(ROOT / "hard_queries.jsonl.gz"),
        hard_rows_manifest=str(ROOT / "hard_queries.jsonl.gz.manifest.json"),
        train_decomposition=str(ROOT / "subintents.pt"),
        teacher_cache=str(ROOT / "teacher.pt"),
    )
    recipe.setdefault("gradient_checkpointing", True)
    recipe.setdefault("gradient_allreduce_bucket_mb", 0.0)
    recipe.setdefault("length_sorted_micro_batches", False)
    smoke = args.smoke_steps > 0
    overrides = {
        "query_micro_batch": args.query_micro_batch_override,
        "gradient_checkpointing": None
        if args.gradient_checkpointing_override is None
        else args.gradient_checkpointing_override == "true",
        "gradient_allreduce_bucket_mb": args.allreduce_bucket_mb_override,
        "length_sorted_micro_batches": None
        if args.length_sorted_micro_batches_override is None
        else args.length_sorted_micro_batches_override == "true",
    }
    smoke_only_overrides = {
        key: value for key, value in overrides.items() if key != "query_micro_batch"
    }
    if any(value is not None for value in smoke_only_overrides.values()) and not smoke:
        raise RuntimeError(
            "checkpointing, all-reduce, and length-sorting overrides are permitted only in isolated smoke runs"
        )
    recipe.update({key: value for key, value in overrides.items() if value is not None})
    parent_metadata = validate_recipe(recipe)
    rank, world, _, device = setup_distributed()
    if world != 4:
        raise RuntimeError("FATR training requires exactly four ranks")
    if 192 % world:
        raise RuntimeError("logical batch must divide world size")
    if smoke != (args.output_dir is not None):
        raise RuntimeError(
            "smoke mode requires both --smoke-steps and a dedicated --output-dir"
        )
    run = args.output_dir if smoke else RUNS / recipe["recipe_id"] / "seed43"
    if run.exists() and any(run.iterdir()):
        manifest_path = run / "manifest.json"
        if (
            not smoke
            and manifest_path.is_file()
            and read_json(manifest_path).get("complete") is True
        ):
            dist.destroy_process_group()
            return
        raise RuntimeError(
            f"non-empty incomplete run directory cannot resume because optimizer states are forbidden: {run}"
        )
    run.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(43)
    torch.cuda.manual_seed_all(43)
    random.seed(43)

    train_skill_cache = Path(recipe.get("train_skill_cache", TRAIN_SKILL_CACHE))
    train_skill_manifest = Path(
        recipe.get("train_skill_manifest", TRAIN_SKILL_MANIFEST)
    )
    train_decomposition = Path(recipe.get("train_decomposition", TRAIN_DECOMPOSITION))
    skill_manifest = read_json(train_skill_manifest)
    if sha256_file(train_skill_cache) != skill_manifest["cache_sha256"]:
        raise RuntimeError("frozen train skill cache hash mismatch")
    skill_payload = torch.load(
        train_skill_cache, map_location="cpu", weights_only=True
    )["train"]
    skill_ids = [str(value) for value in skill_payload["skill_ids"]]
    base_documents = F.normalize(skill_payload["embeddings"].float().to(device), dim=-1)
    skill_index = {value: index for index, value in enumerate(skill_ids)}
    full_tokens = torch.load(FULL_QUERY_TOKENS, map_location="cpu", weights_only=True)[
        "queries"
    ]
    decomposition = select_decomposition_view(
        torch.load(train_decomposition, map_location="cpu", weights_only=True),
        recipe.get("decomposition_view", "filtered"),
    )
    custom_hard_rows = recipe.get("hard_rows")
    hard_rows_path = Path(custom_hard_rows or HARD_ROWS)
    hard_manifest_path = None
    if custom_hard_rows:
        hard_manifest_path = Path(
            recipe.get("hard_rows_manifest", str(hard_rows_path) + ".manifest.json")
        )
        if not hard_rows_path.is_file() or not hard_manifest_path.is_file():
            raise RuntimeError(
                f"recipe-specific hard-8192 artifact is missing: {hard_rows_path}"
            )
        hard_manifest = read_json(hard_manifest_path)
        if (
            hard_manifest.get("complete") is not True
            or int(hard_manifest.get("rows", -1)) != 8192
        ):
            raise RuntimeError(
                f"recipe-specific hard-8192 manifest is incomplete: {hard_manifest_path}"
            )
        if sha256_file(hard_rows_path) != hard_manifest.get("artifact", {}).get(
            "sha256"
        ):
            raise RuntimeError("recipe-specific hard-8192 hash mismatch")
        expected_parent_sha = sha256_file(
            Path(recipe["parent_model"]) / "model.safetensors"
        )
        if hard_manifest.get("selection_model_sha256") != expected_parent_sha:
            raise RuntimeError(
                "hard-8192 was not selected by the declared Stage-1 parent"
            )
    hard_records = load_hard_records(hard_rows_path)
    data_mode = recipe.get("data_mode", "hard")
    broad_records = []
    if data_mode == "mixed":
        broad_records = load_full_records(require_decomposition=decomposition)
        sampler = MixedSampler(
            hard_records,
            broad_records,
            hard_fraction=float(recipe.get("hard_fraction", 0.5)),
            multi_gold_fraction=float(recipe.get("multi_gold_fraction", 0.0)),
            logical_batch=int(recipe["logical_batch"]),
        )
        records = hard_records + broad_records
    elif data_mode == "hard":
        records = hard_records
        sampler = HardSampler(hard_records, logical_batch=int(recipe["logical_batch"]))
    else:
        raise RuntimeError(f"unsupported data_mode: {data_mode}")
    teacher_payload = None
    teacher_path = (
        Path(recipe["teacher_cache"]) if recipe.get("teacher_cache") else None
    )
    if teacher_path:
        teacher_payload = torch.load(
            teacher_path, map_location="cpu", weights_only=True
        )
        if teacher_payload.get("complete") is not True:
            raise RuntimeError("teacher cache is incomplete")
        expected_document_sha = recipe.get("teacher_document_model_sha256")
        if expected_document_sha in (None, "parent"):
            expected_document_sha = sha256_file(
                Path(recipe["parent_model"]) / "model.safetensors"
            )
        if teacher_payload.get("parent_model_sha256") != expected_document_sha:
            raise RuntimeError(
                "teacher cache document geometry does not match this recipe"
            )
        if teacher_payload.get("document_cache_sha256") not in (
            None,
            skill_manifest["cache_sha256"],
        ):
            raise RuntimeError("teacher cache frozen document matrix hash mismatch")
        if (
            recipe.get("frozen_document_layout") == "facets"
            and teacher_payload.get("document_cache_sha256")
            != skill_manifest["cache_sha256"]
        ):
            raise RuntimeError(
                "facet-aligned training requires explicit teacher/matrix binding"
            )
        if not (
            {str(row["query_id"]) for row in records} & set(teacher_payload["rows"])
        ):
            raise RuntimeError("teacher cache has no overlap with the training domain")

    query_encoder = Retriever.from_pretrained(Path(recipe["parent_model"])).to(device)
    if recipe["gradient_checkpointing"]:
        query_encoder.enable_gradient_checkpointing()
    for module in query_encoder.modules():
        if isinstance(module, torch.nn.Dropout) and module.p:
            raise RuntimeError(
                "non-zero dropout violates deterministic GradCache recomputation"
            )
    tokenizer = AutoTokenizer.from_pretrained(
        recipe["parent_model"], local_files_only=True, trust_remote_code=True
    )
    optimizer = torch.optim.AdamW(
        query_encoder.parameters(), lr=recipe["lr"], weight_decay=recipe["weight_decay"]
    )
    schedule_steps, steps = resolve_budget(recipe, args.smoke_steps if smoke else 0)
    warmup = max(1, round(schedule_steps * float(recipe["warmup_ratio"])))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda value: (
            (value + 1) / warmup
            if value < warmup
            else max(0.0, (schedule_steps - value) / max(1, schedule_steps - warmup))
        ),
    )
    ancestry = {
        "training_stage": 2,
        "serial_training_depth": 2,
        "parent_recipe_id": recipe.get(
            "parent_recipe_id", parent_metadata["recipe_id"]
        ),
        "parent_model": str(Path(recipe["parent_model"]).resolve()),
        "parent_model_sha256": sha256_file(
            Path(recipe["parent_model"]) / "model.safetensors"
        ),
        "recipe_id": recipe["recipe_id"],
        "recipe_sha256": stable_json_sha(recipe),
        "initialized_from_stage1": True,
        "direct_stage1_parent": True,
        "not_initialized_from_another_stage2": True,
    }
    manifest = {
        "complete": False,
        "recipe": recipe,
        "ancestry": ancestry,
        "hard_query_count": len(hard_records),
        "broad_query_count": len(broad_records),
        "data_mode": data_mode,
        "skill_count": len(skill_ids),
        "document_encoder_trainable": False,
        "document_geometry_adapter_trainable": bool(
            recipe.get("document_adapter_rank", 0)
        ),
        "query_encoder_trainable": True,
        "optimizer_state_policy": "never_saved",
        "resume_supported": False,
        "official_test_used_for_training": False,
        "hard_rows": artifact(
            hard_rows_path,
            "train-only hard-8192 selected by this recipe's Stage-1 parent",
        ),
        "hard_rows_manifest": (
            artifact(hard_manifest_path, "hard-8192 selection provenance")
            if hard_manifest_path
            else None
        ),
        "train_skill_cache": artifact(
            train_skill_cache, "frozen parent-model train skill vectors"
        ),
        "train_decomposition": artifact(
            train_decomposition, "frozen train-only subintent views"
        ),
        "decomposition_view": recipe.get("decomposition_view", "filtered"),
        "teacher_cache": artifact(teacher_path, "frozen train-only teacher")
        if teacher_path
        else None,
        "engineering_smoke": smoke,
        "training_stop_steps": steps,
        "lr_schedule_steps": schedule_steps,
        "auxiliary_schedule_steps": schedule_steps,
        "warmup_steps": warmup,
        "protocol": "final endpoint; no optimizer state",
        "updated_utc": utc_now(),
    }
    if rank == 0:
        atomic_json(run / "TRAINING_ANCESTRY.json", ancestry)
        atomic_json(run / "manifest.json", manifest)

    local_batch = 192 // world
    seen: set[str] = set()
    for step in range(1, steps + 1):
        global_rows = sampler.batch(step)
        seen.update(str(row["query_id"]) for row in global_rows)
        local = global_rows[rank * local_batch : (rank + 1) * local_batch]
        sequences = [full_tokens[str(row["query_id"])] for row in local]
        gold = skill_gold_indices(local, skill_ids)
        if recipe.get("encode_training_subintents", True):
            intent_sequences, intent_owners, aligned, per_query_aligned = (
                aligned_intents(local, decomposition, skill_index)
            )
        else:
            intent_sequences, intent_owners, aligned = [], [], []
            per_query_aligned = [[] for _ in local]
        teacher_rows = (
            [teacher_payload["rows"].get(str(row["query_id"])) for row in local]
            if teacher_payload
            else None
        )

        optimizer.zero_grad(set_to_none=True)
        query_encoder.train()
        documents = query_encoder.adapt_documents(base_documents)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.time()
        micro = int(recipe["query_micro_batch"])
        length_sorted = recipe["length_sorted_micro_batches"]
        states, entropy = encode_queries_no_grad(
            query_encoder, sequences, micro, device, length_sorted
        )
        states = states.detach().requires_grad_(True)
        intents = (
            encode_anchor_no_grad(
                query_encoder, intent_sequences, micro, device, length_sorted
            )
            .detach()
            .requires_grad_(True)
        )
        total, metrics = compute_objective(
            states,
            intents,
            intent_owners,
            aligned,
            gold,
            documents,
            recipe,
            teacher_rows,
            per_query_aligned=per_query_aligned,
            training_progress=step / schedule_steps,
            base_documents=base_documents,
        )
        total.backward()
        if states.grad is None:
            raise RuntimeError("reasoning states received no gradient")
        backward_queries(
            query_encoder, sequences, states.grad.detach(), micro, device, length_sorted
        )
        if intent_sequences and intents.grad is not None:
            backward_anchor(
                query_encoder,
                intent_sequences,
                intents.grad.detach(),
                micro,
                device,
                length_sorted,
            )
        allreduce_gradients(
            query_encoder, world, float(recipe["gradient_allreduce_bucket_mb"])
        )
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(query_encoder.parameters(), 1.0)
        )
        if not math.isfinite(grad_norm):
            raise RuntimeError("non-finite gradient")
        optimizer.step()
        scheduler.step()
        torch.cuda.synchronize()
        elapsed = states.new_tensor(time.time() - started)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        for key, value in list(metrics.items()):
            tensor = states.new_tensor(float(value))
            dist.all_reduce(tensor)
            metrics[key] = float(tensor / world)
        entropy_mean = (
            entropy.float().mean() if entropy.numel() else states.new_tensor(0.0)
        )
        dist.all_reduce(entropy_mean)
        entropy_mean /= world
        if states.shape[1] > 1:
            off_diagonal = torch.matmul(states[:, 1:], states[:, 1:].transpose(1, 2))
            diversity = (
                1.0
                - (
                    (off_diagonal.sum(dim=(1, 2)) - (states.shape[1] - 1))
                    / max(1, (states.shape[1] - 1) * (states.shape[1] - 2))
                ).mean()
                if states.shape[1] > 2
                else 1.0 - (states[:, 0] * states[:, 1]).sum(dim=-1).mean()
            )
        else:
            diversity = states.new_tensor(0.0)
        dist.all_reduce(diversity)
        diversity /= world
        if rank == 0:
            status = {
                **metrics,
                "step": step,
                "max_steps": steps,
                "lr_schedule_steps": schedule_steps,
                "auxiliary_schedule_steps": schedule_steps,
                "grad_norm": grad_norm,
                "query_token_entropy": float(entropy_mean),
                "state_cosine_diversity": float(diversity.detach()),
                "queries_per_second": 192 / float(elapsed),
                "step_seconds": float(elapsed),
                "estimated_remaining_seconds": (steps - step) * float(elapsed),
                "unique_queries_seen": len(seen),
                "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                "query_micro_batch": micro,
                "gradient_checkpointing": recipe["gradient_checkpointing"],
                "gradient_allreduce_bucket_mb": recipe["gradient_allreduce_bucket_mb"],
                "length_sorted_micro_batches": length_sorted,
                "lr": scheduler.get_last_lr()[0],
                "updated_utc": utc_now(),
            }
            append_jsonl(run / "train_metrics.jsonl", status)
            atomic_json(run / "training_status.json", status)
            print(status, flush=True)
        if not smoke and step in recipe["checkpoint_steps"]:
            dist.barrier()
            if rank == 0:
                save_checkpoint(
                    query_encoder,
                    tokenizer,
                    run / f"checkpoint-{step}",
                    recipe,
                    ancestry,
                    manifest,
                    step,
                )
            dist.barrier()

    if rank == 0:
        atomic_json(
            run / ("SMOKE_COMPLETE.json" if smoke else "manifest.json"),
            {
                **manifest,
                "complete": True,
                "steps": steps,
                "scientific_training_run": not smoke,
                "optimizer_state_saved": False,
                "updated_utc": utc_now(),
            },
        )
    if rank == 0 and not smoke and recipe.get("export_final_checkpoint", False):
        endpoint = run / f"checkpoint-{steps}"
        if not (endpoint / "model.safetensors").is_file():
            raise RuntimeError("final endpoint inference weights missing")
        (run / "final").symlink_to(endpoint.name, target_is_directory=True)
        atomic_json(
            run / "FINAL_EXPORT.json",
            {
                "checkpoint": str(endpoint),
                "step": steps,
                "lr_schedule_steps": schedule_steps,
                "inference_only": True,
            },
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
