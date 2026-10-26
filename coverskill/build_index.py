"""Build one frozen vector per skill from four separately encoded facets."""

from __future__ import annotations
import argparse, os
from pathlib import Path
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoTokenizer
from .common import DATASET, atomic_json, atomic_torch_save, iter_jsonl, sha256_file
from .modeling import Retriever, cached_features
from .views import four_facets, FACET_KEYS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--split", choices=["train", "test"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-skills", type=int, default=3)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError("Refusing to overwrite an index.")
    if a.split == "train":
        from .common import train_sources

        train_sources()
    ancestor = __import__("json").loads(
        (a.model / "TRAINING_ANCESTRY.json").read_text()
    )
    if ancestor["serial_training_depth"] != 1:
        raise ValueError("Encode skills with the MGCA Stage-1 parent.")
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    source = DATASET / f"skills/{a.split}.jsonl"
    skills = list(iter_jsonl(source))
    positions = list(range(rank, len(skills), world))
    vectors = []
    tokenizer = AutoTokenizer.from_pretrained(a.model, trust_remote_code=False)
    model = Retriever.from_pretrained(a.model).to(device).eval()
    with torch.inference_mode():
        for at in range(0, len(positions), a.batch_skills):
            texts = [
                text
                for pos in positions[at : at + a.batch_skills]
                for text in four_facets(skills[pos])
            ]
            tokens = tokenizer(
                texts, truncation=True, max_length=512, add_special_tokens=True
            )["input_ids"]
            chunks = []
            micro = 12 if a.split == "train" else 4
            for begin in range(0, len(tokens), micro):
                chunks.append(
                    model.encode_anchor_features(
                        cached_features(
                            [torch.tensor(x) for x in tokens[begin : begin + micro]],
                            device,
                        )
                    )
                )
            z = torch.cat(chunks).reshape(-1, 4, model.hidden_size)
            if a.split == "train":
                z = F.normalize(z.float(), dim=-1)
            # Normalized slots -> normalized mean -> frozen FP16 storage.
            vectors.append(F.normalize(z.float().mean(1), dim=-1).cpu().half())
    part = a.output.with_name(f".{a.output.name}.rank{rank}.pt")
    atomic_torch_save(part, {"positions": positions, "embeddings": torch.cat(vectors)})
    dist.barrier()
    if rank == 0:
        matrix = torch.empty(len(skills), model.hidden_size, dtype=torch.float16)
        for r in range(world):
            path = a.output.with_name(f".{a.output.name}.rank{r}.pt")
            x = torch.load(path, map_location="cpu", weights_only=True)
            matrix[x["positions"]] = x["embeddings"]
            path.unlink()
        values = {"skill_ids": [str(s["id"]) for s in skills], "embeddings": matrix}
        atomic_torch_save(a.output, {"train": values} if a.split == "train" else values)
        atomic_json(
            a.output.with_suffix(".manifest.json"),
            {
                "complete": True,
                "split": a.split,
                "layout": "facets",
                "max_length": 512,
                "model": str(a.model.resolve()),
                "model_sha256": sha256_file(a.model / "model.safetensors"),
                "skill_count": len(skills),
                "cache_sha256": sha256_file(a.output),
                "skills_sha256": sha256_file(source),
                "view_slots": list(FACET_KEYS),
                "aggregation": "normalized mean of normalized vectors; FP16 storage",
                "official_test_qrels_used": False,
                "query_or_qrels_used": False,
            },
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
