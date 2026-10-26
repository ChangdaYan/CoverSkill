"""Query encoding and exact single-vector retrieval; qrels are not read."""

from __future__ import annotations
import argparse, gzip, json, os
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoTokenizer
from .common import DATASET, QUERY_PROMPT, atomic_torch_save, iter_jsonl
from .modeling import Retriever, cached_features


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--top-k", type=int, default=50)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    x = torch.load(a.index, map_location="cpu", weights_only=True)
    ids = [str(s) for s in x["skill_ids"]]
    documents = F.normalize(x["embeddings"].float(), dim=-1).to(device)
    documents = F.normalize(documents.float(), dim=-1)
    queries = list(iter_jsonl(DATASET / "queries/test.jsonl"))
    positions = list(range(rank, len(queries), world))
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=False)
    model = Retriever.from_pretrained(a.model).to(device).eval()
    tokens = tok(
        [QUERY_PROMPT + str(queries[i]["query"]) for i in positions],
        truncation=True,
        max_length=4096,
        add_special_tokens=True,
    )["input_ids"]
    rows = []
    vectors = []
    with torch.inference_mode():
        for at in range(0, len(tokens), a.batch):
            q = (
                model.encode_anchor_features(
                    cached_features(
                        [torch.tensor(x) for x in tokens[at : at + a.batch]], device
                    )
                )
                .half()
                .float()
            )
            values, indices = (q @ documents.T).topk(min(a.top_k, len(ids)), dim=-1)
            vectors.append(q.cpu())
            for j, (scores, ix) in enumerate(
                zip(values.cpu().tolist(), indices.cpu().tolist())
            ):
                pos = positions[at + j]
                rows.append(
                    {
                        "position": pos,
                        "query_id": str(queries[pos]["id"]),
                        "skills": [ids[i] for i in ix],
                        "scores": scores,
                    }
                )
    part = a.output / f".rank{rank}.pt"
    atomic_torch_save(
        part, {"positions": positions, "vectors": torch.cat(vectors), "rows": rows}
    )
    dist.barrier()
    if rank == 0:
        embeddings = torch.empty(len(queries), model.hidden_size)
        allrows = []
        for r in range(world):
            f = a.output / f".rank{r}.pt"
            x = torch.load(f, map_location="cpu", weights_only=True)
            embeddings[x["positions"]] = x["vectors"]
            allrows.extend(x["rows"])
            f.unlink()
        with gzip.open(a.output / "rankings.tsv.gz", "wt") as f:
            for row in sorted(allrows, key=lambda r: r["position"]):
                for k, (sid, score) in enumerate(zip(row["skills"], row["scores"]), 1):
                    f.write(
                        f"{row['query_id']}\tQ0\t{sid}\t{k}\t{score:.9g}\tCoverSkill\n"
                    )
        np.savez_compressed(
            a.output / "embeddings.npz",
            query_ids=np.array([str(x["id"]) for x in queries]),
            skill_ids=np.array(ids),
            query_embeddings=embeddings.numpy(),
            skill_embeddings=documents.cpu().numpy(),
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
