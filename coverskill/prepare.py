"""Download the original split and tokenize the official training queries."""

from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
from .common import (
    QUERY_PROMPT,
    atomic_json,
    artifact,
    atomic_torch_save,
    iter_jsonl,
    sha256_file,
)
from .common import REVISION


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("download")
    d.add_argument("--output", type=Path, default=Path("data/skillret"))
    d.add_argument(
        "--source",
        type=Path,
        help="Use an already-downloaded copy of the pinned official snapshot.",
    )
    d.add_argument(
        "--only-qrels",
        action="store_true",
        help="Download only test qrels to evaluate released predictions.",
    )
    t = sub.add_parser("tokenize")
    t.add_argument("--data", type=Path, default=Path("data/skillret"))
    t.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    t.add_argument("--output", type=Path, default=Path("work/full_query_tokens.pt"))
    a = p.parse_args()
    if a.command == "download":
        info = {"repository": "ThakiCloud/SKILLRET", "revision": REVISION, "files": {}}
        for kind in ["qrels"] if a.only_qrels else ["queries", "qrels", "skills"]:
            for split in ["test"] if a.only_qrels else ["train", "test"]:
                relative = f"{kind}/{split}.jsonl"
                if a.source is not None:
                    source = a.source / relative
                    if not source.is_file():
                        source = a.source / "data" / relative
                    if not source.is_file():
                        raise FileNotFoundError(source)
                else:
                    from huggingface_hub import hf_hub_download

                    source = hf_hub_download(
                        "ThakiCloud/SKILLRET",
                        filename="data/" + relative,
                        repo_type="dataset",
                        revision=REVISION,
                    )
                target = a.output / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if Path(source).resolve() != target.resolve():
                    shutil.copyfile(source, target)
                info["files"][relative] = sha256_file(target)
        atomic_json(a.output / "dataset.json", info)
    else:
        import torch
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=False)
        rows = list(iter_jsonl(a.data / "queries/train.jsonl"))
        ids = [str(r["id"]) for r in rows]
        tokens = tok(
            [QUERY_PROMPT + str(r["query"]) for r in rows],
            truncation=True,
            max_length=4096,
            add_special_tokens=True,
        )["input_ids"]
        atomic_torch_save(
            a.output, {"queries": dict(zip(ids, [torch.tensor(x) for x in tokens]))}
        )
        atomic_json(
            a.output.with_suffix(".manifest.json"),
            {
                "complete": True,
                "origin": "deterministic tokenization of official train queries",
                "query_count": len(rows),
                "inputs": [
                    artifact(a.data / "queries/train.jsonl", "official train queries")
                ],
                "official_test_used": False,
            },
        )
    print(a.output)


if __name__ == "__main__":
    main()
