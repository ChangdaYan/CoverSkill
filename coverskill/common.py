from __future__ import annotations
import datetime as dt
import gzip, hashlib, io, json, os
from pathlib import Path
from typing import Iterable, Iterator

ROOT = Path(os.environ.get("COVERSKILL_WORK", "work")).resolve()
DATASET = Path(os.environ.get("COVERSKILL_DATA", "data/skillret")).resolve()
RAW_BASE_MODEL = os.environ.get("COVERSKILL_BASE_MODEL", "Qwen/Qwen3-Embedding-0.6B")
QUERY_PROMPT = "Instruct: Given a skill search query, retrieve relevant skills that match the query\nQuery: "
TRAIN_QUERIES = DATASET / "queries/train.jsonl"
TRAIN_QRELS = DATASET / "qrels/train.jsonl"
TRAIN_SKILLS = DATASET / "skills/train.jsonl"
FULL_QUERY_TOKENS = ROOT / "full_query_tokens.pt"
TRAIN_DECOMPOSITION = ROOT / "subintents.pt"
TRAIN_SKILL_CACHE = ROOT / "train_documents.pt"
TRAIN_SKILL_MANIFEST = ROOT / "train_documents.manifest.json"
HARD_ROWS = ROOT / "hard_queries.jsonl.gz"
RUNS = ROOT / "runs"
CONFIGS = Path("configs")
CACHE = ROOT


def resolve_backbone(source):
    path = Path(source)
    if not path.exists():
        if source != "Qwen/Qwen3-Embedding-0.6B":
            raise ValueError("Use the official unmodified pretrained backbone.")
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(source))
    return path.resolve()


def resolve_generator(source):
    path = Path(source)
    if not path.exists():
        if source != "Qwen/Qwen3-32B":
            raise ValueError("Use the unmodified pretrained Qwen3-32B generator.")
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(source))
    if (path / "TRAINING_ANCESTRY.json").exists():
        raise ValueError("A SkillRet-trained generator is not supported.")
    return str(path.resolve())


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def iter_jsonl(path: Path) -> Iterator[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_gzip_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, compresslevel=6, mtime=0
        ) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as text:
                for row in rows:
                    text.write(
                        json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                        + "\n"
                    )
    os.replace(temporary, path)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_sha(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def artifact(path: Path, role: str) -> dict:
    return {
        "path": str(path.resolve()),
        "role": role,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def atomic_torch_save(path: Path, payload: object) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


REVISION = "7cae7cfbad2b0e1ebc9170892f568993aae543b0"


def train_sources(dataset=None):
    root = Path(dataset) if dataset is not None else DATASET
    info = read_json(root / "dataset.json")
    if info["revision"] != REVISION:
        raise ValueError("Use the pinned original SkillRet revision.")
    paths = [
        root / "queries/train.jsonl",
        root / "qrels/train.jsonl",
        root / "skills/train.jsonl",
    ]
    for p in paths:
        if sha256_file(p) != info["files"][str(p.relative_to(root))]:
            raise ValueError(f"Dataset input changed: {p}")
    return [artifact(p, "official training split") for p in paths]


def stage1_preflight(args):
    inputs = train_sources(args.queries.parent.parent)
    parent = Path(args.checkpoint)
    if parent.exists() and (
        (parent / "TRAINING_ANCESTRY.json").exists()
        or (parent / "training_manifest.json").exists()
    ):
        raise ValueError(
            "Stage 1 must initialize from an unmodified pretrained backbone."
        )
    if (
        parent.exists()
        and sha256_file(parent / "model.safetensors")
        != "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd"
    ):
        raise ValueError(
            "Local backbone differs from the unmodified Qwen3-Embedding-0.6B release."
        )
    if not parent.exists() and str(args.checkpoint) != "Qwen/Qwen3-Embedding-0.6B":
        raise ValueError(
            "Use Qwen/Qwen3-Embedding-0.6B or its unmodified local snapshot."
        )
    source = read_json(args.view_manifest)
    if (
        not source.get("complete")
        or source.get("official_test_used")
        or not source.get("generation_uses_official_train_only")
    ):
        raise ValueError("Invalid subintent generation provenance.")
    known = {str(r["id"]) for r in iter_jsonl(args.queries)}
    for row in iter_jsonl(args.coverage_rewrites):
        if str(row["parent_query_id"]) not in known:
            raise ValueError("Rewrite parent outside train.")
    inputs += [
        artifact(args.subintent_cache, "pretrained-LLM train-only subintents"),
        artifact(args.coverage_rewrites, "train-only rewrites"),
    ]
    if parent.exists():
        inputs.extend(
            artifact(p, "unmodified pretrained backbone")
            for p in sorted(parent.glob("*.safetensors"))
        )
    atomic_json(
        Path(args.output_dir) / "preflight.json",
        {
            "training_depth": 1,
            "parent": str(args.checkpoint),
            "origin": "unmodified pretrained backbone",
            "inputs": inputs,
            "official_test_used": False,
        },
    )


def stage2_preflight(recipe):
    inputs = train_sources()
    parent = Path(recipe["parent_model"])
    ancestry = read_json(parent / "TRAINING_ANCESTRY.json")
    if ancestry.get("serial_training_depth") != 1 or not ancestry.get(
        "shared_trainable_query_document_encoder"
    ):
        raise ValueError("Stage 2 requires its own depth-one MGCA parent.")
    weight_sha = sha256_file(parent / "model.safetensors")
    docs = read_json(recipe["train_skill_manifest"])
    hard = read_json(recipe["hard_rows_manifest"])
    teacher = read_json(recipe["teacher_cache"] + ".manifest.json")
    if not (
        docs["complete"]
        and docs["model_sha256"] == weight_sha
        and docs["split"] == "train"
        and docs["layout"] == "facets"
    ):
        raise ValueError("Wrong Stage-1 document matrix.")
    if (
        hard["selection_model_sha256"] != weight_sha
        or hard["document_cache"]["sha256"] != docs["cache_sha256"]
    ):
        raise ValueError("Hard queries do not belong to this parent and matrix.")
    if (
        teacher["teacher_model_sha256"] != weight_sha
        or teacher["document_cache_sha256"] != docs["cache_sha256"]
    ):
        raise ValueError("Teacher and parent document space differ.")
    if any(x.get("official_test_used", False) for x in [docs, hard, teacher]):
        raise ValueError("Test data cannot construct training artifacts.")
    if (
        recipe["logical_batch"],
        recipe["dynamic_negative_count"],
        recipe["dynamic_negative_multiplier"],
    ) != (192, 96, 2):
        raise ValueError(
            "This release contains the main method, with batch192 and Top96x2."
        )
    inputs.extend(
        artifact(Path(recipe[k]), k)
        for k in [
            "train_skill_cache",
            "hard_rows",
            "teacher_cache",
            "train_decomposition",
        ]
    )
    atomic_json(
        ROOT / "stage2_preflight.json",
        {
            "training_depth": 2,
            "parent": str(parent),
            "inputs": inputs,
            "official_test_used": False,
        },
    )


def validate_parent_matrix(parent, cache):
    parent, cache = Path(parent), Path(cache)
    a = read_json(parent / "TRAINING_ANCESTRY.json")
    d = read_json(cache.with_suffix(".manifest.json"))
    if (
        a.get("serial_training_depth") != 1
        or not d.get("complete")
        or d.get("split") != "train"
    ):
        raise ValueError(
            "A train matrix from the own depth-one Stage-1 parent is required."
        )
    if d["model_sha256"] != sha256_file(parent / "model.safetensors") or d[
        "cache_sha256"
    ] != sha256_file(cache):
        raise ValueError("Frozen matrix and declared parent differ.")
