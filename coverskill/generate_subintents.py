"""Generate train-only, evidence-validated gold-anchored subintents.

All training query/gold slots are generated directly with an unmodified LLM.
Raw attempts and rejected judgements are preserved for each local run.
"""

from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
import torch
from .common import (
    HARD_ROWS,
    QUERY_PROMPT,
    TRAIN_QRELS,
    TRAIN_QUERIES,
    TRAIN_SKILLS,
    artifact,
    atomic_json,
    atomic_torch_save,
    iter_jsonl,
    sha256_file,
    utc_now,
)
from .data import load_qrels, skill_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate frozen train-only Qwen3-32B intent views"
    )
    parser.add_argument("--mode", choices=("gold_anchored",), default="gold_anchored")
    parser.add_argument("--domain", choices=("full",), default="full")
    parser.add_argument("--model", default="Qwen/Qwen3-32B")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.54)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument(
        "--json-retries",
        type=int,
        default=2,
        help="Retry only individual generations whose structured JSON cannot be parsed.",
    )
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--engineering-only", action="store_true")
    parser.add_argument(
        "--sampler-backend", choices=("flashinfer", "native"), default="flashinfer"
    )
    return parser.parse_args()


def normalize(value: str) -> str:
    return " ".join(str(value).split())


def evidence_is_supported(query: str, spans: list[str]) -> bool:
    query_normalized = normalize(query).casefold()
    return bool(spans) and all(
        (
            normalize(span).casefold() in query_normalized
            for span in spans
            if normalize(span)
        )
    )


def gold_prompt(query: str, skill: dict) -> str:
    return f"Create one training-only capability intent for the part of the query that supports the reference skill.\nThe reference skill may only select which existing query requirement to focus on; it is not a source of new query\nfacts. Copy one or more exact evidence spans from the query. If the query does not support this reference, set\nsupported=false and return an empty subintent. Never output a skill ID or introduce a tool/capability term that only\nappears in the reference. Return JSON only.\n\nQuery:\n{query}\n\nReference skill (attention selector only):\n{skill_text(skill)[:5000]}\n"


def exact_skill_name_leaked(query: str, skill: dict, subintent: str) -> bool:
    name = normalize(skill.get("name") or "")
    return (
        len(name) >= 4
        and name.casefold() not in query.casefold()
        and (name.casefold() in subintent.casefold())
    )


def _balanced_json_objects(text: str):
    """Yield balanced JSON objects without being confused by braces in strings."""
    start = None
    depth = 0
    quoted = False
    escaped = False
    for index, character in enumerate(text):
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
            continue
        if character == '"':
            quoted = True
        elif character == "{":
            if depth == 0:
                start = index
            depth += 1
        elif character == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                yield text[start : index + 1]
                start = None


def extract_payload(raw: str, mode: str) -> dict:
    """Extract and minimally validate one query-only or gold-anchored payload."""
    text = str(raw or "").strip()
    text = re.sub("<think>.*?</think>", "", text, flags=re.S | re.I).strip()
    fenced = re.findall("```(?:json)?\\s*(.*?)\\s*```", text, flags=re.S | re.I)
    candidates = [*fenced, *_balanced_json_objects(text), text]
    seen = set()
    errors = []
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            payload = json.loads(candidate)
            if not isinstance(payload, dict):
                raise ValueError("top-level JSON must be an object")
            values = [payload]
            if not isinstance(payload.get("supported"), bool):
                raise ValueError("supported must be boolean")
            for value in values:
                if not isinstance(value, dict):
                    raise ValueError("each subintent must be an object")
                if not isinstance(value.get("subintent"), str):
                    raise ValueError("subintent must be a string")
                if not isinstance(value.get("query_evidence"), list):
                    raise ValueError("query_evidence must be an array")
                if not isinstance(value.get("confidence"), (int, float)):
                    raise ValueError("confidence must be numeric")
            return payload
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            errors.append(str(exc))
    summary = errors[-1] if errors else "empty output or no JSON object"
    raise ValueError(f"invalid structured JSON: {summary}")


def render_no_thinking(tokenizer, prompt: str, retry_number: int = 0) -> str:
    """Render Qwen3 chat input while guaranteeing non-thinking mode."""
    if retry_number:
        prompt += f"\n\nJSON correction attempt {retry_number}: Return the requested JSON object immediately. Do not include analysis, reasoning, Markdown fences, or any text outside the JSON object."
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError as exc:
        raise RuntimeError(
            "Qwen3 tokenizer does not support enable_thinking=False"
        ) from exc


def output_text_and_finish_reason(output) -> tuple[str, str]:
    candidates = getattr(output, "outputs", None) or []
    if not candidates:
        return ("", "missing_output")
    candidate = candidates[0]
    return (
        str(getattr(candidate, "text", "") or ""),
        str(getattr(candidate, "finish_reason", "unknown")),
    )


def raw_records_sha256(records: dict[str, dict]) -> str:
    """Hash JSON-only raw records without modifying or normalizing their contents."""
    digest = hashlib.sha256()
    for key in sorted(records):
        digest.update(key.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(
            json.dumps(
                records[key], ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    from .common import train_sources
    from .common import resolve_generator, resolve_backbone

    train_sources()
    args.model = resolve_generator(args.model)
    args.tokenizer = str(resolve_backbone(args.tokenizer))
    if args.max_items and (not args.engineering_only):
        raise RuntimeError("--max-items requires --engineering-only")
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = (
        "1" if args.sampler_backend == "flashinfer" else "0"
    )
    triton = (
        args.output.parent / "triton_generation" / f"pid-{os.getpid()}-{time.time_ns()}"
    )
    triton.mkdir(parents=True, exist_ok=False)
    os.environ["TRITON_CACHE_DIR"] = str(triton)
    query_rows = {str(row["id"]): row for row in iter_jsonl(TRAIN_QUERIES)}
    qrels = load_qrels(TRAIN_QRELS)
    all_domain_ids = [qid for qid in query_rows if qid in qrels]
    domain_ids = all_domain_ids
    hard_ids = set()
    reused_hard = None
    reused_hard_path = None
    skills = {str(row["id"]): row for row in iter_jsonl(TRAIN_SKILLS)}
    items = []
    items = [(qid, sid) for qid in domain_ids for sid in sorted(qrels[qid])]
    if args.max_items:
        items = items[: args.max_items]
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    generation_tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=False
    )
    encoder_tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=True, local_files_only=False
    )
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        disable_custom_all_reduce=True,
        enforce_eager=True,
    )
    query_schema = {
        "type": "object",
        "properties": {
            "subintents": {
                "type": "array",
                "minItems": 1,
                "maxItems": 5,
                "items": {
                    "type": "object",
                    "properties": {
                        "subintent": {"type": "string"},
                        "query_evidence": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": ["subintent", "query_evidence", "confidence"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["subintents"],
        "additionalProperties": False,
    }
    gold_schema = {
        "type": "object",
        "properties": {
            "supported": {"type": "boolean"},
            "subintent": {"type": "string"},
            "query_evidence": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["supported", "subintent", "query_evidence", "confidence"],
        "additionalProperties": False,
    }
    schema = gold_schema
    sampling = SamplingParams(
        temperature=0,
        max_tokens=args.max_new_tokens,
        structured_outputs=StructuredOutputsParams(
            json=schema, disable_additional_properties=True
        ),
    )
    partial = args.output.with_suffix(args.output.suffix + ".partial.jsonl")
    completed: dict[str, dict] = {}
    if partial.is_file():
        for row in iter_jsonl(partial):
            required_raw_fields = {
                "raw_payload",
                "unfiltered_subintents",
                "raw_generation_attempts",
            }
            if not required_raw_fields <= set(row):
                raise RuntimeError(
                    f"partial cache predates raw-preserving schema v2: {partial}; archive it before resuming"
                )
            completed[str(row["item_key"])] = row
    keys = [f"{qid}\t{sid or ''}" for qid, sid in items]
    unexpected_partial_keys = set(completed).difference(keys)
    if unexpected_partial_keys:
        examples = sorted(unexpected_partial_keys)[:3]
        raise RuntimeError(
            f"partial cache contains rows outside the current disjoint generation partition (examples={examples}); use a partial file from this exact generation domain"
        )
    missing = [
        (key, qid, sid) for key, (qid, sid) in zip(keys, items) if key not in completed
    ]
    reused_item_count = 0
    total_item_count = reused_item_count + len(items)
    completed_at_start = len(completed)
    started = time.time()
    progress = args.output.with_suffix(args.output.suffix + ".progress.json")
    for begin in range(0, len(missing), args.batch_size):
        batch = missing[begin : begin + args.batch_size]
        base_prompts = {}
        for key, qid, sid in batch:
            base_prompts[key] = gold_prompt(str(query_rows[qid]["query"]), skills[sid])
        pending = list(batch)
        payloads = {}
        successful_generations = {}
        raw_generation_attempts: dict[str, list[dict]] = defaultdict(list)
        parse_failures: dict[str, list[dict]] = defaultdict(list)
        for retry_number in range(args.json_retries + 1):
            if not pending:
                break
            prompts = [
                render_no_thinking(
                    generation_tokenizer, base_prompts[key], retry_number
                )
                for key, _, _ in pending
            ]
            outputs = llm.generate(prompts, sampling, use_tqdm=False)
            if len(outputs) != len(pending):
                raise RuntimeError(
                    f"vLLM returned {len(outputs)} outputs for {len(pending)} prompts"
                )
            retry_pending = []
            for item, output in zip(pending, outputs):
                key, _, _ = item
                raw, finish_reason = output_text_and_finish_reason(output)
                try:
                    payload = extract_payload(raw, args.mode)
                    payloads[key] = payload
                    successful_generations[key] = {
                        "attempt": retry_number + 1,
                        "finish_reason": finish_reason,
                        "raw_output_text": raw,
                    }
                    raw_generation_attempts[key].append(
                        {
                            "attempt": retry_number + 1,
                            "finish_reason": finish_reason,
                            "parse_valid": True,
                            "raw_output_text": raw,
                        }
                    )
                except ValueError as exc:
                    raw_generation_attempts[key].append(
                        {
                            "attempt": retry_number + 1,
                            "finish_reason": finish_reason,
                            "parse_valid": False,
                            "parse_error": str(exc),
                            "raw_output_text": raw,
                        }
                    )
                    parse_failures[key].append(
                        {
                            "attempt": retry_number + 1,
                            "error": str(exc),
                            "finish_reason": finish_reason,
                        }
                    )
                    retry_pending.append(item)
            pending = retry_pending
        for key, qid, sid in batch:
            generation_valid = key in payloads
            parsed_payload = payloads.get(
                key,
                {
                    "supported": False,
                    "subintent": "",
                    "query_evidence": [],
                    "confidence": 0.0,
                },
            )
            raw_payload = copy.deepcopy(parsed_payload) if generation_valid else None
            raw_payload_before_filter = (
                json.dumps(
                    raw_payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                if raw_payload is not None
                else None
            )
            unfiltered = (
                copy.deepcopy(raw_payload["subintents"])
                if generation_valid and False
                else [copy.deepcopy(raw_payload)]
                if generation_valid
                else []
            )
            source = copy.deepcopy(unfiltered)
            query = str(query_rows[qid]["query"])
            accepted = []
            filter_decisions = []
            for value_index, value in enumerate(source):
                subintent = normalize(value.get("subintent", ""))
                evidence = [
                    normalize(span)
                    for span in value.get("query_evidence", [])
                    if normalize(span)
                ]
                reasons = []
                if not value.get("supported", True):
                    reasons.append("model_unsupported")
                if not subintent:
                    reasons.append("empty_subintent")
                if not evidence:
                    reasons.append("missing_query_evidence")
                elif not evidence_is_supported(query, evidence):
                    reasons.append("nonverbatim_query_evidence")
                leaked = bool(
                    sid and exact_skill_name_leaked(query, skills[sid], subintent)
                )
                if leaked:
                    reasons.append("exact_skill_name_leakage")
                if not reasons:
                    accepted.append(
                        {
                            "subintent": subintent,
                            "query_evidence": evidence,
                            "confidence": float(value.get("confidence", 0.0)),
                        }
                    )
                filter_decisions.append(
                    {
                        "subintent_index": value_index,
                        "accepted": not reasons,
                        "reasons": reasons,
                    }
                )
            if raw_payload_before_filter is not None:
                raw_payload_after_filter = json.dumps(
                    raw_payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                if raw_payload_after_filter != raw_payload_before_filter:
                    raise RuntimeError("filtering mutated the original raw payload")
            row = {
                "item_key": key,
                "query_id": qid,
                "skill_id": sid,
                "generation_valid": generation_valid,
                "generation_attempt_count": len(raw_generation_attempts[key]),
                "raw_generation_attempts": raw_generation_attempts[key],
                "successful_generation": successful_generations.get(key),
                "raw_payload": raw_payload,
                "unfiltered_subintents": unfiltered,
                "filtered_subintents": accepted,
                "accepted": copy.deepcopy(accepted),
                "filter_decisions": filter_decisions,
                "parse_failures": parse_failures[key],
            }
            completed[key] = row
            with partial.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
                handle.flush()
                os.fsync(handle.fileno())
        done = reused_item_count + len(completed)
        newly_completed = max(1, len(completed) - completed_at_start)
        elapsed = max(1e-06, time.time() - started)
        atomic_json(
            progress,
            {
                "status": "generating",
                "mode": args.mode,
                "domain": args.domain,
                "completed": done,
                "reused_hard_items": reused_item_count,
                "invalid_fallbacks": 0
                + sum(
                    (
                        not row.get("generation_valid", True)
                        for row in completed.values()
                    )
                ),
                "total": total_item_count,
                "estimated_remaining_seconds": (len(items) - len(completed))
                * elapsed
                / newly_completed,
                "updated_utc": utc_now(),
            },
        )
    raw_generation_records: dict[str, dict] = {}
    raw_subintent_tokens: dict[str, list[torch.Tensor]] = defaultdict(list)
    raw_alignments: dict[str, list[list[tuple[str, float]]]] = defaultdict(list)
    filtered_subintent_tokens: dict[str, list[torch.Tensor]] = defaultdict(list)
    filtered_alignments: dict[str, list[list[tuple[str, float]]]] = defaultdict(list)
    filter_reason_counts: Counter = Counter()
    accepted_rows = rejected = 0
    unfiltered_subintent_count = 0

    def alignment_for(
        qid: str, sid: str | None, value: dict
    ) -> list[tuple[str, float]]:
        confidence = float(value.get("confidence", 0.0))
        if sid is not None:
            return [(sid, confidence)]
        branch = []
        query_gold = sorted(qrels[qid])
        evidence_text = " ".join(
            (str(span) for span in value.get("query_evidence", []))
        ).casefold()
        for gold_sid in query_gold:
            name = normalize(skills[gold_sid].get("name") or "").casefold()
            if name and name in evidence_text:
                branch.append((gold_sid, confidence))
        if not branch and len(query_gold) == 1:
            branch = [(query_gold[0], confidence)]
        return branch

    for key in keys:
        row = completed[key]
        qid, sid = (row["query_id"], row["skill_id"])
        raw_generation_records[key] = {
            field: copy.deepcopy(row[field])
            for field in (
                "item_key",
                "query_id",
                "skill_id",
                "generation_valid",
                "generation_attempt_count",
                "raw_generation_attempts",
                "successful_generation",
                "raw_payload",
                "unfiltered_subintents",
            )
        }
        unfiltered_subintent_count += len(row["unfiltered_subintents"])
        for value in row["unfiltered_subintents"]:
            raw_text = value.get("subintent", "")
            if not isinstance(raw_text, str) or not raw_text.strip():
                continue
            encoded = encoder_tokenizer(
                QUERY_PROMPT + raw_text,
                truncation=True,
                max_length=1024,
                add_special_tokens=True,
            )["input_ids"]
            raw_subintent_tokens[qid].append(torch.tensor(encoded, dtype=torch.long))
            raw_alignments[qid].append(alignment_for(qid, sid, value))
        for value in row["filtered_subintents"]:
            encoded = encoder_tokenizer(
                QUERY_PROMPT + value["subintent"],
                truncation=True,
                max_length=1024,
                add_special_tokens=True,
            )["input_ids"]
            filtered_subintent_tokens[qid].append(
                torch.tensor(encoded, dtype=torch.long)
            )
            filtered_alignments[qid].append(alignment_for(qid, sid, value))
            accepted_rows += 1
        for decision in row["filter_decisions"]:
            filter_reason_counts.update(decision["reasons"])
        if not row["filtered_subintents"]:
            rejected += 1
    invalid_rows = [
        row for row in completed.values() if not row.get("generation_valid", True)
    ]
    recovered_rows = [
        row
        for row in completed.values()
        if row.get("generation_valid", True)
        and int(row.get("generation_attempt_count", 1)) > 1
    ]
    if len(raw_generation_records) != total_item_count:
        raise RuntimeError(
            f"raw generation record count mismatch: {len(raw_generation_records)} != {total_item_count}"
        )
    raw_digest = raw_records_sha256(raw_generation_records)
    filtered_tokens = dict(filtered_subintent_tokens)
    filtered_alignment_rows = dict(filtered_alignments)
    final = {
        "schema_version": 2,
        "complete": True,
        "mode": args.mode,
        "domain": args.domain,
        "raw_generation_records": raw_generation_records,
        "raw_generation_records_sha256": raw_digest,
        "raw_subintent_tokens": dict(raw_subintent_tokens),
        "raw_alignments": dict(raw_alignments),
        "filtered_subintent_tokens": filtered_tokens,
        "filtered_alignments": filtered_alignment_rows,
        "subintent_tokens": filtered_tokens,
        "alignments": filtered_alignment_rows,
        "query_count": len(domain_ids)
        if not args.max_items
        else len(set((r["query_id"] for r in completed.values()))),
        "item_count": total_item_count,
        "raw_generation_record_count": len(raw_generation_records),
        "unfiltered_generated_subintent_count": unfiltered_subintent_count,
        "accepted_subintent_count": accepted_rows,
        "rejected_item_count": rejected,
        "filter_rejection_reason_counts": dict(sorted(filter_reason_counts.items())),
        "invalid_generation_fallback_count": 0 + len(invalid_rows),
        "recovered_after_json_retry_count": 0 + len(recovered_rows),
        "json_retries_per_item": args.json_retries,
        "thinking_mode_enabled": False,
        "raw_generation_preserved": True,
        "raw_generation_layer_immutable": True,
        "filtered_view_derived_from_deep_copy": True,
        "default_training_view": "filtered",
        "hard_partition_reused": False,
        "reused_hard_query_count": 0,
        "reused_hard_item_count": reused_item_count,
        "generation_uses_official_train_only": True,
        "official_test_used": False,
        "gold_is_attention_selector_only": True,
        "verbatim_query_evidence_required": True,
        "engineering_only": args.engineering_only,
    }
    atomic_torch_save(args.output, final)
    atomic_json(
        args.output.with_suffix(args.output.suffix + ".manifest.json"),
        {
            **{
                key: value
                for key, value in final.items()
                if key
                not in {
                    "raw_generation_records",
                    "raw_subintent_tokens",
                    "raw_alignments",
                    "filtered_subintent_tokens",
                    "filtered_alignments",
                    "subintent_tokens",
                    "alignments",
                }
            },
            "model": str(Path(args.model).resolve()),
            "model_index_sha256": sha256_file(
                Path(args.model) / "model.safetensors.index.json"
            ),
            "reused_hard_artifact": None,
            "artifact": artifact(
                args.output, "frozen train-only Qwen3-32B subintent/alignment cache"
            ),
            "updated_utc": utc_now(),
        },
    )
    failure_path = args.output.with_suffix(args.output.suffix + ".failures.json")
    if invalid_rows:
        atomic_json(
            failure_path,
            {
                "complete": True,
                "fallback_policy": "empty supervision for only the permanently invalid item",
                "failure_count": len(invalid_rows),
                "items": invalid_rows,
                "updated_utc": utc_now(),
            },
        )
    else:
        failure_path.unlink(missing_ok=True)
    progress.unlink(missing_ok=True)
    partial.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
