"""Qwen3-32B generation and independent verification of complete-gold queries.

The raw record stream is append-only and resumable. Rejected generations stay in
the stream for auditing; no output is silently relabelled or deleted.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

from .common import atomic_json, iter_jsonl, sha256_file, utc_now


def compact(text: str) -> str:
    return " ".join(str(text).split())


def render(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def generation_prompt(task: dict) -> str:
    requirements = "\n".join(
        f"- facet {i + 1}: {facet['intent']}" for i, facet in enumerate(task["facets"])
    )
    focus = next(
        i + 1
        for i, facet in enumerate(task["facets"])
        if facet["skill_id"] == task["focus_skill_id"]
    )
    return f"""Rewrite the complete USER REQUEST below as one natural, realistic full user request.
Make facet {focus} clearer in the full request, but preserve EVERY listed facet and all constraints
from the original request. Do not add a new task, tool, file type, platform, or capability.
Keep every number, version, path, code identifier, required field name, and output format exactly.
Keep the entire request, including its final requirement. End with a complete sentence.
Do not write a subtask, numbered list, explanation, or skill names. Do not copy the
facet wording mechanically. Return JSON only: {{"rewrite": "..."}}.

Original complete request:
{task["query"]}

Requirements already supported by that request:
{requirements}
"""


def verification_prompt(task: dict, rewrite: str) -> str:
    requirements = "\n".join(
        f"- {facet['skill_id']}: {facet['intent']}" for facet in task["facets"]
    )
    return f"""Independently audit a rewritten user request. The candidate must remain a complete,
natural request with exactly the same required capability facets as the original.
Reject if any original facet is missing, if a new task is introduced, or if it
mentions a skill/tool name absent from the original. For each retained facet,
copy a short EXACT substring from the candidate as evidence. Do not infer a
facet solely from the reference descriptions. Return JSON only.

Original: {task["query"]}
Candidate: {rewrite}
Required facets (IDs are bookkeeping, not content to repeat):
{requirements}
"""


def parse_json(text: str) -> dict | None:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        value = json.loads(text[start : end + 1])
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None


def deterministic_checks(
    task: dict, rewrite: str, verdict: dict | None, finish_reason: str = "stop"
) -> list[str]:
    reasons = []
    original = compact(task["query"])
    rewrite = compact(rewrite)
    if not rewrite or rewrite.casefold() == original.casefold():
        reasons.append("empty_or_identical")
    if finish_reason != "stop":
        reasons.append("generation_incomplete")
    if rewrite and re.search(
        r"\b(?:a|the|of|to|with|for|like|that|between|named)$", rewrite, re.I
    ):
        reasons.append("sentence_truncated")
    if rewrite and not re.search(r"[.!?][\"'”’)]?$", rewrite):
        reasons.append("missing_terminal_punctuation")
    numbers = lambda value: set(
        re.findall(r"(?<!\w)\d+(?:[.,]\d+)*(?:%|[A-Za-z]+)?", value)
    )
    if numbers(original) - numbers(rewrite):
        reasons.append("numeric_constraint_changed")
    code_literals = set(re.findall(r"`([^`]+)`", original))
    if any(f"`{literal}`" not in rewrite for literal in code_literals):
        reasons.append("code_identifier_changed")
    if len(rewrite) < max(35, int(0.55 * len(original))) or len(rewrite) > min(
        4500, int(1.8 * len(original)) + 100
    ):
        reasons.append("length_out_of_range")
    if re.search(r"\b(?:skill|facet)\s*[1-3]\b", rewrite, re.I):
        reasons.append("meta_language")
    for facet in task["facets"]:
        name = compact(facet.get("skill_name") or "")
        if (
            len(name) >= 5
            and name.casefold() not in original.casefold()
            and name.casefold() in rewrite.casefold()
        ):
            reasons.append("skill_name_leak")
            break
    if (
        not verdict
        or verdict.get("same_complete_requirements") is not True
        or verdict.get("natural") is not True
    ):
        reasons.append("verifier_reject")
    if verdict and verdict.get("new_requirements") is True:
        reasons.append("new_requirements")
    evidence = verdict.get("evidence", []) if verdict else []
    by_skill = {
        str(x.get("skill_id")): compact(x.get("span", ""))
        for x in evidence
        if isinstance(x, dict)
    }
    for sid in task["gold_skill_ids"]:
        span = by_skill.get(sid, "")
        if len(span) < 4 or span.casefold() not in rewrite.casefold():
            reasons.append(f"missing_exact_evidence:{sid}")
    return sorted(set(reasons))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-32B")
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--engineering-only", action="store_true")
    args = parser.parse_args()
    from .common import train_sources
    from .common import resolve_generator

    train_sources()
    args.model = resolve_generator(args.model)
    if args.max_items and not args.engineering_only:
        raise SystemExit("--max-items is allowed only for engineering smoke")
    if args.output.exists():
        raise SystemExit(f"immutable complete output exists: {args.output}")
    tasks = list(iter_jsonl(args.tasks))
    if args.max_items:
        tasks = tasks[: args.max_items]
    by_qid = {x["parent_query_id"]: x for x in tasks}
    if len(by_qid) != len(tasks):
        raise SystemExit("task IDs are not unique")
    partial = args.output.with_suffix(args.output.suffix + ".partial.jsonl")
    completed = {}
    if partial.is_file():
        for row in iter_jsonl(partial):
            completed[row["parent_query_id"]] = row
    missing = [task for task in tasks if task["parent_query_id"] not in completed]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    os.environ["TRITON_CACHE_DIR"] = str(args.output.parent / "triton_cache")
    Path(os.environ["TRITON_CACHE_DIR"]).mkdir(exist_ok=True)
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=False
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
    rewrite_schema = {
        "type": "object",
        "properties": {"rewrite": {"type": "string"}},
        "required": ["rewrite"],
        "additionalProperties": False,
    }
    verify_schema = {
        "type": "object",
        "properties": {
            "same_complete_requirements": {"type": "boolean"},
            "natural": {"type": "boolean"},
            "new_requirements": {"type": "boolean"},
            "evidence": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "skill_id": {"type": "string"},
                        "span": {"type": "string"},
                    },
                    "required": ["skill_id", "span"],
                    "additionalProperties": False,
                },
            },
        },
        "required": [
            "same_complete_requirements",
            "natural",
            "new_requirements",
            "evidence",
        ],
        "additionalProperties": False,
    }
    sampling_generate = SamplingParams(
        temperature=0.2,
        max_tokens=args.max_new_tokens,
        structured_outputs=StructuredOutputsParams(
            json=rewrite_schema, disable_additional_properties=True
        ),
    )
    sampling_verify = SamplingParams(
        temperature=0,
        max_tokens=args.max_new_tokens,
        structured_outputs=StructuredOutputsParams(
            json=verify_schema, disable_additional_properties=True
        ),
    )
    started = time.time()
    for begin in range(0, len(missing), args.batch_size):
        batch = missing[begin : begin + args.batch_size]
        gen_prompts = [render(tokenizer, generation_prompt(x)) for x in batch]
        generated = llm.generate(gen_prompts, sampling_generate, use_tqdm=False)
        drafts = [parse_json(out.outputs[0].text) for out in generated]
        finish_reasons = [str(out.outputs[0].finish_reason) for out in generated]
        rewrites = [compact((x or {}).get("rewrite", "")) for x in drafts]
        valid_indices = [i for i, rewrite in enumerate(rewrites) if rewrite]
        verdicts = [None] * len(batch)
        if valid_indices:
            verify_prompts = [
                render(tokenizer, verification_prompt(batch[i], rewrites[i]))
                for i in valid_indices
            ]
            verified = llm.generate(verify_prompts, sampling_verify, use_tqdm=False)
            for i, out in zip(valid_indices, verified):
                verdicts[i] = parse_json(out.outputs[0].text)
        with partial.open("a", encoding="utf-8") as handle:
            for task, rewrite, verdict, finish_reason in zip(
                batch, rewrites, verdicts, finish_reasons
            ):
                reasons = deterministic_checks(task, rewrite, verdict, finish_reason)
                row = {
                    "parent_query_id": task["parent_query_id"],
                    "gold_skill_ids": task["gold_skill_ids"],
                    "focus_skill_id": task["focus_skill_id"],
                    "rewrite": rewrite,
                    "accepted": not reasons,
                    "rejection_reasons": reasons,
                    "verifier": verdict,
                    "generation_finish_reason": finish_reason,
                }
                handle.write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
                completed[row["parent_query_id"]] = row
            handle.flush()
            os.fsync(handle.fileno())
        done = len(completed)
        elapsed = max(1e-6, time.time() - started)
        atomic_json(
            args.output.with_suffix(args.output.suffix + ".progress.json"),
            {
                "status": "generating",
                "completed": done,
                "total": len(tasks),
                "accepted": sum(x["accepted"] for x in completed.values()),
                "estimated_remaining_seconds": (len(tasks) - done)
                * elapsed
                / max(1, done - (len(tasks) - len(missing))),
                "updated_utc": utc_now(),
            },
        )
    with args.output.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(
                json.dumps(
                    completed[task["parent_query_id"]],
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
    atomic_json(
        args.output.with_suffix(args.output.suffix + ".manifest.json"),
        {
            "complete": True,
            "task_sha256": sha256_file(args.tasks),
            "records_sha256": sha256_file(args.output),
            "count": len(tasks),
            "accepted": sum(x["accepted"] for x in completed.values()),
            "engineering_only": args.engineering_only,
            "generation_model": args.model,
            "enable_thinking": False,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "official_test_used": False,
            "updated_utc": utc_now(),
        },
    )
    print(
        f"complete {len(tasks)} rows, {sum(x['accepted'] for x in completed.values())} accepted"
    )


if __name__ == "__main__":
    main()
