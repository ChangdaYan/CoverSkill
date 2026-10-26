from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from .modeling import similarity_scores


def masked_target_ce(
    logits: torch.Tensor, target: int, sibling_targets: list[int]
) -> torch.Tensor:
    masked = logits.clone()
    for sibling in sibling_targets:
        if sibling != target:
            masked[sibling] = -torch.inf
    return F.cross_entropy(masked[None], torch.tensor([target], device=logits.device))


def dynamic_offsets(
    anchor_logits: torch.Tensor,
    gold: list[list[int]],
    count: int,
    multiplier: float,
) -> tuple[torch.Tensor, list[list[int]]]:
    masked = anchor_logits.detach().clone()
    for owner, targets in enumerate(gold):
        masked[owner, targets] = -torch.inf
    quota = min(count, anchor_logits.shape[1] - max(map(len, gold)))
    selected = torch.topk(masked, k=quota, dim=-1).indices
    offsets = torch.zeros_like(anchor_logits)
    offsets.scatter_(1, selected, math.log(multiplier))
    return offsets, selected.cpu().tolist()


def multigold_retrieval_loss(
    logits: torch.Tensor, gold: list[list[int]], offsets: torch.Tensor
) -> torch.Tensor:
    values = []
    for owner, targets in enumerate(gold):
        terms = torch.stack(
            [
                masked_target_ce(logits[owner] + offsets[owner], target, targets)
                for target in targets
            ]
        )
        values.append(terms.mean())
    return torch.stack(values).mean()


def explicit_intent_loss(
    intent_embeddings: torch.Tensor,
    intent_owners: list[int],
    aligned: list[list[tuple[int, float]]],
    gold: list[list[int]],
    documents: torch.Tensor,
    offsets: torch.Tensor,
    temperature: float,
    anchor: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    if not intent_owners:
        return anchor.sum() * 0, 0
    logits = intent_embeddings @ documents.T / temperature
    per_query: dict[int, list[torch.Tensor]] = {}
    for branch, (owner, targets) in enumerate(zip(intent_owners, aligned)):
        if not targets:
            continue
        weights = logits.new_tensor([weight for _, weight in targets])
        terms = torch.stack(
            [
                masked_target_ce(logits[branch] + offsets[owner], target, gold[owner])
                for target, _ in targets
            ]
        )
        value = (terms * weights / weights.sum().clamp_min(1e-12)).sum()
        per_query.setdefault(owner, []).append(value)
    if not per_query:
        return anchor.sum() * 0, 0
    return torch.stack(
        [torch.stack(values).mean() for values in per_query.values()]
    ).mean(), len(per_query)


def per_gold_effect_loss(
    student_scores: torch.Tensor,
    teacher_rows: list[dict],
    gold: list[list[int]],
    temperature: float,
    max_pairs: int,
) -> tuple[torch.Tensor, int, int]:
    """Use only teacher-improved gold/confuser margins, including rejected queries.

    The cache contains parent and explicit scores on the union of both Top-50
    sets and all gold skills. This is not a query-level teacher-win gate: a
    gold may gain even when another gold makes the aggregate teacher worse.
    """
    values = []
    pairs_total = 0
    for owner, row in enumerate(teacher_rows):
        if row is None or row.get("effect_indices") is None:
            continue
        indices = row["effect_indices"].tolist()
        positions = {int(value): index for index, value in enumerate(indices)}
        teacher = row["effect_scores"].float().tolist()
        parent = row["effect_anchor_scores"].float().tolist()
        positive_set = set(gold[owner])
        pairs = []
        for positive in positive_set:
            if positive not in positions:
                continue
            p = positions[positive]
            for negative in indices:
                if negative in positive_set:
                    continue
                n = positions[negative]
                teacher_margin = teacher[p] - teacher[n]
                parent_margin = parent[p] - parent[n]
                improvement = teacher_margin - parent_margin
                if improvement <= 0.005 or parent_margin > 0.05:
                    continue
                # Prefer corrections around the current ranking boundary, not
                # easy negatives or arbitrary large score changes.
                priority = min(0.20, improvement) - max(0.0, -parent_margin) * 0.01
                pairs.append(
                    (
                        priority,
                        int(positive),
                        int(negative),
                        min(2.0, improvement / 0.05),
                    )
                )
        if not pairs:
            continue
        chosen = sorted(pairs, reverse=True)[:max_pairs]
        positives = torch.tensor(
            [p for _, p, _, _ in chosen], device=student_scores.device
        )
        negatives = torch.tensor(
            [n for _, _, n, _ in chosen], device=student_scores.device
        )
        weights = student_scores.new_tensor([w for _, _, _, w in chosen])
        errors = F.softplus(
            (student_scores[owner, negatives] - student_scores[owner, positives])
            / temperature
        )
        values.append((errors * weights / weights.mean().clamp_min(1e-6)).mean())
        pairs_total += len(chosen)
    zero = student_scores.sum() * 0
    return (
        (torch.stack(values).mean(), len(values), pairs_total)
        if values
        else (zero, 0, 0)
    )


def compute_objective(
    states,
    intent_embeddings,
    intent_owners,
    aligned,
    gold,
    documents,
    recipe,
    teacher_rows=None,
    per_query_aligned=None,
    training_progress=1.0,
    base_documents=None,
):
    temperature = float(recipe["retrieval_temperature"])
    scores = similarity_scores(states, documents)
    offsets, selected = dynamic_offsets(
        scores / temperature,
        gold,
        int(recipe["dynamic_negative_count"]),
        float(recipe["dynamic_negative_multiplier"]),
    )
    retrieval = multigold_retrieval_loss(scores / temperature, gold, offsets)
    intent, intent_queries = explicit_intent_loss(
        intent_embeddings,
        intent_owners,
        aligned,
        gold,
        documents,
        offsets,
        temperature,
        states,
    )
    local, queries, pairs = per_gold_effect_loss(
        scores,
        teacher_rows or [None] * len(gold),
        gold,
        float(recipe["per_gold_effect_temperature"]),
        int(recipe["per_gold_effect_max_pairs"]),
    )
    begin, end = float(recipe["aux_ramp_begin"]), float(recipe["aux_ramp_end"])
    ramp = min(1.0, max(0.0, (training_progress - begin) / max(1e-6, end - begin)))
    total = (
        retrieval
        + float(recipe["intent_weight"]) * intent
        + ramp * float(recipe["per_gold_effect_weight"]) * local
    )
    return total, {
        "total_loss": float(total.detach()),
        "retrieval_loss": float(retrieval.detach()),
        "intent_loss": float(intent.detach()),
        "local_loss": float(local.detach()),
        "queries_with_intent_loss": intent_queries,
        "queries_with_local_loss": queries,
        "selected_local_pairs": pairs,
        "auxiliary_ramp": ramp,
        "emphasized_negatives_per_query": sum(map(len, selected)) / len(selected),
    }
