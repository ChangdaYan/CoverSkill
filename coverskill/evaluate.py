"""Evaluate saved rankings using qrels obtained separately from SkillRet."""

import argparse, gzip, json, math
from pathlib import Path


def read_rankings(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    rows = {}
    with opener(path, "rt") as f:
        for line in f:
            qid, _, sid, rank, score, _ = line.split()
            rows.setdefault(qid, []).append((int(rank), sid, float(score)))
    return {q: sorted(v) for q, v in rows.items()}


def evaluate(rankings, qrels, cutoffs=(5, 10, 15)):
    gold = {}
    with Path(qrels).open() as f:
        for line in f:
            r = json.loads(line)
            if int(r.get("relevance", 1)) > 0:
                gold.setdefault(str(r["query_id"]), set()).add(str(r["skill_id"]))
    if set(rankings) != set(gold):
        raise ValueError("Rankings must contain every and only official test query.")
    out = {}
    for k in cutoffs:
        ndcg = recall = complete = 0.0
        for q, g in gold.items():
            order = [sid for _, sid, _ in rankings[q][:k]]
            if len(order) != len(set(order)):
                raise ValueError("Duplicate skill in a ranking.")
            n = sum(s in g for s in order)
            dcg = sum((s in g) / math.log2(i + 2) for i, s in enumerate(order))
            ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(g))))
            ndcg += dcg / ideal
            recall += n / len(g)
            complete += n == len(g)
        out[str(k)] = {
            "NDCG": ndcg / len(gold),
            "Recall": recall / len(gold),
            "Completeness": complete / len(gold),
        }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rankings", type=Path, default=Path("results/rankings.tsv.gz"))
    p.add_argument("--qrels", type=Path, default=Path("data/skillret/qrels/test.jsonl"))
    a = p.parse_args()
    print(json.dumps(evaluate(read_rankings(a.rankings), a.qrels), indent=2))


if __name__ == "__main__":
    main()
