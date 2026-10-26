"""Recompute retrieval from the released embeddings, without model weights."""

import argparse, gzip
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings", type=Path, default=Path("results/embeddings.npz"))
    p.add_argument("--output", type=Path, default=Path("work/from_vectors.tsv.gz"))
    p.add_argument("--top-k", type=int, default=50)
    a = p.parse_args()
    z = np.load(a.embeddings, allow_pickle=False)
    q, d = z["query_embeddings"], z["skill_embeddings"]
    qi, di = z["query_ids"], z["skill_ids"]
    # Do not renormalize: the published arrays retain the evaluator's exact
    # FP16-query/FP32-document score convention.
    a.output.parent.mkdir(parents=True, exist_ok=True)
    count = min(a.top_k, len(d))
    with gzip.open(a.output, "wt") as f:
        for at in range(0, len(q), 128):
            s = q[at : at + 128] @ d.T
            ix = np.argpartition(-s, count - 1, axis=1)[:, :count]
            values = np.take_along_axis(s, ix, axis=1)
            order = np.argsort(-values, axis=1, kind="stable")
            ix = np.take_along_axis(ix, order, axis=1)
            values = np.take_along_axis(values, order, axis=1)
            for j, (indices, scores) in enumerate(zip(ix, values)):
                for k, (index, score) in enumerate(zip(indices, scores), 1):
                    f.write(
                        f"{qi[at + j]}\tQ0\t{di[index]}\t{k}\t{score:.9g}\tCoverSkill\n"
                    )
    print(a.output)


if __name__ == "__main__":
    main()
