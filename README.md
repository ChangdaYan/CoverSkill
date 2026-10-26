# CoverSkill

**CoverSkill: Coverage-Aware Alignment and Facet-Aligned Targeted Refinement for Agent Skill Retrieval**

Minimal implementation of the two-stage, single-vector retriever. This repository contains data preparation and generation, the main training pipeline, inference/evaluation, and the final model's test rankings and embeddings. It does not contain model weights, SkillRet data, generated training data, analysis or ablation experiments.

## Reported model

The released predictions use the final **492-update MGCA → 300-update FATR** model and a frozen four-facet mean skill index, on the **original SkillRet official test**: 4,997 queries and 6,660 candidate skills.

| Metric (%) | @5 | @10 | @15 |
| --- | ---: | ---: | ---: |
| NDCG | 82.26 | **84.34** | 85.09 |
| Recall | 84.86 | **90.18** | 92.49 |
| Completeness | 73.26 | **81.53** | 85.49 |

All numbers are single-run point estimates. Completeness is the fraction of queries whose entire gold set is retrieved in the top K.

## Evaluate the released results

Run the commands below from this repository's root. Obtain the original dataset from [ThakiCloud/SKILLRET](https://huggingface.co/datasets/ThakiCloud/SKILLRET). The download command pins revision `7cae7cfbad2b0e1ebc9170892f568993aae543b0`; the dataset is downloaded locally and is not redistributed here.

```bash
pip install huggingface-hub numpy
python -m coverskill.prepare download --output data/skillret --only-qrels
python -m coverskill.evaluate \
  --rankings results/rankings.tsv.gz \
  --qrels data/skillret/qrels/test.jsonl
```

`results/rankings.tsv.gz` is the formal run's Top-50 ranking for **every** test query (249,850 rows), in TREC format:

```text
query_id  Q0  skill_id  rank  score  CoverSkill
```

`results/embeddings.npz` contains only IDs and numeric arrays, loadable with `numpy.load(..., allow_pickle=False)`:

| Array | Shape | Meaning |
| --- | --- | --- |
| `query_ids` | (4997,) | Row IDs of the test queries |
| `skill_ids` | (6660,) | Row IDs of the test candidate skills |
| `query_embeddings` | (4997, 1024) | Final FATR query vectors |
| `skill_embeddings` | (6660, 1024) | Frozen MGCA four-facet mean skill vectors |

There are no query texts, skill documents or relevance labels in these files. The skill vectors come from the **Stage-1** encoder; the query vectors come from the **Stage-2** encoder. Query vectors preserve the formal evaluator's FP16 rounding, stored in FP32 arrays. Skill vectors are FP32-normalized from the frozen FP16 index. Rank using their dot product **without normalizing the supplied arrays again**.

To recompute retrieval on a CPU, without any model checkpoint:

```bash
python -m coverskill.search_vectors --output work/from_vectors.tsv.gz
python -m coverskill.evaluate --rankings work/from_vectors.tsv.gz
```

This reproduces the reported @10 metrics. Very close score ties can be ordered differently across CPU BLAS and CUDA implementations; the supplied ranking file is the formal reported run.

## Reproduce training

Use Linux, Python 3.11 or newer, and four CUDA GPUs. The reference training environment is recorded in `requirements.txt`; the reference runs used Python 3.13 and four A100 40 GB GPUs. No GPU scheduler or experiment search is included.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export CUDA_VISIBLE_DEVICES=0,1,2,3
export COVERSKILL_DATA="$PWD/data/skillret"
export COVERSKILL_WORK="$PWD/work"
export COVERSKILL_BASE_MODEL=Qwen/Qwen3-Embedding-0.6B
bash scripts/reproduce.sh data
```

An unmodified local snapshot can be supplied through `COVERSKILL_BASE_MODEL`. Stage 1 must start from the original [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), rather than any SkillRet-trained checkpoint.

If the pinned SkillRet snapshot has already been downloaded, set `SKILLRET_SOURCE=/path/to/snapshot/data` before the `data` command to prepare it locally without network access. The standalone equivalent is `python -m coverskill.prepare download --source /path/to/snapshot/data --output data/skillret` (add `--only-qrels` when only evaluating the released rankings).

### 1. Generate train-only supervision

Run offline generation in a **separate environment** with `requirements-generation.txt` and an unmodified [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B):

```bash
# In the generation environment, from this repository's root:
pip install -r requirements-generation.txt
export GENERATOR_MODEL=Qwen/Qwen3-32B
bash scripts/reproduce.sh generate
```

The pipeline generates one query-supported subintent per official training query/gold pair. It keeps raw attempts, validates exact evidence spans, and rejects unsupported or skill-name-leaking outputs. Complete query rewrites emphasize a weakly expressed requirement while preserving the original requirements and gold set; a separate LLM verification and deterministic checks filter them. Raw and rejected generations remain in the local work directory.

The original run retained 86,831 subintents and 12,349 complete rewrites. The release regenerates all training slots directly with the pretrained generator and does not require historical partitions, trained retriever caches, or hidden checkpoints. LLM outputs and downstream retraining can vary across runtime versions and hardware; the supplied test artifacts preserve the reported run.

### 2. Train MGCA

Return to the training environment:

```bash
bash scripts/reproduce.sh mgca
```

Multi-Granularity Coverage Alignment jointly updates a shared query/skill encoder. Original full queries and accepted full rewrites are the primary inputs; aligned subintents provide auxiliary alignment. Individual Identity, Procedure, Interface and Constraints views supply skill-side supervision. The objective combines primary contrastive alignment, subintent alignment, group coverage and weakest-gold coverage. Batches follow the original query-unique round-robin order.

### 3. Build the frozen skill space and train FATR

```bash
bash scripts/reproduce.sh fatr
```

Facet-Aligned Targeted Refinement uses only its own MGCA parent. Each training skill is represented by the normalized mean of four separately encoded, normalized facets. This frozen training matrix is used for hard-query mining, teacher comparisons and the student retrieval objective.

The parent selects 8,192 hard training queries, stratified as 2,048/4,096/2,048 one-/two-/three-gold requests. Training mixes these with ordinary training queries 1:1. The student emphasizes its current Top-96 non-gold negatives by a factor of two. A frozen parent teacher blends full-query and strongest subintent similarities (0.8/0.2). Local gold–non-gold supervision retains up to 12 pairs per query whose teacher margin improves by more than 0.005 and whose parent margin is at most 0.05. The teacher does not have to rank a retained pair correctly. Subintent alignment remains an auxiliary objective; only the query encoder is updated.

| Setting | MGCA | FATR |
| --- | ---: | ---: |
| Final update | 492 | 300 |
| Linear LR schedule horizon | 600 | 450 |
| Warmup updates | 60 | 45 |
| Peak learning rate | 2e-5 | 8e-6 |
| Global batch | 384 | 192 |
| Gradient-cache microbatch per GPU | 24 | 12 |
| Query token limit | 4096 | 4096 |
| Training skill input | Individual views, up to 4096 tokens | Frozen four-facet mean, 512 tokens/facet |
| Subintent alignment weight | 0.08 | 0.08 |
| Group / weakest-gold weights | 0.08 / 0.16 | — |
| Local pair-loss weight | — | 0.10 |
| Retrieval / pair temperature | 0.05 | 0.05 |
| Weakest-gold temperature | 0.15 | — |
| Seed | 43 | 43 |

Stopping and LR horizons are separate: the run ends at update 492/300 while preserving the 600/450-update schedule prefix. The local auxiliary weight ramps from zero to full strength between 5% and 20% of the Stage-2 **schedule horizon**. The final checkpoint is used; optimizer state is never saved. Local input provenance records and parent checks are generated when training runs; these runtime files are not part of this repository.

### 4. Build the test index and evaluate

```bash
bash scripts/reproduce.sh evaluate
```

The test skill index is built independently from test skill documents, using the same four-facet construction and frozen MGCA document encoder. The final FATR encoder encodes each full query once. No test subintents, query rewrites or relevance labels are used during encoding or retrieval. Evaluation reads the official test qrels after predictions are saved.

## Files

```text
coverskill/           Main-method preparation, generation, training and retrieval
configs/              The two training configurations
scripts/reproduce.sh  Reproduction commands
results/              Formal test rankings and embeddings only
```

The training implementation uses PyTorch, Hugging Face Transformers/Datasets, Sentence Transformers and Accelerate; generation uses vLLM. The custom cached contrastive objective builds on Sentence Transformers' CachedMultipleNegativesRankingLoss. These dependencies are installed separately and retain their own licenses. SkillRet data and pretrained model weights retain their original terms. This repository's code is released under the [MIT License](LICENSE).
