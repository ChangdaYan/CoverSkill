#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export COVERSKILL_WORK="${COVERSKILL_WORK:-$PWD/work}"
export COVERSKILL_DATA="${COVERSKILL_DATA:-$PWD/data/skillret}"
export COVERSKILL_BASE_MODEL="${COVERSKILL_BASE_MODEL:-Qwen/Qwen3-Embedding-0.6B}"
export TOKENIZERS_PARALLELISM=false
phase="${1:-help}"
case "$phase" in
  data)
    python -m coverskill.prepare download --output "$COVERSKILL_DATA"
    python -m coverskill.prepare tokenize --data "$COVERSKILL_DATA" --model "$COVERSKILL_BASE_MODEL" --output "$COVERSKILL_WORK/full_query_tokens.pt"
    ;;
  generate)
    python -m coverskill.generate_subintents --output "$COVERSKILL_WORK/subintents.pt" --tokenizer "$COVERSKILL_BASE_MODEL" --model "${GENERATOR_MODEL:-Qwen/Qwen3-32B}" --tensor-parallel-size 4
    python -m coverskill.prepare_rewrites --source-cache "$COVERSKILL_WORK/subintents.pt" --output "$COVERSKILL_WORK/rewrite_tasks.jsonl.gz"
    python -m coverskill.generate_rewrites --tasks "$COVERSKILL_WORK/rewrite_tasks.jsonl.gz" --output "$COVERSKILL_WORK/rewrites_raw.jsonl" --model "${GENERATOR_MODEL:-Qwen/Qwen3-32B}" --tensor-parallel-size 4
    python -m coverskill.curate_rewrites --source "$COVERSKILL_WORK/rewrites_raw.jsonl" --output "$COVERSKILL_WORK/rewrites.jsonl"
    ;;
  stage1)
    torchrun --standalone --nproc_per_node=4 -m coverskill.train_stage1 --base-model "$COVERSKILL_BASE_MODEL" --data "$COVERSKILL_DATA" --work "$COVERSKILL_WORK" --output "$COVERSKILL_WORK/stage1"
    ;;
  refine)
    parent="$COVERSKILL_WORK/stage1/final"
    torchrun --standalone --nproc_per_node=4 -m coverskill.build_index --model "$parent" --split train --output "$COVERSKILL_WORK/train_documents.pt"
    torchrun --standalone --nproc_per_node=4 -m coverskill.mine_queries --model "$parent" --document-cache "$COVERSKILL_WORK/train_documents.pt" --document-manifest "$COVERSKILL_WORK/train_documents.manifest.json" --output "$COVERSKILL_WORK/hard_queries.jsonl.gz" --selection-id FIXED_FINAL_F2_OWN_HARD8192_v1 --micro-batch 12
    torchrun --standalone --nproc_per_node=4 -m coverskill.build_teacher --all-train --teacher-model "$parent" --document-cache "$COVERSKILL_WORK/train_documents.pt" --decomposition "$COVERSKILL_WORK/subintents.pt" --output "$COVERSKILL_WORK/teacher.pt" --effect-depth 50 --micro-batch 12
    torchrun --standalone --nproc_per_node=4 -m coverskill.train_stage2 --parent "$parent"
    ;;
  evaluate)
    torchrun --standalone --nproc_per_node=4 -m coverskill.build_index --model "$COVERSKILL_WORK/stage1/final" --split test --batch-skills 4 --output "$COVERSKILL_WORK/test_documents.pt"
    torchrun --standalone --nproc_per_node=4 -m coverskill.retrieve --model "$COVERSKILL_WORK/runs/FASR/seed43/final" --index "$COVERSKILL_WORK/test_documents.pt" --output "$COVERSKILL_WORK/predictions"
    python -m coverskill.evaluate --rankings "$COVERSKILL_WORK/predictions/rankings.tsv.gz" --qrels "$COVERSKILL_DATA/qrels/test.jsonl"
    ;;
  *) printf 'Usage: bash scripts/reproduce.sh {data|generate|stage1|refine|evaluate}\n'; exit 2 ;;
esac
