#!/usr/bin/env bash
# Eagle3 online training for mistralai/Mistral-Medium-3.5-128B (in-VM, 8×H100 80GB, TP=8).
#
# Plan: docs/plans/mistral-medium-3-5-eagle3.md (in the umbrella repo).
# Workspace: experiments/Mistral-Medium-3.5/ (in the umbrella repo).
#
# Notes
# -----
# - Target is dense 128B Mistral3 (text_config.model_type=ministral3): 88 layers,
#   hidden 12288, GQA 96:8, vocab 131072. Native FP8 published quant.
# - Mistral-Medium-3.5's stock config.json publishes
#   architectures: ["Mistral3ForConditionalGeneration"] (the multimodal wrapper).
#   Override it to ["Ministral3ForCausalLM"] on disk before launching, so
#   sglang routes the load to the text-only head and the Pixtral vision
#   tower never instantiates. See architecture-notes.md §"Top-level config".
# - SGLANG_ENABLE_JIT_DEEPGEMM=0 set as a precaution (FP8-MoE precedent on
#   GLM-4.7-FP8; Mistral-Medium-3.5 is FP8-dense — flag is harmless if unused).
# - Chat template "mistral-medium-3-5" registered in
#   specforge/data/template.py: [INST]user[/INST]assistant</s>.

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$SCRIPT_DIR")

NUM_GPUS=${1:-8}
TP_SIZE=${2:-8}
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-64}

# Persistent-disk paths (in-vm-only.md). Override TRAIN_DATA / OUTPUT_DIR via env if needed.
TARGET_PATH=${TARGET_PATH:-/mnt/persistent/models/Mistral-Medium-3.5-128B}
TRAIN_DATA=${TRAIN_DATA:-/mnt/persistent/training-data/mixed_54k.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/persistent/checkpoints/Mistral-Medium-3.5-Eagle3-exp-a}
CACHE_DIR=${CACHE_DIR:-/mnt/persistent/.cache/mistral-medium-3-5-train}

mkdir -p "$OUTPUT_DIR" "$CACHE_DIR"

# FP8 published target: keep DeepGEMM JIT off as a precaution.
export SGLANG_ENABLE_JIT_DEEPGEMM=0

echo "[run] target=$TARGET_PATH"
echo "[run] draft-config=$ROOT_DIR/configs/mistral-medium-3-5-eagle3.json"
echo "[run] data=$TRAIN_DATA"
echo "[run] output=$OUTPUT_DIR"
echo "[run] tp=$TP_SIZE  num-gpus=$NUM_GPUS"

torchrun \
    --standalone \
    --nproc_per_node "$NUM_GPUS" \
    "$ROOT_DIR/scripts/train_eagle3.py" \
    --target-model-path "$TARGET_PATH" \
    --draft-model-config "$ROOT_DIR/configs/mistral-medium-3-5-eagle3.json" \
    --train-data-path "$TRAIN_DATA" \
    --build-dataset-num-proc "$BUILD_DATASET_NUM_PROC" \
    --output-dir "$OUTPUT_DIR" \
    --tp-size "$TP_SIZE" \
    --target-model-backend sglang \
    --num-epochs 6 \
    --batch-size 1 \
    --learning-rate 5e-5 \
    --max-length 4096 \
    --chat-template mistral-medium-3-5 \
    --embedding-key model.language_model.embed_tokens.weight \
    --cache-dir "$CACHE_DIR" \
    --dist-timeout 60 \
    --sglang-mem-fraction-static 0.40
