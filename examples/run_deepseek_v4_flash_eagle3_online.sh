#!/usr/bin/env bash
#
# Eagle3 ONLINE training for deepseek-ai/DeepSeek-V4-Flash.
#
# Cloned from run_deepseek_v3_671b_eagle3_online.sh and adapted for V4-Flash:
#   - Target model: DeepSeek-V4-Flash (284B/13B FP8/FP4 mixed; 1M context)
#   - Draft config: configs/deepseek-v4-flash-eagle3.json (1-layer Llama,
#     hidden_size=4096, draft_vocab_size=32000, aux_layer_ids=[1, 21, 41])
#   - chat-template: deepseek-v4 (custom; V4 has NO Jinja chat_template
#     in tokenizer_config.json — uses Python encoding/encoding_dsv4.py).
#     The chat template registration in our sglang fork is TODO(phase1) per
#     experiments/DeepSeek-V4-Flash/architecture-notes.md "Open risks #10".
#     If it's not yet registered when this script runs, training will fail
#     loudly — fix is to add the template (see deepseek_v4 entry in our
#     sglang fork's chat_template registry).
#
# Critical Eagle3 rules applied (CLAUDE.md rules #1, #2, #5):
#   - --target-model-backend sglang (rule #1; never hf)
#   - Default sglang attention backend is FlashInfer (rule #2). For V4
#     specifically, the per-V4 triton override applies INSIDE our sglang
#     fork's V4Attention path (sparse_attn_v4 -> NSA tilelang). This script
#     doesn't pass --sglang-attention-backend explicitly because the V4
#     model file does the routing internally.
#   - Phase 1 fork-extension is required: sglang fork must have
#     deepseek_v4.py with full V4Attention + load_weights bodies before
#     this script can succeed.
#
# Hardware: 2x H100:8 (TP=16) preferred per architecture-notes.md;
#           H200:8 single-node fallback after 60-min PENDING (rule #18).
# FP8/FP4: V4 ships native FP8 non-experts + FP4 experts. Set
#          SGLANG_ENABLE_JIT_DEEPGEMM=0 (rule from
#          benchmarks/gpu/docs/eagle3_learnings.md) to avoid JIT issues
#          with FP8 path.
#
# Usage:
#   bash run_deepseek_v4_flash_eagle3_online.sh [NUM_GPUS] [TP_SIZE]
# Defaults: NUM_GPUS=8, TP_SIZE=8 (single-node H100); for multi-node
# 2xH100, the SkyPilot recipe (tw-taas/examples/recipes/eagle3-deepseek-
# v4-flash/task-aws-h100-2node.yaml) sets NUM_GPUS=8 and TP_SIZE=16 via
# torchrun multi-node.

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

NUM_GPUS=${1:-8}
TP_SIZE=${2:-8}
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-64}

# FP8 / FP4 path — disable DeepGEMM JIT to avoid FP8 path issues on V4.
export SGLANG_ENABLE_JIT_DEEPGEMM=${SGLANG_ENABLE_JIT_DEEPGEMM:-0}

# Train eagle3 online.
torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path deepseek-ai/DeepSeek-V4-Flash \
    --draft-model-config $ROOT_DIR/configs/deepseek-v4-flash-eagle3.json \
    --train-data-path $ROOT_DIR/cache/dataset/perfect-blend.jsonl \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --output-dir $ROOT_DIR/outputs/deepseek-v4-flash-eagle3-perfect-blend-online \
    --tp-size $TP_SIZE \
    --target-model-backend sglang \
    --num-epochs 6 \
    --batch-size 1 \
    --learning-rate 5e-5 \
    --max-length 4096 \
    --chat-template deepseek-v4 \
    --cache-dir $ROOT_DIR/cache \
    --dist-timeout 60 \
    --sglang-mem-fraction-static 0.80
