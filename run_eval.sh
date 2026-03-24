#!/bin/bash

# ==============================
# Define Checkpoint Directory
# ==============================
CKPT_DIR=./checkpoints/LLaVA_covft_exp4_last12L_layernorm
# ==============================
# Cambrain Benchmarks
# ==============================
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/run_benckmark_cambrain.sh --ckpt $CKPT_DIR --benchmark mmvp
echo "✅ Done evaluation on mmvp, by using ${CKPT_DIR} model"

CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/run_benckmark_cambrain.sh --ckpt $CKPT_DIR --benchmark realworldqa
echo "✅ Done evaluation on realworldqa, by using ${CKPT_DIR} model"

CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/run_benckmark_cambrain.sh --ckpt $CKPT_DIR --benchmark coco
echo "✅ Done evaluation on coco, by using ${CKPT_DIR} model"

CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/run_benckmark_cambrain.sh --ckpt $CKPT_DIR --benchmark ade
echo "✅ Done evaluation on ade, by using ${CKPT_DIR} model"

CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/run_benckmark_cambrain.sh --ckpt $CKPT_DIR --benchmark omni
echo "✅ Done evaluation on omni, by using ${CKPT_DIR} model"

CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/run_benckmark_cambrain.sh --ckpt $CKPT_DIR --benchmark mmmu
echo "✅ Done evaluation on mmmu, by using ${CKPT_DIR} model"

CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/run_benckmark_cambrain.sh --ckpt $CKPT_DIR --benchmark ai2d
echo "✅ Done evaluation on ai2d, by using ${CKPT_DIR} model"

# ==============================
# Other Benchmarks
# ==============================
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/gqa.sh --ckpt $CKPT_DIR
echo "✅ Done evaluation on gqa, by using ${CKPT_DIR} model"

CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/mme.sh --ckpt $CKPT_DIR
echo "✅ Done evaluation on mme, by using ${CKPT_DIR} model"

CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/mmbench.sh --ckpt $CKPT_DIR
echo "✅ Done evaluation on mmbench, by using ${CKPT_DIR} model"

CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/mmbench_cn.sh --ckpt $CKPT_DIR
echo "✅ Done evaluation on mmbench_cn, by using ${CKPT_DIR} model"

CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/sqa.sh --ckpt $CKPT_DIR
echo "✅ Done evaluation on sqa, by using ${CKPT_DIR} model"

CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/textvqa.sh --ckpt $CKPT_DIR
echo "✅ Done evaluation on textvqa, by using ${CKPT_DIR} model"
