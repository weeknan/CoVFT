#!/bin/bash

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --ckpt) CKPT_DIR="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$CKPT_DIR" ]; then
    echo "❌ Error: --ckpt not provided"
    exit 1
fi

CKPT_NAME=$(basename $CKPT_DIR)
SPLIT="mmbench_dev_cn_20231003"

python -m llava.eval.model_vqa_mmbench \
    --model-path $CKPT_DIR \
    --question-file ./playground/data/eval/mmbench_cn/$SPLIT.tsv \
    --answers-file ./playground/data/eval/mmbench_cn/answers/$SPLIT/${CKPT_NAME}.jsonl \
    --lang cn \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p playground/data/eval/mmbench_cn/answers_upload/$SPLIT
python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench_cn/$SPLIT.tsv \
    --result-dir ./playground/data/eval/mmbench_cn/answers/$SPLIT \
    --upload-dir ./playground/data/eval/mmbench_cn/answers_upload/$SPLIT \
    --experiment ${CKPT_NAME}

echo "✅ Done evaluation on mmbench_cn, by using ${CKPT_NAME} model"
