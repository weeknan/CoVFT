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

python -m llava.eval.model_vqa_science \
    --model-path $CKPT_DIR \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/${CKPT_NAME}.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/${CKPT_NAME}.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/${CKPT_NAME}_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/${CKPT_NAME}_result.json

echo "✅ Done evaluation on sqa, by using ${CKPT_NAME} model"
