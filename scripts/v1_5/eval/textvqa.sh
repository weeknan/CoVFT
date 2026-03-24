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

python -m llava.eval.model_vqa_loader \
    --model-path $CKPT_DIR \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./playground/data/eval/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/${CKPT_NAME}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/${CKPT_NAME}.jsonl

echo "✅ Done evaluation on textvqa, by using ${CKPT_NAME} model"
