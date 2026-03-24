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
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/${CKPT_NAME}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd ./playground/data/eval/MME
python convert_answer_to_mme.py --experiment ${CKPT_NAME}

cd eval_tool
python calculation.py --results_dir answers/${CKPT_NAME}

echo "✅ Done evaluation on mme, by using ${CKPT_NAME} model"
