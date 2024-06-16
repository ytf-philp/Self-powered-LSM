#!/bin/bash


if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <data_path> <model_path> <save_path> "
    exit 1
fi
DATA=$1
MODEL=$2
SAVE_PATH=$3

# 执行Python脚本
python -m torch.distributed.run  --nproc_per_node=4 \
    --master_port=21572 \
    /self-powered/scripts/train_stage0_evaluate_asr.py \
    --data $DATA \
    --save_path $SAVE_PATH \
    --overwrite_output_dir \
    --output_dir $MODEL+"1" \
    --whisper_model $MODEL \
    --llama_model $MODEL \
    --Blsp_model $MODEL \
    --safetensor True \
    --remove_unused_columns False \
    --seed 1 \
    --do_train True \
    --bf16 True \
    --learning_rate 2e-5 \
    --weight_decay 0.05 \
    --max_grad_norm 1.0 \
    --warmup_steps 100 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 3 \
    --disable_tqdm True \
    --logging_steps 10 \
    --save_steps 500 \
    --save_total_limit 4
