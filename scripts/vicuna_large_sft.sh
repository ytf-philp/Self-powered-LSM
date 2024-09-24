
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python -m torch.distributed.run --nproc_per_node=4 self-powered/src/train_stage0.py \
    --deepspeed self-powered/config/dp_config_zero1.json \
    --data self-powered/data/process/processed_final \
    --output_dir self-powered/model/new_output/whisper_large_final_new \
    --logging_dir self-powered//tensorboard/whisper_large_stage1_final_new \
    --Blsp_model self-powered/model/whisper_large_7B \
    --llama_model self-powered/model/whisper_large_7B \
    --whisper_model self-powered/model/whisper_large_7B \
    --remove_unused_columns False \
    --seed 1 \
    --do_train True \
    --bf16  True \
    --learning_rate 2e-5 \
    --weight_decay 0.05 \
    --max_grad_norm 1.0 \
    --warmup_steps 100 \
    --per_device_train_batch_size 12 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 1 \
    --disable_tqdm True \
    --logging_steps 10 \
    --save_steps 500 \
    --save_total_limit 4


