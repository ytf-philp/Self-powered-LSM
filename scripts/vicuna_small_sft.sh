
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
python -m torch.distributed.run --nproc_per_node=8 /self-powered/train_stage0_sft.py \
    --deepspeed /self-powered/config/dp_config_zero1.json \
    --data /data/process/processed_sft \
    --output_dir /self-powered/model/new_output/whisper_small_vicuna_nolora_sft_new \
    --logging_dir /self-powered/tensorboard/whisper_small_vicuna_nolora \
    --Blsp_model /self-powered/model/new_output/whisper_small_nolora_2epoch \
    --llama_model /self-powered/model/new_output/whisper_small_nolora_2epoch \
    --whisper_model /self-powered/model/new_output/whisper_small_nolora_2epoch \
    --remove_unused_columns False \
    --seed 1 \
    --do_train True \
    --bf16  True \
    --learning_rate 2e-5 \
    --weight_decay 0.05 \
    --max_grad_norm 1.0 \
    --warmup_steps 100 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 2 \
    --disable_tqdm True \
    --logging_steps 10 \
    --save_steps 100 \
    --save_total_limit 4
