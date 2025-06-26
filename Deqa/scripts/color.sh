#!/bin/bash
export PYTHONPATH=./:$PYTHONPATH

LOAD="/home/lrz/Deqa/model_weight"

deepspeed --include localhost:0,1 --master_port 6688 /home/lrz/Deqa/src/train/train_mem.py \
    --deepspeed /home/lrz/Deqa/scripts/zero3.json \
    --lora_enable True \
    --model_name_or_path $LOAD \
    --version v1 \
    --dataset_type pair \
    --level_prefix "The quality of the image is" \
    --level_names excellent good fair poor bad \
    --softkl_loss True \
    --weight_rank 1.0 \
    --weight_softkl 1.0 \
    --weight_next_token 0.05 \
    --continuous_rating_loss True \
    --closeset_rating_loss True \
    --use_fix_std True \
    --detach_pred_std True \
    --data_paths /home/lrz/Deqa/data/merged_fidelity.json \
    --data_weights 1 1 1 \
    --image_folder /home/lrz/Q-Align-main/playground/DIQA-5000_phase1/merge \
    --output_dir /home/lrz/Deqa/output/model/deqa_lora_color_merge \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --tune_visual_abstractor True \
    --freeze_vision_model False \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --report_to tensorboard
