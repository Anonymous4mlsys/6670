python evaluate_merged_model.py --task_names "cb" "rte" --model_key "meta-llama/Llama-3.1-8B"\
    --devices 0 --batch_size 1 --inference_batch_size 1 --max_length 512 --runs 1 --epochs 0\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name single_task --precision "bf16-true" \
    --merge_model_dirs "cb_lora_r_16_single_task_run_0/epoch_epoch=1.pt" "rte_lora_r_16_single_task_run_0/epoch_epoch=0.pt"\
    --merge_strategy "ties"
