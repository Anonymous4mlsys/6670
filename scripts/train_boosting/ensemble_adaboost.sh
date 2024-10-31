python evaluate_merged_model.py --task_names "rte" --model_key "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"\
    --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 1 --epochs 0\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name merging_tinyllama --precision "bf16-true" \
    --merge_model_dirs "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_rte_lora_r_16_pairwise_run_0/epoch_epoch=3.pt" \
    "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_rte_lora_r_16_adaboost_iteration_1_run_0/epoch_epoch=1.pt"\
    --merge_strategy "weighted_ensemble" --merge_weights 3.3141860046725258 1.2256119794294171


python evaluate_merged_model.py --task_names "rte" --model_key "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"\
    --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 1 --epochs 0\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name merging_tinyllama --precision "bf16-true" \
    --merge_model_dirs "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_rte_lora_r_16_pairwise_run_0/epoch_epoch=3.pt" \
    "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_rte_lora_r_16_adaboost_iteration_1_run_0/epoch_epoch=1.pt"\
    "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_rte_lora_r_16_adaboost_iteration_2_run_0/epoch_epoch=1.pt"\
     --merge_strategy "weighted_ensemble" --merge_weights 3.3141860046725258 1.2256119794294171 "-3.1997723543025884"

