# python custom_train.py --dataset_key strategy_qa --model_key "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T" --train_key ft_cot --preset_key ft_cot_t70_64aug\
#     --devices 2 --batch_size 8 --inference_batch_size 32 --runs 1\
#     --train_lora --lora_rank 16 --lora_alpha 128 \
#     --save_name meta_train_new --precision "bf16-true" --epochs 5 # --use_qlora --optimizer "paged_adamw_32bit" 

python sample_train_results.py --dataset_key strategy_qa --model_key "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T" --preset_key ft_cot_t70_64aug\
    --train_lora --lora_rank 16 --lora_alpha 128 --epochs 5\
    --project_dim 200 --device 1 \
    --load_sample_task_dir strategy_qa_flan_t5_base_ft_cot_t70_64aug_run_0_scale_0.4_project_100_subset_size_0.5_clusters_100\
    --load_clusters --num_clusters 100 --scale 0.4 --save_name sampled_true_performance