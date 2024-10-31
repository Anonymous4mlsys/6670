python fast_estimate_compute_gradients.py --dataset_key strategy_qa --model_key "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T" --preset_key ft_cot_t70_64aug\
    --load_model_dir TinyLlama_TinyLlama-1.1B-intermediate-step-1431k-3T_strategy_qa_ft_cot_t70_64aug_lora_r_16_meta_train_epoch_4\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --run 0 --project_dim 100 --device 0


python fast_estimate_linear_regression.py --dataset_key strategy_qa --model_key "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T" --preset_key ft_cot_t70_64aug\
    --load_model_dir TinyLlama_TinyLlama-1.1B-intermediate-step-1431k-3T_strategy_qa_ft_cot_t70_64aug_lora_r_16_meta_train_epoch_4\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --run 0 --project_dim 100 --device 0 \
    --load_clusters --num_clusters 100 --number_of_subsets 50 --scale 0.1\
    --load_sample_task_dir strategy_qa_flan_t5_base_ft_cot_t70_64aug_run_0_scale_0.4_project_100_subset_size_0.5_clusters_100
