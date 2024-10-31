# python fast_estimate_linear_regression_eval_instruction.py --train_instruction --target_task truthfulqa --model_key TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
#     --train_lora --lora_rank 128 --lora_alpha 512 --devices 1\
#     --run 0 --project_dimension 100 --scale 1\
#     --gradient_dir "Instruction_TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_lora_r_128_dim_100_run_0"\
#     --save_name "sample_truthfulqa_scale_1.0" --number_of_subsets 100 --subset_size 0.5

python fast_estimate_linear_regression_eval_instruction.py --train_instruction --target_task truthfulqa --model_key TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
    --train_lora --lora_rank 128 --lora_alpha 512 --devices 0\
    --run 0 --project_dimension 100 --scale 2\
    --gradient_dir "Instruction_TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_lora_r_128_dim_100_run_0"\
    --save_name "sample_truthfulqa_scale_2.0" --number_of_subsets 100 --subset_size 0.5

python fast_estimate_linear_regression_eval_instruction.py --train_instruction --target_task truthfulqa --model_key TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
    --train_lora --lora_rank 128 --lora_alpha 512 --devices 0\
    --run 0 --project_dimension 100 --scale 0.2\
    --gradient_dir "Instruction_TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_lora_r_128_dim_100_run_0"\
    --save_name "sample_truthfulqa_scale_0.2" --number_of_subsets 100 --subset_size 0.5