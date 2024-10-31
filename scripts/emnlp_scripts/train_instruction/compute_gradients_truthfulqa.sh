python fast_estimate_eval_approximation_alpaca.py --load_truthfulqa \
    --model_key "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T" --train_lora --lora_rank 128 --lora_alpha 512 --precision "bf16-true"\
    --batch_size 8 --max_length 256 --project_gradients --project_dimension 100 \
    --devices 0 1 2 --strategy auto --compute_pretrained_outputs --save_name truthful_qa --num_batches_gradients 12000\
    --load_model_dir Instruction__TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_lora_r_128_meta_initialization_run_0

python fast_estimate_eval_approximation_alpaca.py --load_toxigen \
    --model_key "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T" --train_lora --lora_rank 128 --lora_alpha 512 --precision "bf16-true"\
    --batch_size 8 --max_length 256 --project_gradients --project_dimension 100 \
    --devices 0 1 2 --strategy auto --compute_pretrained_outputs --save_name toxigen --num_batches_gradients 12000\
    --load_model_dir Instruction__TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_lora_r_128_meta_initialization_run_0
