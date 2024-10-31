python fast_estimate_eval_approximation_alpaca.py \
    --model_key flan_t5_base --train_lora --lora_rank 4 --lora_alpha 32\
    --batch_size 4 --project_gradients --project_dimension 100\
    --scale 0.1 --seed 0 --compute_pretrained_outputs --precision "bf16-true" --devices 2 --strategy auto\
    --load_model_dir Alpaca_flan_t5_base_lora_r_4_meta_initialization_alpaca_run_0_epoch_3\
    --save_name fast_approximation --downsample 400 --num_batches_gradients 100

for seed in 0 1 2 3 4
do
python fast_estimate_eval_approximation_alpaca.py \
    --model_key flan_t5_base --train_lora --lora_rank 4 --lora_alpha 32 --precision "bf16-true"\
    --batch_size 4 --project_gradients --project_dimension 100\
    --scale 0.10 --seed $seed --devices 2 --strategy auto\
    --load_model_dir Alpaca_flan_t5_base_lora_r_4_meta_initialization_alpaca_run_0_epoch_3\
    --save_name fast_approximation  --downsample 400 --num_batches_gradients 100

python fast_estimate_eval_approximation_alpaca.py \
    --model_key flan_t5_base --train_lora --lora_rank 4 --lora_alpha 32 --precision "bf16-true"\
    --batch_size 4 --project_gradients --project_dimension 100\
    --scale 0.08 --seed $seed --devices 2 --strategy auto\
    --load_model_dir Alpaca_flan_t5_base_lora_r_4_meta_initialization_alpaca_run_0_epoch_3\
    --save_name fast_approximation  --downsample 400 --num_batches_gradients 100

python fast_estimate_eval_approximation_alpaca.py \
    --model_key flan_t5_base --train_lora --lora_rank 4 --lora_alpha 32 --precision "bf16-true"\
    --batch_size 4 --project_gradients --project_dimension 100\
    --scale 0.06 --seed $seed --devices 2 --strategy auto\
    --load_model_dir Alpaca_flan_t5_base_lora_r_4_meta_initialization_alpaca_run_0_epoch_3\
    --save_name fast_approximation  --downsample 400 --num_batches_gradients 100

python fast_estimate_eval_approximation_alpaca.py \
    --model_key flan_t5_base --train_lora --lora_rank 4 --lora_alpha 32 --precision "bf16-true"\
    --batch_size 4 --project_gradients --project_dimension 100\
    --scale 0.04 --seed $seed --devices 2 --strategy auto\
    --load_model_dir Alpaca_flan_t5_base_lora_r_4_meta_initialization_alpaca_run_0_epoch_3\
    --save_name fast_approximation  --downsample 400 --num_batches_gradients 100

python fast_estimate_eval_approximation_alpaca.py \
    --model_key flan_t5_base --train_lora --lora_rank 4 --lora_alpha 32 --precision "bf16-true"\
    --batch_size 4 --project_gradients --project_dimension 100\
    --scale 0.02 --seed $seed --devices 2 --strategy auto\
    --load_model_dir Alpaca_flan_t5_base_lora_r_4_meta_initialization_alpaca_run_0_epoch_3\
    --save_name fast_approximation  --downsample 400 --num_batches_gradients 100
done
