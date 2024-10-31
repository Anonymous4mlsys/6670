# python fast_estimate_eval_approximation_alpaca.py \
#     --model_key "EleutherAI/gpt-neo-1.3B" --train_lora --lora_rank 4 --lora_alpha 32 --precision "bf16-true"\
#     --batch_size 4 --max_length 256 --project_gradients --project_dimension 100 \
#     --devices 0 --strategy auto --compute_pretrained_outputs --save_name fast_approximation_gradients \
#     --downsample 400 --num_batches_gradients 1000\
#     # --load_model_dir Alpaca_EleutherAI-gpt-neo-1.3B_lora_r_4_meta_initialization_alpaca_run_0_epoch_epoch_9

# save_name_dir="Alpaca_EleutherAI-gpt-neo-1.3B_lora_r_4_meta_initialization_alpaca_run_0_epoch_epoch_9"

for seed in 0 1 2 3 4
do
python fast_estimate_eval_approximation_alpaca.py \
    --model_key "EleutherAI/gpt-neo-1.3B" --train_lora --lora_rank 4 --lora_alpha 32 --precision "bf16-true"\
    --batch_size 2 --max_length 256 --project_gradients --project_dimension 100 \
    --devices 0 --strategy auto --save_name fast_approximation_gradients \
    --downsample 400 --num_batches_gradients 200\
    --abs_scale 12.393977290799555\
    --scale 0.04 --seed $seed
    # --load_model_dir  $save_name_dir
done

# # python fast_estimate_eval_approximation_alpaca.py \
# #     --model_key "EleutherAI/gpt-neo-125m" --train_lora --lora_rank 4 --lora_alpha 32 --precision "bf16-true"\
# #     --batch_size 2 --max_length 256 --project_gradients --project_dimension 100 \
# #     --devices 0 --strategy auto --save_name fast_approximation_gradients \
# #     --downsample 400 --num_batches_gradients 200 --load_model_dir  $save_name_dir\
# #     --scale 0.1 --seed $seed

# python fast_estimate_eval_approximation_alpaca.py \
#     --model_key "EleutherAI/gpt-neo-125m" --train_lora --lora_rank 4 --lora_alpha 32 --precision "bf16-true"\
#     --batch_size 4 --max_length 256 --project_gradients --project_dimension 100 \
#     --devices 0 --strategy auto --save_name gradients \
#     --downsample 400 --num_batches_gradients 200 --load_model_dir Alpaca_EleutherAI-gpt-neo-125m_lora_r_4_meta_initialization_alpaca_run_0_epoch_epoch_9\
#     --scale 0.08 --seed $seed

# python fast_estimate_eval_approximation_alpaca.py \
#     --model_key "EleutherAI/gpt-neo-125m" --train_lora --lora_rank 4 --lora_alpha 32 --precision "bf16-true"\
#     --batch_size 4 --max_length 256 --project_gradients --project_dimension 100 \
#     --devices 0 --strategy auto --save_name gradients \
#     --downsample 400 --num_batches_gradients 200 --load_model_dir Alpaca_EleutherAI-gpt-neo-125m_lora_r_4_meta_initialization_alpaca_run_0_epoch_epoch_9\
#     --scale 0.06 --seed $seed

# python fast_estimate_eval_approximation_alpaca.py \
#     --model_key "EleutherAI/gpt-neo-125m" --train_lora --lora_rank 4 --lora_alpha 32 --precision "bf16-true"\
#     --batch_size 4 --max_length 256 --project_gradients --project_dimension 100 \
#     --devices 0 --strategy auto --save_name gradients \
#     --downsample 400 --num_batches_gradients 200 --load_model_dir Alpaca_EleutherAI-gpt-neo-125m_lora_r_4_meta_initialization_alpaca_run_0_epoch_epoch_9\
#     --scale 0.04 --seed $seed

# python fast_estimate_eval_approximation_alpaca.py \
#     --model_key "EleutherAI/gpt-neo-125m" --train_lora --lora_rank 4 --lora_alpha 32 --precision "bf16-true"\
#     --batch_size 4 --max_length 256 --project_gradients --project_dimension 100 \
#     --devices 0 --strategy auto --save_name gradients \
#     --downsample 400 --num_batches_gradients 200 --load_model_dir Alpaca_EleutherAI-gpt-neo-125m_lora_r_4_meta_initialization_alpaca_run_0_epoch_epoch_9\
#     --scale 0.02 --seed $seed
# done
# python fast_estimate_eval_approximation_alpaca.py \
#     --model_key "EleutherAI/gpt-neo-125m" --train_lora --lora_rank 4 --lora_alpha 32 --precision "bf16-true"\
#     --batch_size 4 --max_length 256 --project_gradients --project_dimension 100 \
#     --devices 0 --strategy auto --save_name fast_approximation_gradients \
#     --downsample 400 --num_batches_gradients 200 --load_model_dir Alpaca_EleutherAI-gpt-neo-125m_lora_r_4_quantized_meta_initialization_run_0_epoch_epoch_6\
#     --scale 0.1 --seed 3