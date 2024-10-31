# compute gradients

save_name_dir="Alpaca_EleutherAI-gpt-neo-1.3B_lora_r_4_quantized_meta_initialization_run_0_epoch_epoch_6"
# python fast_estimate_eval_approximation_alpaca.py \
#     --model_key "EleutherAI/gpt-neo-1.3B" --train_lora --lora_rank 4 --lora_alpha 32 --precision "bf16-true"\
#     --batch_size 2 --max_length 256 --project_gradients --project_dimension 100 \
#     --devices 0 --strategy auto --compute_pretrained_outputs --save_name fast_approximation_quantized_gradients \
#     --downsample 400 --num_batches_gradients 1000 \
#     --use_qlora
    # --load_model_dir $save_name_dir\

# evaluate
for seed in 0 1 2 3 4
do
python fast_estimate_eval_approximation_alpaca.py \
    --model_key "EleutherAI/gpt-neo-1.3B" --train_lora --lora_rank 4 --lora_alpha 32 --precision "bf16-true"\
    --batch_size 2 --max_length 256 --project_gradients --project_dimension 100 \
    --devices 0 --strategy auto --save_name fast_approximation_gradients \
    --downsample 400 --num_batches_gradients 200 \
    --scale 0.06 --seed $seed --use_qlora\
    --abs_scale 52.324992881479155
    # --load_model_dir  $save_name_dir
done