python custom_train_alpaca.py --model_key flan_t5_base \
    --lr 5e-5 --batch_size 16 --max_length 256 --epochs 10\
    --train_lora --lora_rank 4 --lora_alpha 32\
    --strategy auto --devices 1 --runs 1 --precision "bf16-true" --accumulate 1 --save_name "meta_initialization_alpaca"
