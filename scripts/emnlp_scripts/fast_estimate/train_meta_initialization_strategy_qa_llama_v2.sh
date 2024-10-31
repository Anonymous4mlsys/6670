for rank in 64 128 32
do
for lr in 3e-4
do
python custom_train.py --dataset_key strategy_qa --model_key "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T" --train_key ft_cot --preset_key ft_cot_t70_64aug\
    --devices 1 --batch_size 8 --inference_batch_size 8 --runs 1 --lr $lr\
    --train_lora --lora_rank $rank --lora_alpha $((rank*8)) \
    --save_name meta_train_new --epochs 10 --precision "bf16-true"
done
done

# python custom_train.py --dataset_key strategy_qa --model_key "../llama/llama-3/Meta-Llama-3-8B-hf" --train_key ft_cot --preset_key ft_cot_t70_64aug\
#     --devices 2 --batch_size 8 --inference_batch_size 8 --runs 1\
#     --train_lora --lora_rank 16 --lora_alpha 128 \
#     --save_name meta_train --epochs 5 --use_qlora --optimizer "paged_adamw_32bit" --precision "bf16-true" --save_every_epoch

# python custom_train.py --dataset_key strategy_qa --model_key "EleutherAI/gpt-neo-1.3B"  --train_key ft_cot --preset_key ft_cot_t70_64aug\
#     --devices 2 --batch_size 8 --inference_batch_size 8 --runs 1\
#     --train_lora --lora_rank 16 --lora_alpha 128 \
#     --save_name meta_train --epochs 5 --use_qlora --optimizer "paged_adamw_32bit" --precision "bf16-true" --save_every_epoch
    

