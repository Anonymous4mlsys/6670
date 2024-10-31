# cb multirc
# rte
# copa boolq
# wic
# wsc.fixed
# winogrande_debiased
# story_cloze
# hellaswag
# python custom_train_glue_mtl.py --task_names "cb" \
#     --model_key "meta-llama/Llama-3.2-1B" \
#     --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5 \
#     --train_lora --lora_rank 16 --lora_alpha 128 \
#     --save_name task_grouping_8 --epochs 10 --write_results --precision "bf16-true" 

# python custom_train_glue_mtl.py --task_names "multirc" \
#     --model_key "meta-llama/Llama-3.2-1B" \
#     --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5 \
#     --train_lora --lora_rank 16 --lora_alpha 128 \
#     --save_name task_grouping_8 --epochs 10 --write_results --precision "bf16-true" 

python custom_train_glue_mtl.py --task_names "boolq" \
    --model_key "meta-llama/Llama-3.2-1B" \
    --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5 \
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name task_grouping_8 --epochs 10 --write_results --precision "bf16-true" 

python custom_train_glue_mtl.py --task_names "copa" \
    --model_key "meta-llama/Llama-3.2-1B" \
    --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5 \
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name task_grouping_8 --epochs 10 --write_results --precision "bf16-true" 