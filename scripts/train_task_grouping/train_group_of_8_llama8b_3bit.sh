# Group = 8
# cb multirc
# rte
# copa boolq
# wic
# wsc.fixed
# winogrande_debiased
# story_cloze
# hellaswag

# python custom_train_glue_mtl.py --task_names "cb" "multirc" \
#     --model_key "meta-llama/Llama-3.1-8B" \
#     --devices 5 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5 \
#     --train_lora --lora_rank 16 --lora_alpha 128 \
#     --save_name task_grouping_8 --epochs 10 --write_results --precision "bf16-true" 

python custom_train_glue_mtl.py --task_names "rte" \
    --model_key "meta-llama/Llama-3.1-8B" \
    --devices 5 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5 \
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name task_grouping_8_3bit --epochs 10 --write_results --precision "bf16-true" \
    --use_3bit --optimizer "paged_adamw_32bit"

python custom_train_glue_mtl.py --task_names "copa" "boolq" \
    --model_key "meta-llama/Llama-3.1-8B" \
    --devices 5 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5 \
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name task_grouping_8_3bit --epochs 10 --write_results --precision "bf16-true" \
    --use_3bit --optimizer "paged_adamw_32bit"

python custom_train_glue_mtl.py --task_names "wic" \
    --model_key "meta-llama/Llama-3.1-8B" \
    --devices 5 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5 \
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name task_grouping_8_3bit --epochs 10 --write_results --precision "bf16-true" \
    --use_3bit --optimizer "paged_adamw_32bit"

python custom_train_glue_mtl.py --task_names "winogrande_debiased" \
    --model_key "meta-llama/Llama-3.1-8B" \
    --devices 5 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5 \
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name task_grouping_8_3bit --epochs 10 --write_results --precision "bf16-true" \
    --use_3bit --optimizer "paged_adamw_32bit"

python custom_train_glue_mtl.py --task_names "hellaswag" \
    --model_key "meta-llama/Llama-3.1-8B" \
    --devices 5 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5 \
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name task_grouping_8_3bit --epochs 10 --write_results --precision "bf16-true" \
    --use_3bit --optimizer "paged_adamw_32bit"