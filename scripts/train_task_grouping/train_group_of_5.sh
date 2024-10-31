# Group = 5
# cb multirc
# rte winogrande_debiased
# copa wic wsc.fixed boolq
# hellaswag
# story_cloze

# python custom_train_glue_mtl.py --task_names "cb" "multirc" \
#     --model_key "meta-llama/Llama-3.2-1B" \
#     --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5 \
#     --train_lora --lora_rank 16 --lora_alpha 128 \
#     --save_name task_grouping_5 --epochs 10 --write_results --precision "bf16-true" 

python custom_train_glue_mtl.py --task_names "rte" "winogrande_debiased" \
    --model_key "meta-llama/Llama-3.2-1B" \
    --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5 \
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name task_grouping_5 --epochs 10 --write_results --precision "bf16-true" 

# python custom_train_glue_mtl.py --task_names "copa" "wic" "wsc.fixed" "boolq" \
#     --model_key "meta-llama/Llama-3.2-1B" \
#     --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5 \
#     --train_lora --lora_rank 16 --lora_alpha 128 \
#     --save_name task_grouping_5 --epochs 10 --write_results --precision "bf16-true" 

python custom_train_glue_mtl.py --task_names "hellaswag" \
    --model_key "meta-llama/Llama-3.2-1B" \
    --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5 \
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name task_grouping_5 --epochs 10 --write_results --precision "bf16-true" 

python custom_train_glue_mtl.py --task_names "story_cloze" \
    --model_key "meta-llama/Llama-3.2-1B" \
    --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5 \
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name task_grouping_5 --epochs 10 --write_results --precision "bf16-true" 

# Group = 7
# cb multirc
# rte
# copa wsc.fixed boolq
# wic
# winogrande_debiased
# story_cloze
# hellaswag


# python custom_train_glue_mtl.py --task_names "cb" "multirc" \
#     --model_key "meta-llama/Llama-3.2-1B" \
#     --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5 \
#     --train_lora --lora_rank 16 --lora_alpha 128 \
#     --save_name task_grouping_7 --epochs 10 --write_results --precision "bf16-true" 

python custom_train_glue_mtl.py --task_names "rte" \
    --model_key "meta-llama/Llama-3.2-1B" \
    --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5 \
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name task_grouping_7 --epochs 10 --write_results --precision "bf16-true" 

python custom_train_glue_mtl.py --task_names "copa" "wsc.fixed" "boolq" \
    --model_key "meta-llama/Llama-3.2-1B" \
    --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5 \
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name task_grouping_7 --epochs 10 --write_results --precision "bf16-true" 

python custom_train_glue_mtl.py --task_names "wic" \
    --model_key "meta-llama/Llama-3.2-1B" \
    --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5 \
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name task_grouping_7 --epochs 10 --write_results --precision "bf16-true" 

python custom_train_glue_mtl.py --task_names "winogrande_debiased" \
    --model_key "meta-llama/Llama-3.2-1B" \
    --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5 \
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name task_grouping_7 --epochs 10 --write_results --precision "bf16-true" 

# python custom_train_glue_mtl.py --task_names "hellaswag" \
#     --model_key "meta-llama/Llama-3.2-1B" \
#     --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5 \
#     --train_lora --lora_rank 16 --lora_alpha 128 \
#     --save_name task_grouping_7 --epochs 10 --write_results --precision "bf16-true" 