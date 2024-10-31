# cb multirc
# rte winogrande_debiased story_cloze
# copa wic wsc.fixed boolq
# hellaswag

# python custom_train_glue_mtl.py --task_names "cb" "multirc" \
#     --model_key "meta-llama/Llama-3.1-8B" \
#     --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5 \
#     --train_lora --lora_rank 16 --lora_alpha 128 \
#     --save_name task_grouping_4 --epochs 10 --write_results --precision "32" 

CUDA_VISIBLE_DEVICES=1 python custom_train_glue_mtl.py --task_names "rte" "winogrande_debiased" "story_cloze" \
    --model_key "meta-llama/Llama-3.1-8B" \
    --devices 0 --batch_size 4 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5 \
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name task_grouping_4_3bit --epochs 10 --write_results --precision "32" \
    --use_3bit --optimizer "paged_adamw_32bit"

CUDA_VISIBLE_DEVICES=1 python custom_train_glue_mtl.py --task_names "copa" "wic" "wsc.fixed" "boolq" \
    --model_key "meta-llama/Llama-3.1-8B" \
    --devices 0 --batch_size 4 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5 \
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name task_grouping_4_3bit --epochs 10 --write_results --precision "32" \
    --use_3bit --optimizer "paged_adamw_32bit"

CUDA_VISIBLE_DEVICES=1 python custom_train_glue_mtl.py --task_names "hellaswag" \
    --model_key "meta-llama/Llama-3.1-8B" \
    --devices 0 --batch_size 4 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5 \
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name task_grouping_4_3bit --epochs 10 --write_results --precision "32" \
    --use_3bit --optimizer "paged_adamw_32bit"

# cb multirc
# rte winogrande_debiased story_cloze hellaswag
# copa wic wsc.fixed boolq

# python custom_train_glue_mtl.py --task_names "cb" "multirc" \
#     --model_key "meta-llama/Llama-3.1-8B" \
#     --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5 \
#     --train_lora --lora_rank 16 --lora_alpha 128 \
#     --save_name task_grouping_3 --epochs 10 --write_results --precision "32" \
#     --use_3bit --optimizer "paged_adamw_32bit"

CUDA_VISIBLE_DEVICES=1 python custom_train_glue_mtl.py --task_names "rte" "winogrande_debiased" "story_cloze" "hellaswag" \
    --model_key "meta-llama/Llama-3.1-8B" \
    --devices 0 --batch_size 4 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5 \
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name task_grouping_3_3bit --epochs 10 --write_results --precision "32" \
    --use_3bit --optimizer "paged_adamw_32bit"

# python custom_train_glue_mtl.py --task_names "copa" "wic" "wsc.fixed" "boolq" \
#     --model_key "meta-llama/Llama-3.1-8B" \
#     --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5 \
#     --train_lora --lora_rank 16 --lora_alpha 128 \
#     --save_name task_grouping_3 --epochs 10 --write_results --precision "32" \
#     --use_3bit --optimizer "paged_adamw_32bit"
