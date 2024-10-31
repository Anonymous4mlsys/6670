# Group = 2
# cb multirc
# rte copa wic wsc.fixed boolq winogrande_debiased story_cloze hellaswag

# Gourp = 3
# cb multirc
# rte winogrande_debiased story_cloze hellaswag
# copa wic wsc.fixed boolq

# Group = 4 
# cb multirc
# rte winogrande_debiased story_cloze
# copa wic wsc.fixed boolq
# hellaswag

# Group = 5
# cb multirc
# rte winogrande_debiased
# copa wic wsc.fixed boolq
# hellaswag
# story_cloze

# Group = 6
# cb multirc
# rte winogrande_debiased
# copa wsc.fixed boolq
# wic
# hellaswag
# story_cloze

# Group = 7
# cb multirc
# rte
# copa wsc.fixed boolq
# wic
# winogrande_debiased
# story_cloze
# hellaswag

# Group = 8
# cb multirc
# rte
# copa boolq
# wic
# wsc.fixed
# winogrande_debiased
# story_cloze
# hellaswag

python custom_train_glue_mtl.py --task_names "rte" "winogrande_debiased" "story_cloze" "hellaswag" \
    --model_key "meta-llama/Llama-3.1-8B" \
    --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5 \
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name task_grouping_3 --epochs 10 --write_results --precision "bf16-true" \
    --use_qlora --optimizer "paged_adamw_32bit"