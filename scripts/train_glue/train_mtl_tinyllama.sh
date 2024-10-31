python custom_train_glue_mtl.py --task_names "cb" "rte" "copa" "wic" "wsc.fixed" "boolq" "multirc" "winogrande_debiased" "story_cloze" "hellaswag" "anli_r1" "anli_r2" "anli_r3" \
    --model_key  "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"\
    --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name mtl --epochs 10 --write_results --precision "bf16-true" 

python custom_train_glue_mtl.py --task_names "cb" "rte" "copa" "wic" "wsc.fixed" "boolq" "multirc" "winogrande_debiased" "story_cloze" "hellaswag" "anli_r1" "anli_r2" "anli_r3" \
    --model_key  "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"\
    --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 3 --lr 5e-5\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name mtl_4bit --epochs 10 --use_qlora --optimizer "paged_adamw_32bit" --precision "bf16-true" 
