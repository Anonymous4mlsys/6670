# cb multirc
# rte winogrande_debiased story_cloze hellaswag
# copa wic wsc.fixed boolq

python custom_train_glue_mtl.py --task_names "cb" "multirc" "rte" "winogrande_debiased" "story_cloze" "hellaswag"  "copa" "wic" "wsc.fixed" "boolq"\
    --model_key "meta-llama/Llama-3.2-1B" \
    --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5 \
    --save_name task_grouping_1 --epochs 10 --write_results --precision "bf16-true" 

    # --train_lora --lora_rank 16 --lora_alpha 128 \