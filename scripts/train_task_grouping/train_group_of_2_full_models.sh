# cb multirc
# rte winogrande_debiased story_cloze hellaswag
# copa wic wsc.fixed boolq

for lr in 1e-5 2e-5 1e-6 
do
# python custom_train_glue_mtl.py --task_names "cb" "multirc" \
#     --model_key "meta-llama/Llama-3.2-1B" \
#     --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 1 --lr $lr \
#     --save_name task_grouping_3 --epochs 10 --write_results --precision "bf16-true" 

python custom_train_glue_mtl.py --task_names "rte" "winogrande_debiased" "story_cloze" "hellaswag" \
    --model_key "meta-llama/Llama-3.2-1B" \
    --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 1 --lr $lr \
    --save_name task_grouping_3 --epochs 10 --write_results --precision "bf16-true" 

# python custom_train_glue_mtl.py --task_names "copa" "wic" "wsc.fixed" "boolq" \
#     --model_key "meta-llama/Llama-3.2-1B" \
#     --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr $lr \
#     --save_name task_grouping_3 --epochs 10 --write_results --precision "bf16-true" 
done


# python custom_train_glue_mtl.py --task_names "cb" "multirc" \
#     --model_key "meta-llama/Llama-3.2-1B" \
#     --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5 \
#     --save_name task_grouping_2 --epochs 10 --write_results --precision "bf16-true" 

# python custom_train_glue_mtl.py --task_names "rte" "copa" "wic" "wsc.fixed" "boolq" "winogrande_debiased" "story_cloze" "hellaswag" \
#     --model_key "meta-llama/Llama-3.2-1B" \
#     --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5 \
#     --save_name task_grouping_2 --epochs 10 --write_results --precision "bf16-true" 