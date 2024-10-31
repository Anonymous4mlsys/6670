for task_name in "cb" "rte" "copa" "multirc"
do
python custom_train_glue_mtl.py --task_name $task_name --model_key "meta-llama/Llama-3.1-8B"\
    --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 512 --runs 2 --lr 5e-5\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name single_task --epochs 10 --precision "bf16-true" --use_qlora --optimizer "paged_adamw_32bit" --accumulate 2

# python custom_train_glue.py --task_name $task_name --model_key "meta-llama/Llama-3.1-8B"\
#     --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 128 --runs 3 --lr 5e-5\
#     --train_lora --lora_rank 16 --lora_alpha 128 \
#     --save_name single_task_4bit --epochs 10 --use_qlora --optimizer "paged_adamw_32bit" --precision "bf16-true" 

# python custom_train_glue.py --task_name $task_name --model_key "meta-llama/Llama-3.1-8B"\
#     --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 128 --runs 3 --lr 5e-5\
#     --train_lora --lora_rank 16 --lora_alpha 128 \
#     --save_name single_task_3bit --epochs 10 --use_3bit --optimizer "paged_adamw_32bit" --precision "32"

# python custom_train_glue.py --task_name $task_name --model_key "meta-llama/Llama-3.1-8B"\
#     --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 128 --runs 3 --lr 5e-5\
#     --train_lora --lora_rank 16 --lora_alpha 128 \
#     --save_name single_task_2bit --epochs 10 --use_2bit --optimizer "paged_adamw_32bit" --precision "32"
done