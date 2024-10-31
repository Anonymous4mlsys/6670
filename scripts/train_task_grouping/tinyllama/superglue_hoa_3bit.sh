
# ['CB|RTE|COPA|WiC|BoolQ', 'CB|WSC']
# ['CB|RTE|WiC|BoolQ', 'CB|COPA', 'CB|WSC']
# ['CB|RTE|BoolQ', 'CB|COPA', 'CB|WiC', 'CB|WSC']

python custom_train_glue_mtl.py --task_names "cb" "rte" "copa" "wic" "boolq"  \
    --model_key  "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"\
    --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name hoa_4bit --epochs 10 --use_3bit --optimizer "paged_adamw_32bit" --precision "32"

python custom_train_glue_mtl.py --task_names "cb" "rte" "wic" "boolq" \
    --model_key  "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"\
    --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name hoa_4bit --epochs 10 --use_3bit --optimizer "paged_adamw_32bit" --precision "32"
    
python custom_train_glue_mtl.py --task_names "cb" "rte" "boolq" \
    --model_key  "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"\
    --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name hoa_4bit --epochs 10 --use_3bit --optimizer "paged_adamw_32bit" --precision "32"
