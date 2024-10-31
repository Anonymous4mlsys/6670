# ["CB", "RTE", "COPA", "WiC", "wsc.fixed", "BoolQ"]

# 0 1 2 4
# 3 5

# 0 5 "CB" "BoolQ"
# 1 2 4  "RTE", "COPA", "WSC.fixed"
# 3 "WiC"

# 0 "CB"
# 1 2 4 "RTE", "COPA", "WSC.fixed"
# 3 "WiC"
# 5 "BoolQ"

python custom_train_glue_mtl.py --task_names "rte" "copa" "wsc.fixed" \
    --model_key  "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"\
    --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name boost_4bit --epochs 10 --use_qlora --optimizer "paged_adamw_32bit" --precision "bf16-true" 

python custom_train_glue_mtl.py --task_names "cb" "rte" "copa" "wsc.fixed" \
    --model_key  "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"\
    --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name boost_4bit --epochs 10 --use_qlora --optimizer "paged_adamw_32bit" --precision "bf16-true" 

# python custom_train_glue_mtl.py --task_names "wic" "boolq" \
#     --model_key  "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"\
#     --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5\
#     --train_lora --lora_rank 16 --lora_alpha 128 \
#     --save_name boost_4bit --epochs 10 --use_qlora --optimizer "paged_adamw_32bit" --precision "bf16-true" 

# python custom_train_glue_mtl.py --task_names "cb" "boolq" \
#     --model_key  "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"\
#     --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5\
#     --train_lora --lora_rank 16 --lora_alpha 128 \
#     --save_name boost_4bit --epochs 10 --use_qlora --optimizer "paged_adamw_32bit" --precision "bf16-true" 

# python custom_train_glue_mtl.py --task_names "wic" \
#     --model_key  "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"\
#     --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5\
#     --train_lora --lora_rank 16 --lora_alpha 128 \
#     --save_name boost_4bit --epochs 10 --use_qlora --optimizer "paged_adamw_32bit" --precision "bf16-true" 