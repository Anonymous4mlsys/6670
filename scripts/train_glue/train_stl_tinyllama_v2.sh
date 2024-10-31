task_names=("anli_r1" "anli_r2" "anli_r3" "hellaswag") # "cb" "rte" "copa" "wic" "wsc.fixed"
length=${#task_names[@]}


for ((j = 0; j < $length; j++)); do
  python custom_train_glue_mtl_adapter.py --task_names "${task_names[$j]}" \
      --model_key  "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"\
      --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5\
      --train_lora --lora_rank 16 --lora_alpha 128 \
      --save_name cb_pairwise --epochs 10 --write_results

# python custom_train_glue_mtl.py --task_names "${task_names[$j]}" \
#     --model_key  "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"\
#     --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 3 --lr 5e-5\
#     --train_lora --lora_rank 16 --lora_alpha 128 \
#     --save_name pairwise_4bit --epochs 10 --use_qlora --optimizer "paged_adamw_32bit" --precision "bf16-true" 

#   python custom_train_glue_mtl.py --task_names "${task_names[$j]}" \
#       --model_key  "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"\
#       --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5\
#       --train_lora --lora_rank 16 --lora_alpha 128 \
#       --save_name pairwise_3bit --epochs 10 --use_3bit --optimizer "paged_adamw_32bit" --precision "32"
done
