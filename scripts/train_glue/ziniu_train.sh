task_names=("cb")
#"cb" "rte" "copa" "wic" "wsc.fixed" "boolq" "multirc" "story_cloze" "hellaswag" "winogrande_debiased"
length=${#task_names[@]}
key="meta-llama/Llama-3.2-3B"
# "EleutherAI/gpt-j-6b"
# "meta-llama/Llama-3.2-1B"
# "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
device=0
batch_s=4


for ((j = 0; j < $length; j++)); do
  python custom_train_glue_mtl_seq_bn.py --task_names "${task_names[$j]}" \
      --model_key  $key\
      --devices $device --batch_size $batch_s --inference_batch_size 8 --max_length 256 --runs 1 --lr 1e-4\
      --reduction_factor 128\
      --save_name pairwise_adapter --epochs 10 --write_results\
      --train_adapter  --precision "bf16-true" 
done

# --use_adapter

  # python custom_train_glue_mtl.py --task_names  "${task_names[$i]}" "${task_names[$j]}" \
  #     --model_key  $key\
  #     --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 128 --runs 3 --lr 5e-5\
  #     --train_lora --lora_rank 16 --lora_alpha 128 \
  #     --save_name pairwise_4bit --epochs 10 --use_qlora --optimizer "paged_adamw_32bit" --precision "bf16-true" 

#   python custom_train_glue_mtl.py --task_names "${task_names[$j]}" \
#       --model_key  $key\
#       --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5\
#       --train_lora --lora_rank 16 --lora_alpha 128 \
#       --save_name pairwise_3bit --epochs 10 --use_3bit --optimizer "paged_adamw_32bit" --precision "32"
