task_names=("cb" "multirc" "rte" "winogrande_debiased" "story_cloze" "hellaswag"  "copa" "wic" "wsc.fixed" "boolq") 
length=${#task_names[@]}

python custom_train_glue_mtl.py --task_name "cb" "multirc" "rte" "winogrande_debiased" "story_cloze" "hellaswag"  "copa" "wic" "wsc.fixed" "boolq"\
    --downsample_ratio 1e-3 --minimum_samples 500 --minimum_samples_validation 100\
    --model_key "meta-llama/Llama-3.2-1B"\
    --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 2e-5\
    --save_name downsampled_mtl_full_model --epochs 10 --write_results

for ((j = 0; j < $length; j++)); do
  python custom_train_glue_mtl.py --task_name "${task_names[$j]}" \
      --downsample_ratio 1e-3 --minimum_samples 500 --minimum_samples_validation 100\
      --model_key "meta-llama/Llama-3.2-1B"\
      --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 2e-5\
      --save_name downsampled_single_task_full_model --epochs 10 --write_results
done

for ((j = 0; j < $length; j++)); do
for ((i = j+1; i < $length; i++)); do
  python custom_train_glue_mtl.py --task_name "${task_names[$j]}" "${task_names[$i]}" \
      --downsample_ratio 1e-3 --minimum_samples 500 --minimum_samples_validation 100\
      --model_key "meta-llama/Llama-3.2-1B"\
      --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 2e-5\
      --save_name downsampled_pairwise_full_model --epochs 10 --write_results
done
done

#   --train_lora --lora_rank 16 --lora_alpha 128 \
