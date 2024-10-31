# for model in "meta-llama/Llama-3.2-1B" # "meta-llama/Llama-3.2-3B" "meta-llama/Llama-3.1-8B"
# do
# for num_models in 1 2 3 4 5 6 7 8 
# do
# # python measure_memory.py --task_names "cb" --model_key $model\
# #     --devices 0 --runs 1 --epochs 0\
# #     --save_name single_task --precision "bf16-true" \
# #     --batch_size 1 --inference_batch_size 1 --max_length 1  --num_ensembled_models $num_models

# python measure_memory.py --task_names "cb" --model_key $model\
#     --devices 0 --runs 1 --epochs 0\
#     --save_name single_task --precision "bf16-true" \
#     --batch_size 1 --inference_batch_size 1 --max_length 512  --num_ensembled_models $num_models
# done
# done


# for model in "meta-llama/Llama-3.2-3B" "meta-llama/Llama-3.1-8B" # "meta-llama/Llama-3.2-1B" 
# do
# for num_models in 1 2 3 4 5 6 7 8 
# do
# # python measure_memory.py --task_names "cb" --model_key $model\
# #     --devices 0 --runs 1 --epochs 0\
# #     --save_name single_task --precision "bf16-true" \
# #     --batch_size 1 --inference_batch_size 1 --max_length 1  --num_ensembled_models $num_models\
    
# python measure_memory.py --task_names "cb" --model_key $model\
#     --devices 0 --runs 1 --epochs 0\
#     --save_name single_task --precision "bf16-true" \
#     --batch_size 1 --inference_batch_size 1 --max_length 512  --num_ensembled_models $num_models\
#     --train_lora --lora_rank 16 --lora_alpha 128 --merge_strategy "adapter_ensemble"
# # done
# # done

for model in "meta-llama/CodeLlama-34b-Instruct-hf" # "meta-llama/Llama-3.2-3B" "meta-llama/Llama-3.1-8B" # "meta-llama/Llama-3.2-1B" 
do
for num_models in 1 2 3 4
do
# python measure_memory.py --task_names "cb" --model_key $model\
#     --devices 0 --runs 1 --epochs 0\
#     --save_name single_task --precision "bf16-true" \
#     --batch_size 1 --inference_batch_size 1 --max_length 1  --num_ensembled_models $num_models \
#     --train_lora --lora_rank 16 --lora_alpha 128 --merge_strategy "adapter_ensemble"\
#     --use_qlora --optimizer "paged_adamw_8bit"
    
python measure_memory.py --task_names "cb" --model_key $model\
    --devices 1 --runs 1 --epochs 0\
    --save_name single_task --precision "bf16-true" \
    --batch_size 1 --inference_batch_size 1 --max_length 512  --num_ensembled_models $num_models\
    --train_lora --lora_rank 16 --lora_alpha 128 --merge_strategy "adapter_ensemble"\
    --use_qlora --optimizer "paged_adamw_8bit"
done
done


# for model in "meta-llama/Llama-3.2-1B" "meta-llama/Llama-3.2-3B" "meta-llama/Llama-3.1-8B" # 
# do
# for num_models in 1 2 3 4
# do
# python measure_memory.py --task_names "cb" --model_key $model\
#     --devices 0 --runs 1 --epochs 0\
#     --save_name single_task --precision "bf16-true" \
#     --batch_size 1 --inference_batch_size 1 --max_length 512  --num_ensembled_models $num_models\
#     --train_adapter --reduction_factor 128 --merge_strategy "adapter_ensemble"

# python measure_memory.py --task_names "cb" --model_key $model\
#     --devices 0 --runs 1 --epochs 0\
#     --save_name single_task --precision "bf16-true" \
#     --batch_size 1 --inference_batch_size 1 --max_length 512  --num_ensembled_models $num_models\
#     --train_lora --lora_rank 16 --lora_alpha 128 --merge_strategy "adapter_ensemble"\
#     --train_adapter --use_qadapter --reduction_factor 128 --optimizer "paged_adamw_8bit"
# done
# done

# python measure_memory.py --task_names "cb" --model_key "meta-llama/Llama-3.2-3B"\
#     --devices 0 --runs 1 --epochs 0\
#     --save_name single_task --precision "bf16-true" \
#     --batch_size 1 --inference_batch_size 1 --max_length 512  --num_ensembled_models 2\
#     --train_adapter --reduction_factor 128 --merge_strategy "adapter_ensemble"

# for model in "meta-llama/Llama-3.2-1B" # "meta-llama/Llama-3.2-3B" "meta-llama/Llama-3.1-8B"
# do
# for num_models in 1
# do
# python measure_memory.py --task_names "cb" --model_key $model\
#     --devices 0 --runs 1 --epochs 0\
#     --save_name single_task --precision "bf16-true" \
#     --batch_size 1 --inference_batch_size 1 --max_length 1  --num_ensembled_models $num_models \
#     --train_lora --lora_rank 16 --lora_alpha 128 --merge_strategy "adapter_ensemble"\
#     --use_3bit --optimizer "paged_adamw_8bit"
    
# python measure_memory.py --task_names "cb" --model_key $model\
#     --devices 0 --runs 1 --epochs 0\
#     --save_name single_task --precision "bf16-true" \
#     --batch_size 1 --inference_batch_size 1 --max_length 512  --num_ensembled_models $num_models\
#     --train_lora --lora_rank 16 --lora_alpha 128 --merge_strategy "adapter_ensemble"\
#     --use_3bit --optimizer "paged_adamw_8bit"
# done
# done