# CUDA_VISIBLE_DEVICES=0 python -m eval.truthfulqa.run_eval \
#         --data_dir data/eval/truthfulqa \
#         --save_dir results/trutufulqa/pretrained_TinyLLama \
#         --model_name_or_path TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T\
#         --tokenizer_name_or_path TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T\
#         --metrics mc \
#         --preset qa \
#         --hf_truth_model_name_or_path allenai/truthfulqa-truth-judge-llama2-7B \
#         --hf_info_model_name_or_path allenai/truthfulqa-info-judge-llama2-7B \
#         --eval_batch_size 20 \
#         --load_in_8bit

# for epoch in 0 1 2 3 4 5 6 7 8 9
# do
# CUDA_VISIBLE_DEVICES=2 python -m eval.truthfulqa.run_eval \
#         --data_dir data/eval/truthfulqa \
#         --save_dir "results/truthfulqa/TinyLlama_selected_0.05_epoch_${epoch}" \
#         --model_name_or_path TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T\
#         --tokenizer_name_or_path TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T\
#         --metrics mc \
#         --preset qa \
#         --hf_truth_model_name_or_path allenai/truthfulqa-truth-judge-llama2-7B \
#         --hf_info_model_name_or_path allenai/truthfulqa-info-judge-llama2-7B \
#         --eval_batch_size 20 \
#         --load_in_8bit\
#         --adapter_path "external_lightning_logs/Instruction__TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_lora_r_128_selected_0.05_run_0/epoch_epoch=${epoch}"\
#         --lora_rank 128 --lora_alpha 512

# CUDA_VISIBLE_DEVICES=2 python -m eval.truthfulqa.run_eval \
#         --data_dir data/eval/truthfulqa \
#         --save_dir "results/truthfulqa/TinyLlama_selected_0.10_epoch_${epoch}" \
#         --model_name_or_path TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T\
#         --tokenizer_name_or_path TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T\
#         --metrics mc \
#         --preset qa \
#         --hf_truth_model_name_or_path allenai/truthfulqa-truth-judge-llama2-7B \
#         --hf_info_model_name_or_path allenai/truthfulqa-info-judge-llama2-7B \
#         --eval_batch_size 20 \
#         --load_in_8bit\
#         --adapter_path "external_lightning_logs/Instruction__TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_lora_r_128_selected_0.10_run_0/epoch_epoch=${epoch}"\
#         --lora_rank 128 --lora_alpha 512

# CUDA_VISIBLE_DEVICES=2 python -m eval.truthfulqa.run_eval \
#         --data_dir data/eval/truthfulqa \
#         --save_dir "results/truthfulqa/TinyLlama_selected_0.15_epoch_${epoch}" \
#         --model_name_or_path TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T\
#         --tokenizer_name_or_path TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T\
#         --metrics mc \
#         --preset qa \
#         --hf_truth_model_name_or_path allenai/truthfulqa-truth-judge-llama2-7B \
#         --hf_info_model_name_or_path allenai/truthfulqa-info-judge-llama2-7B \
#         --eval_batch_size 20 \
#         --load_in_8bit\
#         --adapter_path "external_lightning_logs/Instruction__TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_lora_r_128_selected_0.15_run_0/epoch_epoch=${epoch}"\
#         --lora_rank 128 --lora_alpha 512
# done

for epoch in 0 1 2 3 4 5 6 7 8 9
do
# CUDA_VISIBLE_DEVICES=2 python -m eval.toxigen.run_eval \
#     --data_dir data/eval/toxigen/ \
#     --save_dir "results/toxigen/TinyLlama_selected_0.05_epoch_${epoch}" \
#     --model_name_or_path TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T\
#     --tokenizer_name_or_path TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T\
#     --load_in_8bit\
#     --eval_batch_size 256\
#     --adapter_path "external_lightning_logs/Instruction__TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_lora_r_128_selected_0.05_run_0/epoch_epoch=${epoch}"\
#     --lora_rank 128 --lora_alpha 512

CUDA_VISIBLE_DEVICES=0 python -m eval.toxigen.run_eval \
    --data_dir data/eval/toxigen/ \
    --save_dir "results/toxigen/TinyLlama_selected_0.10_epoch_${epoch}" \
    --model_name_or_path TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T\
    --tokenizer_name_or_path TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T\
    --load_in_8bit\
    --eval_batch_size 256\
    --adapter_path "external_lightning_logs/Instruction__TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_lora_r_128_selected_0.10_run_0/epoch_epoch=${epoch}"\
    --lora_rank 128 --lora_alpha 512

CUDA_VISIBLE_DEVICES=0 python -m eval.toxigen.run_eval \
    --data_dir data/eval/toxigen/ \
    --save_dir "results/toxigen/TinyLlama_selected_0.15_epoch_${epoch}" \
    --model_name_or_path TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T\
    --tokenizer_name_or_path TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T\
    --load_in_8bit\
    --eval_batch_size 256\
    --adapter_path "external_lightning_logs/Instruction__TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_lora_r_128_selected_0.15_run_0/epoch_epoch=${epoch}"\
    --lora_rank 128 --lora_alpha 512

CUDA_VISIBLE_DEVICES=0 python -m eval.toxigen.run_eval \
    --data_dir data/eval/toxigen/ \
    --save_dir "results/toxigen/TinyLlama_selected_0.20_epoch_${epoch}" \
    --model_name_or_path TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T\
    --tokenizer_name_or_path TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T\
    --load_in_8bit\
    --eval_batch_size 256\
    --adapter_path "external_lightning_logs/Instruction__TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_lora_r_128_selected_0.20_run_0/epoch_epoch=${epoch}"\
    --lora_rank 128 --lora_alpha 512
done

for epoch in 0 1 2 3 4 5 6 7 8 9
do
CUDA_VISIBLE_DEVICES=0 python -m eval.truthfulqa.run_eval \
        --data_dir data/eval/truthfulqa \
        --save_dir "results/truthfulqa/TinyLlama_selected_0.20_epoch_${epoch}" \
        --model_name_or_path TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T\
        --tokenizer_name_or_path TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T\
        --metrics mc \
        --preset qa \
        --hf_truth_model_name_or_path allenai/truthfulqa-truth-judge-llama2-7B \
        --hf_info_model_name_or_path allenai/truthfulqa-info-judge-llama2-7B \
        --eval_batch_size 20 \
        --load_in_8bit\
        --adapter_path "external_lightning_logs/Instruction__TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_lora_r_128_selected_0.20_run_0/epoch_epoch=${epoch}"\
        --lora_rank 128 --lora_alpha 512
done