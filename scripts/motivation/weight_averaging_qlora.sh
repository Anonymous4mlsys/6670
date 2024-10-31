task_names=("cb" "multirc" "rte" "winogrande_debiased" "story_cloze" "hellaswag"  "copa" "wic" "wsc.fixed" "boolq") 
checkpoints=(
    "meta-llama-Llama-3.2-1B_cb_lora_r_16_downsampled_single_task_qlora_run_0/epoch_epoch=9.pt"
    "meta-llama-Llama-3.2-1B_multirc_lora_r_16_downsampled_single_task_qlora_run_0/epoch_epoch=4.pt"
    "meta-llama-Llama-3.2-1B_rte_lora_r_16_downsampled_single_task_qlora_run_0/epoch_epoch=5.pt"
    "meta-llama-Llama-3.2-1B_winogrande_debiased_lora_r_16_downsampled_single_task_qlora_run_1/epoch_epoch=9.pt"
    "meta-llama-Llama-3.2-1B_story_cloze_lora_r_16_downsampled_single_task_qlora_run_0/epoch_epoch=7.pt"
    "meta-llama-Llama-3.2-1B_hellaswag_lora_r_16_downsampled_single_task_qlora_run_0/epoch_epoch=3.pt"
    "meta-llama-Llama-3.2-1B_copa_lora_r_16_downsampled_single_task_qlora_run_0/epoch_epoch=6.pt"
    "meta-llama-Llama-3.2-1B_wic_lora_r_16_downsampled_single_task_qlora_run_0/epoch_epoch=8.pt"
    "meta-llama-Llama-3.2-1B_wsc.fixed_lora_r_16_downsampled_single_task_qlora_run_1/epoch_epoch=4.pt"
    "meta-llama-Llama-3.2-1B_boolq_lora_r_16_downsampled_single_task_qlora_run_0/epoch_epoch=3.pt"
)
length=${#task_names[@]}

for j in 0 2 4 6 8; do
for ((i = j+1; i < $length; i++)); do
python evaluate_merged_model.py --task_names "${task_names[$j]}" "${task_names[$i]}" --model_key "meta-llama/Llama-3.2-1B"\
    --downsample_ratio 1e-3 --minimum_samples 500 --minimum_samples_validation 100\
    --devices 2 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 1 --epochs 0\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --use_qlora --optimizer "paged_adamw_32bit"\
    --save_name merging_llama1b_qlora --precision "bf16-true" \
    --merge_model_dirs "${checkpoints[$j]}" "${checkpoints[$i]}"\
    --merge_strategy "averaging" --merge_scale 0.8
done
done