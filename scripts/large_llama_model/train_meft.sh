task_names=("story_cloze" "hellaswag"  "copa" "wic" "wsc.fixed" "boolq") 
load_model_dir="meta-llama-CodeLlama-34b-Instruct-hf_cb_multirc_rte_winogrande_debiased_story_cloze_hellaswag_copa_wic_wsc.fixed_boolq_lora_r_16_downsampled_mtl_qlora_run_0/epoch_epoch=9.pt"

hf_key="meta-llama/CodeLlama-34b-Instruct-hf"
bs=4

for task_name in "${task_names[@]}"
do
python custom_train_glue_mtl.py --task_names $task_name \
    --downsample_ratio 1e-3 --minimum_samples 500 --minimum_samples_validation 100 \
    --model_key $hf_key\
    --devices 1 --batch_size $bs --inference_batch_size 8 --max_length 256 --runs 1 --lr 5e-5\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --use_qlora --optimizer "paged_adamw_32bit"\
    --save_name downsampled_mtft_qlora --epochs 10 --write_results --load_model_dir $load_model_dir\
    # --precision "bf16-true"
done