load_model_dir="meta-llama-CodeLlama-34b-Instruct-hf_cb_multirc_rte_winogrande_debiased_story_cloze_hellaswag_copa_wic_wsc.fixed_boolq_lora_r_16_downsampled_mtl_qlora_run_0/epoch_epoch=9.pt"
hf_key="meta-llama/CodeLlama-34b-Instruct-hf"
bs=4
dv=0


python custom_train_glue_mtl.py --task_names "cb" "wsc.fixed" \
    --downsample_ratio 1e-3 --minimum_samples 500 --minimum_samples_validation 100 \
    --model_key $hf_key\
    --devices $dv --batch_size $bs --inference_batch_size 8 --max_length 256 --runs 1 --lr 5e-5\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --use_qlora --optimizer "paged_adamw_32bit"\
    --save_name downsampled_mtft_qlora --epochs 10 --write_results --load_model_dir $load_model_dir --precision "bf16-true"


python custom_train_glue_mtl.py --task_names "rte" "boolq" "winogrande_debiased" "story_cloze" "hellaswag" \
    --downsample_ratio 1e-3 --minimum_samples 500 --minimum_samples_validation 100 \
    --model_key $hf_key\
    --devices $dv --batch_size $bs --inference_batch_size 8 --max_length 256 --runs 1 --lr 5e-5\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --use_qlora --optimizer "paged_adamw_32bit"\
    --save_name downsampled_mtft_qlora --epochs 10 --write_results --load_model_dir $load_model_dir --precision "bf16-true"


python custom_train_glue_mtl.py --task_names "copa" "wic" \
    --downsample_ratio 1e-3 --minimum_samples 500 --minimum_samples_validation 100 \
    --model_key $hf_key\
    --devices $dv --batch_size $bs --inference_batch_size 8 --max_length 256 --runs 1 --lr 5e-5\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --use_qlora --optimizer "paged_adamw_32bit"\
    --save_name downsampled_mtft_qlora --epochs 10 --write_results --load_model_dir $load_model_dir --precision "bf16-true"


# Group = 2
# cb wsc.fixed multirc
# rte copa wic boolq winogrande_debiased story_cloze hellaswag

# Group = 4
# cb wsc.fixed
# rte boolq winogrande_debiased story_cloze hellaswag
# copa wic
# multirc

# Group = 5
# cb wsc.fixed
# rte boolq hellaswag
# copa wic winogrande_debiased
# multirc
# story_cloze

# Group = 6
# cb wsc.fixed
# rte boolq hellaswag
# copa wic winogrande_debiased
# multirc
# story_cloze

# Group = 8
# cb wsc.fixed
# rte
# copa hellaswag
# wic
# multirc
# boolq
# winogrande_debiased
# story_cloze
