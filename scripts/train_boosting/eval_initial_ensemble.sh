python custom_train_boosting_model.py --task_names "cb" "rte" "copa" "wic" "wsc.fixed" "boolq" "multirc" "winogrande_debiased" "story_cloze" "hellaswag"\
    --model_key "meta-llama/Llama-3.2-1B" \
    --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --lr 5e-5\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name task_group_2 --n_estimators 0 --epochs 1 --write_results --precision "bf16-true" --train_gradient_boosting \
    --load_model_dirs "meta-llama-Llama-3.2-1B_cb_multirc_lora_r_16_task_grouping_2_run_0/epoch_epoch=3.pt" \
    "meta-llama-Llama-3.2-1B_rte_copa_wic_wsc.fixed_boolq_winogrande_debiased_story_cloze_hellaswag_lora_r_16_task_grouping_2_run_0/epoch_epoch=7.pt"

python custom_train_boosting_model.py --task_names "cb" "rte" "copa" "wic" "wsc.fixed" "boolq" "multirc" "winogrande_debiased" "story_cloze" "hellaswag"\
    --model_key "meta-llama/Llama-3.2-1B" \
    --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --lr 5e-5\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name task_group_3 --n_estimators 0 --epochs 1 --write_results --precision "bf16-true" --train_gradient_boosting \
    --load_model_dirs "meta-llama-Llama-3.2-1B_cb_multirc_lora_r_16_task_grouping_2_run_0/epoch_epoch=3.pt" \
    "meta-llama-Llama-3.2-1B_rte_winogrande_debiased_story_cloze_hellaswag_lora_r_16_task_grouping_4_run_0/epoch_epoch=8.pt" \
    "meta-llama-Llama-3.2-1B_copa_wic_wsc.fixed_boolq_lora_r_16_task_grouping_4_run_0/epoch_epoch=9.pt" 

python custom_train_boosting_model.py --task_names "cb" "rte" "copa" "wic" "wsc.fixed" "boolq" "multirc" "winogrande_debiased" "story_cloze" "hellaswag"\
    --model_key "meta-llama/Llama-3.2-1B" \
    --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --lr 5e-5\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name task_group_4 --n_estimators 0 --epochs 1 --write_results --precision "bf16-true" --train_gradient_boosting \
    --load_model_dirs "meta-llama-Llama-3.2-1B_cb_multirc_lora_r_16_task_grouping_2_run_0/epoch_epoch=3.pt" \
        "meta-llama-Llama-3.2-1B_rte_winogrande_debiased_story_cloze_hellaswag_lora_r_16_task_grouping_4_run_0/epoch_epoch=8.pt" \
        "meta-llama-Llama-3.2-1B_copa_wic_wsc.fixed_boolq_lora_r_16_task_grouping_4_run_0/epoch_epoch=9.pt" \
        "meta-llama-Llama-3.2-1B_hellaswag_lora_r_16_task_grouping_4_run_0/epoch_epoch=4.pt"

python custom_train_boosting_model.py --task_names "cb" "rte" "copa" "wic" "wsc.fixed" "boolq" "multirc" "winogrande_debiased" "story_cloze" "hellaswag"\
    --model_key "meta-llama/Llama-3.2-1B" \
    --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --lr 5e-5\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name task_group_4 --n_estimators 0 --epochs 1 --write_results --precision "bf16-true" --train_gradient_boosting \
    --load_model_dirs "meta-llama-Llama-3.2-1B_cb_multirc_lora_r_16_task_grouping_2_run_0/epoch_epoch=3.pt" \
        "meta-llama-Llama-3.2-1B_rte_winogrande_debiased_lora_r_16_task_grouping_5_run_0/epoch_epoch=5.pt"\
        "meta-llama-Llama-3.2-1B_copa_wic_wsc.fixed_boolq_lora_r_16_task_grouping_4_run_0/epoch_epoch=9.pt"\
        "meta-llama-Llama-3.2-1B_hellaswag_lora_r_16_task_grouping_4_run_0/epoch_epoch=4.pt"\
        "meta-llama-Llama-3.2-1B_story_cloze_lora_r_16_task_grouping_5_run_0/epoch_epoch=7.pt"

# Group 2
    # "meta-llama-Llama-3.2-1B_cb_multirc_lora_r_16_task_grouping_2_run_0/epoch_epoch=3.pt",
    # "meta-llama-Llama-3.2-1B_rte_copa_wic_wsc.fixed_boolq_winogrande_debiased_story_cloze_hellaswag_lora_r_16_task_grouping_2_run_0/epoch_epoch=7.pt",

# Group 3
    # "meta-llama-Llama-3.2-1B_cb_multirc_lora_r_16_task_grouping_2_run_0/epoch_epoch=3.pt",
    # "meta-llama-Llama-3.2-1B_rte_winogrande_debiased_story_cloze_hellaswag_lora_r_16_task_grouping_4_run_0/epoch_epoch=8.pt",
    # "meta-llama-Llama-3.2-1B_copa_wic_wsc.fixed_boolq_lora_r_16_task_grouping_4_run_0/epoch_epoch=9.pt"

# Group 4
    # "meta-llama-Llama-3.2-1B_cb_multirc_lora_r_16_task_grouping_2_run_0/epoch_epoch=3.pt",
    # "meta-llama-Llama-3.2-1B_rte_winogrande_debiased_story_cloze_hellaswag_lora_r_16_task_grouping_4_run_0/epoch_epoch=8.pt",
    # "meta-llama-Llama-3.2-1B_copa_wic_wsc.fixed_boolq_lora_r_16_task_grouping_4_run_0/epoch_epoch=9.pt",
    # "meta-llama-Llama-3.2-1B_hellaswag_lora_r_16_task_grouping_4_run_0/epoch_epoch=4.pt",

# Group 5
    # "meta-llama-Llama-3.2-1B_cb_multirc_lora_r_16_task_grouping_2_run_0/epoch_epoch=3.pt",
    # "meta-llama-Llama-3.2-1B_rte_winogrande_debiased_lora_r_16_task_grouping_5_run_0/epoch_epoch=5.pt",
    # "meta-llama-Llama-3.2-1B_copa_wic_wsc.fixed_boolq_lora_r_16_task_grouping_4_run_0/epoch_epoch=9.pt",
    # "meta-llama-Llama-3.2-1B_hellaswag_lora_r_16_task_grouping_4_run_0/epoch_epoch=4.pt",
    # "meta-llama-Llama-3.2-1B_story_cloze_lora_r_16_task_grouping_5_run_0/epoch_epoch=7.pt",

# Group 6 & 7
    # "meta-llama-Llama-3.2-1B_cb_multirc_lora_r_16_task_grouping_2_run_0/epoch_epoch=3.pt",
    # "meta-llama-Llama-3.2-1B_rte_lora_r_16_task_grouping_7_run_0/epoch_epoch=4.pt",
    # "meta-llama-Llama-3.2-1B_copa_wsc.fixed_boolq_lora_r_16_task_grouping_7_run_0/epoch_epoch=9.pt",
    # "meta-llama-Llama-3.2-1B_wic_lora_r_16_task_grouping_7_run_0/epoch_epoch=9.pt",
    # "meta-llama-Llama-3.2-1B_winogrande_debiased_lora_r_16_task_grouping_7_run_0/epoch_epoch=8.pt",
    # "meta-llama-Llama-3.2-1B_hellaswag_lora_r_16_task_grouping_4_run_0/epoch_epoch=4.pt",
    # "meta-llama-Llama-3.2-1B_story_cloze_lora_r_16_task_grouping_5_run_0/epoch_epoch=7.pt",

# Group 8
    # "meta-llama-Llama-3.2-1B_cb_multirc_lora_r_16_task_grouping_2_run_0/epoch_epoch=3.pt",
    # "meta-llama-Llama-3.2-1B_rte_lora_r_16_task_grouping_7_run_0/epoch_epoch=4.pt",
    # "meta-llama-Llama-3.2-1B_copa_boolq_lora_r_16_task_grouping_8_run_0/epoch_epoch=4.pt",
    # "meta-llama-Llama-3.2-1B_wic_lora_r_16_task_grouping_7_run_0/epoch_epoch=9.pt",
    # "meta-llama-Llama-3.2-1B_winogrande_debiased_lora_r_16_task_grouping_7_run_0/epoch_epoch=8.pt",
    # "meta-llama-Llama-3.2-1B_hellaswag_lora_r_16_task_grouping_4_run_0/epoch_epoch=4.pt",
    # "meta-llama-Llama-3.2-1B_story_cloze_lora_r_16_task_grouping_5_run_0/epoch_epoch=7.pt",
    # "meta-llama-Llama-3.2-1B_wsc.fixed_lora_r_16_single_task_16bit_run_0/epoch_epoch=8.pt",
