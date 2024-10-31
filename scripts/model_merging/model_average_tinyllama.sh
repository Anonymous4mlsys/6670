scale=0.8
# python evaluate_merged_model.py --task_names "cb" "rte" --model_key "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"\
#     --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 1 --epochs 0\
#     --train_lora --lora_rank 16 --lora_alpha 128 \
#     --save_name merging_tinyllama --precision "bf16-true" \
#     --merge_model_dirs "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_cb_lora_r_16_pairwise_run_0/epoch_epoch=6.pt" "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_rte_lora_r_16_pairwise_run_0/epoch_epoch=3.pt"\
#     --merge_strategy "arithmetic" --merge_scale $scale

# python evaluate_merged_model.py --task_names "cb" "rte" --model_key "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"\
#     --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 1 --epochs 0\
#     --train_lora --lora_rank 16 --lora_alpha 128 \
#     --save_name merging_tinyllama --precision "bf16-true" \
#     --merge_model_dirs "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_cb_lora_r_16_pairwise_run_1/epoch_epoch=4.pt" "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_rte_lora_r_16_pairwise_run_1/epoch_epoch=3.pt"\
#     --merge_strategy "arithmetic" --merge_scale $scale


python evaluate_merged_model.py --task_names "cb" "copa" --model_key "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"\
    --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 1 --epochs 0\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name merging_tinyllama --precision "bf16-true" \
    --merge_model_dirs "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_cb_lora_r_16_pairwise_run_0/epoch_epoch=6.pt" "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_copa_lora_r_16_pairwise_run_0/epoch_epoch=8.pt"\
    --merge_strategy "arithmetic" --merge_scale $scale

python evaluate_merged_model.py --task_names "cb" "copa" --model_key "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"\
    --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 1 --epochs 0\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name merging_tinyllama --precision "bf16-true" \
    --merge_model_dirs "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_cb_lora_r_16_pairwise_run_1/epoch_epoch=4.pt" "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_copa_lora_r_16_pairwise_run_1/epoch_epoch=9.pt"\
    --merge_strategy "arithmetic" --merge_scale $scale


python evaluate_merged_model.py --task_names "cb" "wic" --model_key "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"\
    --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 1 --epochs 0\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name merging_tinyllama --precision "bf16-true" \
    --merge_model_dirs "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_cb_lora_r_16_pairwise_run_0/epoch_epoch=6.pt" "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_wic_lora_r_16_pairwise_run_1/epoch_epoch=8.pt"\
    --merge_strategy "arithmetic" --merge_scale $scale

python evaluate_merged_model.py --task_names "cb" "wic" --model_key "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"\
    --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 1 --epochs 0\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name merging_tinyllama --precision "bf16-true" \
    --merge_model_dirs "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_cb_lora_r_16_pairwise_run_1/epoch_epoch=4.pt" "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_wic_lora_r_16_pairwise_run_0/epoch_epoch=2.pt"\
    --merge_strategy "arithmetic" --merge_scale $scale



python evaluate_merged_model.py --task_names "cb" "wsc.fixed" --model_key "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"\
    --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 1 --epochs 0\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name merging_tinyllama --precision "bf16-true" \
    --merge_model_dirs "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_cb_lora_r_16_pairwise_run_0/epoch_epoch=6.pt" "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_wsc.fixed_lora_r_16_pairwise_run_0/epoch_epoch=0.pt"\
    --merge_strategy "arithmetic" --merge_scale $scale

python evaluate_merged_model.py --task_names "cb" "wsc.fixed" --model_key "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"\
    --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 1 --epochs 0\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name merging_tinyllama --precision "bf16-true" \
    --merge_model_dirs "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_cb_lora_r_16_pairwise_run_1/epoch_epoch=4.pt" "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_wsc.fixed_lora_r_16_pairwise_run_1/epoch_epoch=8.pt"\
    --merge_strategy "arithmetic" --merge_scale $scale


python evaluate_merged_model.py --task_names "cb" "boolq" --model_key "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"\
    --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 1 --epochs 0\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name merging_tinyllama --precision "bf16-true" \
    --merge_model_dirs "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_cb_lora_r_16_pairwise_run_0/epoch_epoch=6.pt" "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_boolq_lora_r_16_pairwise_run_0/epoch_epoch=6.pt"\
    --merge_strategy "arithmetic" --merge_scale $scale

python evaluate_merged_model.py --task_names "cb" "boolq" --model_key "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"\
    --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 1 --epochs 0\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name merging_tinyllama --precision "bf16-true" \
    --merge_model_dirs "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_cb_lora_r_16_pairwise_run_1/epoch_epoch=4.pt" "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_boolq_lora_r_16_pairwise_run_1/epoch_epoch=8.pt"\
    --merge_strategy "arithmetic" --merge_scale $scale