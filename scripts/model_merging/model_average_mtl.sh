for scale in 1.0 0.8 0.6 0.4 0.2 1.2 1.4 1.6 1.8 2.0
do
python evaluate_merged_model.py --task_names "cb" "rte" "copa" "wic" "wsc.fixed" "boolq" --model_key "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"\
    --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 1 --epochs 0\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name merging_tinyllama --precision "bf16-true" \
    --merge_model_dirs "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_cb_lora_r_16_pairwise_run_0/epoch_epoch=6.pt" \
    "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_rte_lora_r_16_pairwise_run_0/epoch_epoch=3.pt"\
    "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_copa_lora_r_16_pairwise_run_0/epoch_epoch=8.pt"\
    "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_wic_lora_r_16_pairwise_run_1/epoch_epoch=8.pt"\
    "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_wsc.fixed_lora_r_16_pairwise_run_1/epoch_epoch=8.pt"\
    "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_boolq_lora_r_16_pairwise_run_0/epoch_epoch=6.pt"\
    --merge_strategy "averaging" --merge_scale $scale

python evaluate_merged_model.py --task_names "cb" "rte" "copa" "wic" "wsc.fixed" "boolq" --model_key "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"\
    --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 1 --epochs 0\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name merging_tinyllama --precision "bf16-true" \
    --merge_model_dirs "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_cb_lora_r_16_pairwise_run_0/epoch_epoch=6.pt" \
    "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_rte_lora_r_16_pairwise_run_0/epoch_epoch=3.pt"\
    "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_copa_lora_r_16_pairwise_run_0/epoch_epoch=8.pt"\
    "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_wic_lora_r_16_pairwise_run_1/epoch_epoch=8.pt"\
    "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_wsc.fixed_lora_r_16_pairwise_run_1/epoch_epoch=8.pt"\
    "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_boolq_lora_r_16_pairwise_run_0/epoch_epoch=6.pt"\
    --merge_strategy "arithmetic" --merge_scale $scale

python evaluate_merged_model.py --task_names "cb" "rte" "copa" "wic" "wsc.fixed" "boolq" --model_key "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"\
    --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 1 --epochs 0\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name merging_tinyllama --precision "bf16-true" \
    --merge_model_dirs "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_cb_lora_r_16_pairwise_run_0/epoch_epoch=6.pt" \
    "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_rte_lora_r_16_pairwise_run_0/epoch_epoch=3.pt"\
    "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_copa_lora_r_16_pairwise_run_0/epoch_epoch=8.pt"\
    "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_wic_lora_r_16_pairwise_run_1/epoch_epoch=8.pt"\
    "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_wsc.fixed_lora_r_16_pairwise_run_1/epoch_epoch=8.pt"\
    "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_boolq_lora_r_16_pairwise_run_0/epoch_epoch=6.pt"\
    --merge_strategy "ties" --merge_scale $scale --merge_function "topk0.5_mass_dis-mean"

python evaluate_merged_model.py --task_names "cb" "rte" "copa" "wic" "wsc.fixed" "boolq" --model_key "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"\
    --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 1 --epochs 0\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name merging_tinyllama --precision "bf16-true" \
    --merge_model_dirs "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_cb_lora_r_16_pairwise_run_0/epoch_epoch=6.pt" \
    "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_rte_lora_r_16_pairwise_run_0/epoch_epoch=3.pt"\
    "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_copa_lora_r_16_pairwise_run_0/epoch_epoch=8.pt"\
    "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_wic_lora_r_16_pairwise_run_1/epoch_epoch=8.pt"\
    "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_wsc.fixed_lora_r_16_pairwise_run_1/epoch_epoch=8.pt"\
    "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_boolq_lora_r_16_pairwise_run_0/epoch_epoch=6.pt"\
    --merge_strategy "ties" --merge_scale $scale --merge_function "topk0.3_mass_dis-mean"

python evaluate_merged_model.py --task_names "cb" "rte" "copa" "wic" "wsc.fixed" "boolq" --model_key "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"\
    --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 1 --epochs 0\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name merging_tinyllama --precision "bf16-true" \
    --merge_model_dirs "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_cb_lora_r_16_pairwise_run_0/epoch_epoch=6.pt" \
    "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_rte_lora_r_16_pairwise_run_0/epoch_epoch=3.pt"\
    "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_copa_lora_r_16_pairwise_run_0/epoch_epoch=8.pt"\
    "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_wic_lora_r_16_pairwise_run_1/epoch_epoch=8.pt"\
    "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_wsc.fixed_lora_r_16_pairwise_run_1/epoch_epoch=8.pt"\
    "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_boolq_lora_r_16_pairwise_run_0/epoch_epoch=6.pt"\
    --merge_strategy "ties" --merge_scale $scale --merge_function "topk0.1_mass_dis-mean"
done
