# python evaluate_merged_model.py --task_names "copa" --model_key "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"\
#     --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 1 --epochs 1\
#     --train_lora --lora_rank 16 --lora_alpha 128 \
#     --save_name merging_tinyllama --precision "bf16-true" \
#     --merge_model_dirs "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_rte_copa_wsc.fixed_lora_r_16_boost_4bit_run_0/epoch_epoch=6.pt" \
#     "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_cb_lora_r_16_boost_4bit_run_0/epoch_epoch=7.pt"\
#     "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_wic_lora_r_16_boost_4bit_run_0/epoch_epoch=6.pt"\
#     "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_boolq_lora_r_16_boost_4bit_run_0/epoch_epoch=6.pt"\
#     --merge_strategy "simple_ensemble" --use_qlora --optimizer "paged_adamw_32bit"

python evaluate_merged_model.py --task_names "cb" "rte" --model_key "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"\
    --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 1 --epochs 0\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name merging_tinyllama --precision "bf16-true" \
    --merge_model_dirs "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_rte_copa_wsc.fixed_lora_r_16_boost_4bit_run_0/epoch_epoch=6.pt" \
    "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_cb_lora_r_16_boost_4bit_run_0/epoch_epoch=7.pt"\
    "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_wic_lora_r_16_boost_4bit_run_0/epoch_epoch=6.pt"\
    "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_boolq_lora_r_16_boost_4bit_run_0/epoch_epoch=6.pt"\
    --merge_strategy "simple_ensemble" --use_qlora --optimizer "paged_adamw_32bit"

python evaluate_merged_model.py --task_names "copa" "cb" "rte" "wic" "wsc.fixed" "boolq" --model_key "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"\
    --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 1 --epochs 0\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name merging_tinyllama --precision "bf16-true" \
    --merge_model_dirs "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_rte_copa_wsc.fixed_lora_r_16_boost_4bit_run_0/epoch_epoch=6.pt" \
    "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_cb_lora_r_16_boost_4bit_run_0/epoch_epoch=7.pt"\
    "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_wic_lora_r_16_boost_4bit_run_0/epoch_epoch=6.pt"\
    "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_boolq_lora_r_16_boost_4bit_run_0/epoch_epoch=6.pt"\
    --merge_strategy "simple_ensemble" --use_qlora --optimizer "paged_adamw_32bit"

python evaluate_merged_model.py --task_names "copa" "cb" "rte" "wic" "wsc.fixed" "boolq" --model_key "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"\
    --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 1 --epochs 0\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name merging_tinyllama --precision "bf16-true" \
    --merge_model_dirs "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_rte_copa_wsc.fixed_lora_r_16_boost_4bit_run_1/epoch_epoch=9.pt" \
    "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_cb_lora_r_16_boost_4bit_run_1/epoch_epoch=5.pt"\
    "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_wic_lora_r_16_boost_4bit_run_1/epoch_epoch=2.pt"\
    "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_boolq_lora_r_16_boost_4bit_run_1/epoch_epoch=3.pt"\
    --merge_strategy "simple_ensemble" --use_qlora --optimizer "paged_adamw_32bit"

python evaluate_merged_model.py --task_names "copa" "cb" "rte" "wic" "wsc.fixed" "boolq" --model_key "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"\
    --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 1 --epochs 0\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name merging_tinyllama --precision "bf16-true" \
    --merge_model_dirs "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_rte_copa_wsc.fixed_lora_r_16_boost_4bit_run_0/epoch_epoch=6.pt" \
    "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_cb_lora_r_16_boost_4bit_run_0/epoch_epoch=7.pt"\
    "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_wic_lora_r_16_boost_4bit_run_0/epoch_epoch=6.pt"\
    "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_boolq_lora_r_16_boost_4bit_run_0/epoch_epoch=6.pt"\
    --merge_strategy "max_ensemble" --use_qlora --optimizer "paged_adamw_32bit"


python evaluate_merged_model.py --task_names "copa" "cb" "rte" "wic" "wsc.fixed" "boolq" --model_key "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"\
    --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 1 --epochs 0\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name merging_tinyllama --precision "bf16-true" \
    --merge_model_dirs "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_rte_copa_wsc.fixed_lora_r_16_boost_4bit_run_1/epoch_epoch=9.pt" \
    "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_cb_lora_r_16_boost_4bit_run_1/epoch_epoch=5.pt"\
    "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_wic_lora_r_16_boost_4bit_run_1/epoch_epoch=2.pt"\
    "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_boolq_lora_r_16_boost_4bit_run_1/epoch_epoch=3.pt"\
    --merge_strategy "max_ensemble" --use_qlora --optimizer "paged_adamw_32bit"


# k = 3

python evaluate_merged_model.py --task_names "copa" "cb" "rte" "wic" "wsc.fixed" "boolq" --model_key "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"\
    --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 1 --epochs 0\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name merging_tinyllama --precision "bf16-true" \
    --merge_model_dirs "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_rte_copa_wsc.fixed_lora_r_16_boost_4bit_run_0/epoch_epoch=6.pt" \
    "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_cb_boolq_lora_r_16_boost_4bit_run_0/epoch_epoch=9.pt"\
    "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_wic_lora_r_16_boost_4bit_run_0/epoch_epoch=6.pt"\
    --merge_strategy "simple_ensemble" --use_qlora --optimizer "paged_adamw_32bit"

python evaluate_merged_model.py --task_names "copa" "cb" "rte" "wic" "wsc.fixed" "boolq" --model_key "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"\
    --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 1 --epochs 0\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name merging_tinyllama --precision "bf16-true" \
    --merge_model_dirs "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_rte_copa_wsc.fixed_lora_r_16_boost_4bit_run_1/epoch_epoch=9.pt" \
    "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_cb_boolq_lora_r_16_boost_4bit_run_1/epoch_epoch=9.pt"\
    "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_wic_lora_r_16_boost_4bit_run_1/epoch_epoch=2.pt"\
    --merge_strategy "simple_ensemble" --use_qlora --optimizer "paged_adamw_32bit"

python evaluate_merged_model.py --task_names "copa" "cb" "rte" "wic" "wsc.fixed" "boolq" --model_key "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"\
    --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 1 --epochs 0\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name merging_tinyllama --precision "bf16-true" \
    --merge_model_dirs "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_rte_copa_wsc.fixed_lora_r_16_boost_4bit_run_0/epoch_epoch=6.pt" \
    "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_cb_boolq_lora_r_16_boost_4bit_run_0/epoch_epoch=9.pt"\
    "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_wic_lora_r_16_boost_4bit_run_0/epoch_epoch=6.pt"\
    --merge_strategy "max_ensemble" --use_qlora --optimizer "paged_adamw_32bit"


python evaluate_merged_model.py --task_names "copa" "cb" "rte" "wic" "wsc.fixed" "boolq" --model_key "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"\
    --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 1 --epochs 0\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name merging_tinyllama --precision "bf16-true" \
    --merge_model_dirs "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_rte_copa_wsc.fixed_lora_r_16_boost_4bit_run_1/epoch_epoch=9.pt" \
    "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_cb_boolq_lora_r_16_boost_4bit_run_1/epoch_epoch=9.pt"\
    "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_wic_lora_r_16_boost_4bit_run_1/epoch_epoch=2.pt"\
    --merge_strategy "max_ensemble" --use_qlora --optimizer "paged_adamw_32bit"

# python custom_train_glue_mtl.py --task_names "cb" \
#     --model_key  "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"\
#     --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5\
#     --train_lora --lora_rank 16 --lora_alpha 128 \
#     --save_name boost_4bit --epochs 10 --use_qlora --optimizer "paged_adamw_32bit" --precision "bf16-true" 

# python custom_train_glue_mtl.py --task_names "wic" \
#     --model_key  "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"\
#     --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5\
#     --train_lora --lora_rank 16 --lora_alpha 128 \
#     --save_name boost_4bit --epochs 10 --use_qlora --optimizer "paged_adamw_32bit" --precision "bf16-true" 

# python custom_train_glue_mtl.py --task_names "boolq" \
#     --model_key  "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"\
#     --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5\
#     --train_lora --lora_rank 16 --lora_alpha 128 \
#     --save_name boost_4bit --epochs 10 --use_qlora --optimizer "paged_adamw_32bit" --precision "bf16-true" 

# python custom_train_glue_mtl.py --task_names "cb" "boolq" \
#     --model_key  "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"\
#     --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5\
#     --train_lora --lora_rank 16 --lora_alpha 128 \
#     --save_name boost_4bit --epochs 10 --use_qlora --optimizer "paged_adamw_32bit" --precision "bf16-true" 
