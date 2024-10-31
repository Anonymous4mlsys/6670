# MTL
python custom_train_glue_mtl.py --task_name "cb" "multirc" "rte" "winogrande_debiased" "story_cloze" "hellaswag"  "copa" "wic" "wsc.fixed" "boolq"\
    --downsample_ratio 1e-3 --minimum_samples 500 --minimum_samples_validation 100 \
    --model_key "meta-llama/Llama-3.1-8B"\
    --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5\
    --train_adapter --use_qadapter --reduction_factor 128 --precision "bf16-true" \
    --save_name downsampled_mtl_qadapter --epochs 10 --write_results

# cb multirc
# rte winogrande_debiased story_cloze
# copa wic wsc.fixed boolq
# hellaswag

python custom_train_glue_mtl.py --task_names "cb" "multirc" \
    --downsample_ratio 1e-3 --minimum_samples 500 --minimum_samples_validation 100 \
    --model_key "meta-llama/Llama-3.1-8B"\
    --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5\
    --train_adapter --use_qadapter --reduction_factor 128 --precision "bf16-true" \
    --save_name downsampled_mtl_qadapter --epochs 10 --write_results

python custom_train_glue_mtl.py --task_names "rte" "winogrande_debiased" "story_cloze" "hellaswag" \
    --downsample_ratio 1e-3 --minimum_samples 500 --minimum_samples_validation 100 \
    --model_key "meta-llama/Llama-3.1-8B"\
    --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5\
    --train_adapter --use_qadapter --reduction_factor 128 --precision "bf16-true" \
    --save_name downsampled_mtl_qadapter --epochs 10 --write_results

python custom_train_glue_mtl.py --task_names "copa" "wic" "wsc.fixed" "boolq" \
    --downsample_ratio 1e-3 --minimum_samples 500 --minimum_samples_validation 100 \
    --model_key "meta-llama/Llama-3.1-8B"\
    --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5\
    --train_adapter --use_qadapter --reduction_factor 128 --precision "bf16-true" \
    --save_name downsampled_mtl_qadapter --epochs 10 --write_results

python custom_train_glue_mtl.py --task_names "hellaswag" \
    --downsample_ratio 1e-3 --minimum_samples 500 --minimum_samples_validation 100 \
    --model_key "meta-llama/Llama-3.1-8B"\
    --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5\
    --train_adapter --use_qadapter --reduction_factor 128 --precision "bf16-true" \
    --save_name downsampled_mtl_qadapter --epochs 10 --write_results


# Group = 8
# cb multirc
# rte
# copa boolq
# wic
# wsc.fixed
# winogrande_debiased
# story_cloze
# hellaswag

# python custom_train_glue_mtl.py --task_names "cb" "multirc" \
#     --downsample_ratio 1e-3 --minimum_samples 500 --minimum_samples_validation 100 \
#     --model_key "meta-llama/Llama-3.1-8B"\
#     --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5\
#     --train_adapter --use_qadapter --reduction_factor 128 --precision "bf16-true" \
#     --save_name downsampled_mtl_qadapter --epochs 10 --write_results

python custom_train_glue_mtl.py --task_names "rte" \
    --downsample_ratio 1e-3 --minimum_samples 500 --minimum_samples_validation 100 \
    --model_key "meta-llama/Llama-3.1-8B"\
    --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5\
    --train_adapter --use_qadapter --reduction_factor 128 --precision "bf16-true" \
    --save_name downsampled_mtl_qadapter --epochs 10 --write_results

python custom_train_glue_mtl.py --task_names "copa" "boolq" \
    --downsample_ratio 1e-3 --minimum_samples 500 --minimum_samples_validation 100 \
    --model_key "meta-llama/Llama-3.1-8B"\
    --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5\
    --train_adapter --use_qadapter --reduction_factor 128 --precision "bf16-true" \
    --save_name downsampled_mtl_qadapter --epochs 10 --write_results

python custom_train_glue_mtl.py --task_names "wic" \
    --downsample_ratio 1e-3 --minimum_samples 500 --minimum_samples_validation 100 \
    --model_key "meta-llama/Llama-3.1-8B"\
    --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5\
    --train_adapter --use_qadapter --reduction_factor 128 --precision "bf16-true" \
    --save_name downsampled_mtl_qadapter --epochs 10 --write_results

python custom_train_glue_mtl.py --task_names "winogrande_debiased" \
    --downsample_ratio 1e-3 --minimum_samples 500 --minimum_samples_validation 100 \
    --model_key "meta-llama/Llama-3.1-8B"\
    --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5\
    --train_adapter --use_qadapter --reduction_factor 128 --precision "bf16-true" \
    --save_name downsampled_mtl_qadapter --epochs 10 --write_results

# python custom_train_glue_mtl.py --task_names "hellaswag" \
#     --downsample_ratio 1e-3 --minimum_samples 500 --minimum_samples_validation 100 \
#     --model_key "meta-llama/Llama-3.1-8B"\
#     --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5\
#     --train_adapter --use_qadapter --reduction_factor 128 --precision "bf16-true" \
#     --save_name downsampled_mtl_qadapter --epochs 10 --write_results