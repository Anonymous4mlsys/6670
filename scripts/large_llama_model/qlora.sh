hf_key="meta-llama/CodeLlama-34b-Instruct-hf"
dev=1
bs=4
sn=downsampled_mtl_qlora
# "EleutherAI/gpt-j-6b"
# "meta-llama/Llama-3.2-1B"
# "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
# "meta-llama/CodeLlama-7b-hf"
# "openai-community/gpt2-medium"
# "meta-llama/Llama-2-13b-hf"
# "meta-llama/CodeLlama-34b-Instruct-hf"


python custom_train_glue_mtl.py --task_name "cb" "multirc" "rte" "winogrande_debiased" "story_cloze" "hellaswag"  "copa" "wic" "wsc.fixed" "boolq"\
    --downsample_ratio 1e-3 --minimum_samples 500 --minimum_samples_validation 100\
    --model_key $hf_key\
    --devices $dev --batch_size $bs --inference_batch_size 8 --max_length 256 --runs 2 --lr 2e-5 --precision "bf16-true"\
    --optimizer "paged_adamw_32bit"\
    --save_name $sn --epochs 10 --write_results\
    --use_qlora --train_lora --lora_rank 16 --lora_alpha 128 




# task_names=("cb" "multirc" "rte" "winogrande_debiased" "story_cloze" "hellaswag"  "copa" "wic" "wsc.fixed" "boolq") 
# length=${#task_names[@]}
