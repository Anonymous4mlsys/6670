# Boosting finetuning

## Enviroment

Please run following commands:
```bash
pip install -r requirements.txt
mkdir ./results
mkdir ./data
mkdir ./external_lightning_logs
```

Our baseline contains 10 different data sources. For `Story Cloze` dataset, please follow instructions [here](https://cs.rochester.edu/nlp/rocstories/) to fetch. 

### Step 1: compute approximation on pretrained model
```bash
hf_key="meta-llama/CodeLlama-34b-Instruct-hf"
dev=0
bs=4
sn=codellama_glue_10tasks

python fast_estimate_compute_gradients_glue.py --task_names "cb" "rte" "copa" "wic" "wsc.fixed" "boolq" "multirc" "winogrande_debiased" "story_cloze" "hellaswag"\
    --model_key $hf_key\
    --downsample_ratio 1e-3 --minimum_samples 500 --minimum_samples_validation 100\
    --devices $dev --batch_size $bs --inference_batch_size 4 --max_length 256 --runs 1 --lr 5e-5\
    --save_name $sn --epochs 0 --precision "bf16-true" \
    --compute_gradient_steps 10000000 --compute_gradient_seed 0 --project_gradients_dim 200 --start_step 0\
    --train_lora --use_qlora --lora_rank 16 --lora_alpha 128
```

### Step 2: use the gradient to estimate fine-tuned losses

```bash
hf_key="meta-llama/CodeLlama-34b-Instruct-hf"
dev=0
bs=4

python fast_estimate_linear_regression_glue.py --task_names "cb" "rte" "copa" "wic" "wsc.fixed" "boolq" "multirc" "winogrande_debiased" "story_cloze" "hellaswag"\
    --model_key $hf_key\
    --downsample_ratio 1e-3 --minimum_samples 500 --minimum_samples_validation 100\
    --devices $dev --batch_size $bs --inference_batch_size 8 --max_length 256 --runs 1 --lr 5e-5\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --use_qlora --optimizer "paged_adamw_32bit"\
    --save_name codellama_glue_10tasks --epochs 0 --precision "bf16-true" \
    --compute_gradient_seed 0 --project_gradients_dim 200 --scale 0.3 --number_of_subsets 120 --subset_size 3

```

### Step 3: train QLoRA
```bash
hf_key="meta-llama/CodeLlama-34b-Instruct-hf"
dev=1
bs=4
sn=downsampled_mtl_qlora

python custom_train_glue_mtl.py --task_name "cb" "multirc" "rte" "winogrande_debiased" "story_cloze" "hellaswag"  "copa" "wic" "wsc.fixed" "boolq"\
    --downsample_ratio 1e-3 --minimum_samples 500 --minimum_samples_validation 100\
    --model_key $hf_key\
    --devices $dev --batch_size $bs --inference_batch_size 8 --max_length 256 --runs 2 --lr 2e-5 --precision "bf16-true"\
    --optimizer "paged_adamw_32bit"\
    --save_name $sn --epochs 10 --write_results\
    --use_qlora --train_lora --lora_rank 16 --lora_alpha 128 
```

### Step 4: multitask training with single task fine-tuning.
```bash
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
```