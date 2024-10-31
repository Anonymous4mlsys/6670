# Quantized Fine-tuning

## Usage of Adapter

training adapter script: `~/Quantized-boosting-code/quantized-finetuning/scripts/train_glue/train_adapter.sh`

Using ```Bottleneck Adapter```

There are two parts in Bottleneck Adapter: mh_adapter, output_adapter.

Make ```reduction_factor``` larger if you want to reduce the number of adapter parameter. 

## Instruction dataset and Alpaca

Data: `~/in-context-learning/chain-of-thought-finetuning/data/alpaca_data`

Link: https://github.com/HazyResearch/skill-it/tree/main/aux_data 

Package: Pytorch Nightly; Pytorch Lightning

```
conda env create -f environment.yml
```

#### Step 1

Train a model on all data: `./scripts/alpaca/train_alpaca.sh`

Use `custom_train_alpaca.py`
```
python custom_train_alpaca.py --model_key $model_key\
    --lr 5e-5 --batch_size 1 --max_length 256 --epochs 10\
    --train_lora --lora_rank 4 --lora_alpha 32\
    --strategy auto --devices 0 1 2 --runs 1 --accumulate 1 --precision "bf16-true" 
```
Main parameters: 
- `model_key`: huggingface directories: `../llama/llama-3/Meta-Llama-3-8B-hf`, `google/gemma-2b`, `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`, `flan_t5_base`, ...
- `save_name`: name for saving the checkpoints
- `load_model_dir`: the name of the directory of the checkpoint to be loaded
- `write_results`: write the loss results to a csv file

#### Step 2

Compute and project gradients on all training samples: `./script/alpaca/eval_approximation_err.sh`

Run `fast_estimate_eval_approximation_alpaca.py` 
```
python fast_estimate_eval_approximation_alpaca.py \
    --model_key $model_key --train_lora --lora_rank 4 --lora_alpha 32 --precision "bf16-true"\
    --batch_size 2 --max_length 256 --project_gradients --project_dimension 100 \
    --devices 0 1 2 --strategy auto --save_name $save_name --compute_pretrained_outputs
```
- Main parameters: 
  - `--compute_pretrained_outputs` will compute the gradients and outputs at the initialization
  - `--model_key`
  - `--save_name`
  - `--downsample`: downsampling the dataset
  - `--num_batches_gradients`: number of batches to compute gradients. `num_batches_gradients` * `batch_size` = `downsample`

Note: Before `load_model_dir`, use `export_hf_model` to save models as huggingface model checkpoints. 

### Step 3: Compute Linear Approximation Error

Estimate linear regression models on subsets of clusters: `./script/alpaca/eval_approximation_err.sh`

Use `fast_estimate_eval_approximation_alpaca.py` 
```
python fast_estimate_eval_approximation_alpaca.py \
    --model_key $model_key --train_lora --lora_rank 4 --lora_alpha 32 --precision "bf16-true"\
    --batch_size 2 --max_length 256 --project_gradients --project_dimension 100\
    --scale 0.2 --seed $seed --devices 0 1 2 --strategy auto
```
Main parameters:
- `--project_gradients` 
- `--project_dimension`
- `--scale`: control the fine-tuned distance
- `--save_name`

```
save_name = ("Instruction_{}".format(model_key) if args.train_instruction else "Alpaca_{}".format(model_key)) + \
                (f"_lora_r_{args.lora_rank}" if args.train_lora else "") + \
                ("_{}".format(args.save_name) if args.save_name != "none" else "")

gradient_dir = save_name + f"_dim_{args.project_dimension}_run_{args.run}" + ("_pretrained" if args.load_model_dir is None else "")
```

Gradients and outputs are saved to `./gradients/{gradient_dir}/`


### Step 4: Estimate Fine-tune Model Parameters

```
python fast_estimate_linear_regression_alpaca.py \
    --model_key $model_key --train_lora --lora_rank 4 --lora_alpha 32 --precision "bf16-true"\
    --batch_size 16 --project_gradients --project_dimension 200 --devices 2\
    --load_model_dir $load_dir\
    --load_sample_task_dir Alpaca_EleutherAI-gpt-neo-125M_lora_r_4 --number_of_subsets 1000 --subset_size 0.5 --save_name $save_name
```

Main parameters: 
- `--load_sample_task_dir`: if specified, will load previously sampled subsets
  - Otherwise, specify `--number_of_subsets` and `--subset_size`
- `--scale`: controls the norm of the fine-tuned model
- `--save_name`

```
save_name = "Alpaca_{}".format(model_key) + \
            (f"_lora_r_{args.lora_rank}" if args.train_lora else "") + \
            (f"_{args.save_name}" if args.save_name else "") + \
            f"_dim_{args.project_dimension}_run_{args.run}" 
file_dir = os.path.join("./results/", save_name)
```

### Other Operations

Collect ground-truth empirical fine-tuned model losses. 

Use `sample_train_results_alpaca.py`
```
python sample_train_results_alpaca.py \
    --model_key $model_key --precision 32 --lr 5e-5\
    --load_sample_task_dir Alpaca_EleutherAI-gpt-neo-125M_lora_r_4 --devices 0 1 2 --subset_num 100\
    --save_name $save_name
```



## Chain-of-thought fine-tuning



## Cookbook

```
# Default used by the Trainer
fabric = Fabric(precision="32-true", devices=1)

# the same as:
trainer = Trainer(precision="32", devices=1)

# 16-bit mixed precision (model weights remain in torch.float32)
trainer = Trainer(precision="16-mixed", devices=1)

# 16-bit bfloat mixed precision (model weights remain in torch.float32)
trainer = Trainer(precision="bf16-mixed", devices=1)

# 8-bit mixed precision via TransformerEngine (model weights get cast to torch.bfloat16)
trainer = Trainer(precision="transformer-engine", devices=1)

# 16-bit precision (model weights get cast to torch.float16)
trainer = Trainer(precision="16-true", devices=1)

# 16-bit bfloat precision (model weights get cast to torch.bfloat16)
trainer = Trainer(precision="bf16-true", devices=1)

# 64-bit (double) precision (model weights get cast to torch.float64)
trainer = Trainer(precision="64-true", devices=1)
```


## Instructions

Official repository for [Large Language Models Are Reasoning Teachers](https://arxiv.org/abs/2212.10071), by
Namgyu Ho, Laura Schmid, and Se-young Yun.

**🚀 Accepted to ACL 2023.**

This repository contains code for (1) running CoT reasoning on OpenAI models,
and (2) apply Fine-tune-CoT to train students based on OpenAI models *or* custom open-source models such as T5, Flan-T5, GPT-2 on your GPUs, based on 🤗 and Pytorch Lightning.


## Getting Started

### OpenAI API Experiments

OpenAI API experiments are implemented in the `oai` module. Refer to `notebooks/example_oai_finetune_cot.ipynb`
on how to run Fine-tune-CoT from start to finish.

### Custom Experiments (on GPU) 

Custom experiments are implemented in the `custom` module, based on PyTorch Lightning. Refer to `custom_train.py`
and `scripts/custom/*.sh` on how to fine-tune models such as T5, Flan-T5, and GPT-2 using Fine-tune-CoT.

## Setup

```
pip install -r requirements.txt
python setup.py develop
```

### Environment

The code has been tested on Python<=3.10, PyTorch Lightning<=1.9, PyTorch>=2.0

## Data 🚀

We're proud to share *all* of our raw experimental data! All data is organized in json or jsonl format, for your pleasure :)

Cloud storage folder links:

- [Dropbox](https://www.dropbox.com/sh/hwcncpyomx87h20/AACqgVdd-ZzBQ3ncJcKqw0cVa?dl=0)
- [Google Drive](https://drive.google.com/drive/folders/1C6kah3WV36N8omlUl-TeU9tsJADZNaJV?usp=share_link)

### File List

- `dataset.tar.gz`: 12 task datasets compiled in a unified json format
  - Belongs in `PROJECT/data/dataset/`
- `completion_data.tar.gz`: Completion data, i.e., inference data, from all teachers and students, for *all* experiments. About 8GB when uncompressed
  - Belongs in `PROJECT/saved/completion_data/`
- `teacher_completion_data.tar.gz`: Completion data from Zero-shot-CoT (with diverse reasoning) on the default teacher model `text-davinci-002` using the OpenAI API. About 💰 $1000+ worth of goods, with ❤️ from [OSI LAB](http://osi.kaist.ac.kr) at [KAIST](https://kaist.ac.kr) . Subset of `completion_data.tar.gz`.
  - Belongs in `PROJECT/saved/completion_data/`.
- `finetune_data.tar.gz`: *All* data used to fine-tune OpenAI students via the fine-tuning API, in jsonl format. These are derived from teacher completion data and can be generated from our code.
  - Belongs in `PROJECT/saved/finetune_data/`

### Generate Paper Results

After downloading the full `completion_data.tar.gz`, you can run `notebooks/results.ipynb` to generate *all* result tables and figures from our paper. The code will (re-)evaluate all raw text model outputs contained in the completion data.



## Additional Resources

### Template-based Split (Paper Appendix E.3)

Template-based splits for MultiArith and Date Understanding are saved in `/data/splits/*__template.json`

### Few-shot Prompts

Few-shot prompts adapted from Wei 2022 are saved in `/data/few_shot_cot_prompts.json`



## Data Structures

### `data.dataset.Dataset`

```json
{
  "metadata": {
    "dataset_key": "multiarith"
  },
  "data": [
    {
      "sample_index": 0,
      "question": "string",
      "answer": "string",
      "rationale": "string?"
    }
  ]
}
```

### `data.completion.CompletionDataset`

```json
{
  "metadata": {
    "dataset_key": "multiarith",
    "base_model": "curie",
    "finetune_key": "zs_cot_multiarith",
    "train_key": "ft_cot",
    "prediction_template": "ft_cot_token",
  },
  "data": {
    "<sample_index>": [
      {
        "sample_index": 0,
        "completion_index": 0,
        "question": "string",
        "answer": "string",
        "prompt": "string",
        "completion": "string",
        "finish_reason": "string",
        "reasoning_prompt": "string?",
        "reasoning_completion": "string?",
        "reasoning_finish_reason": "string?",
      }
    ]
  }
}
```



## Data Organization

*Needs update.*

- `<model_key>` = `B_<base_model>_T_<train_key>`

### File Organization Pattern

```
saved/
|–– completion_data/
    |–– B_<BASE_MODEL>__C_<COMPLETION_KEY>/
        |-- D_<DATESET_KEY>.json  # base model inference
        |-- F_<FINETUNE_KEY>__D_<DATESET_KEY>.json  # default fine-tuned model inference
        |-- F_<FINETUNE_KEY>__T_<TRAIN_KEY>__D_<DATESET_KEY>.json  # custom fine-tuned model inference
|–– finetune_data/
    |–– P_<PLATFORM_KEY>/
        |–– F_<FINETUNE_KEY>{.*|/}
|–– model_metadata/
    |–– B_<base_model>
        |–– F_<FINETUNE_KEY>__T_<train_key>.json
```

### File Organization Examples

```
saved/
|–– completion_data/
    |–– B_text-davinci-002__C_zs_cot/
    |–– B_text-davinci-002__C_zs_cot_long/
    |–– B_text-davinci-002__C_fs_cot/
    |–– B_curie__C_zs_cot/
    |–– B_curie__C_fs_cot/
    |–– B_curie__C_zs/
    |–– B_curie__C_ft_cot/
|–– finetune_data/
    |–– F_zs_cot_multiarith/  # text-davinci-002_zs_cot
    |–– F_zs_cot_long_multiarith/
|–– model_metadata/
    |–– B_curie/
        |–– F_zs_cot_multiarith.json
```


### Personal Note

![accepted](acl2023.jpg)

