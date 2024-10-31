# %%
import argparse
import logging
import os
import wandb

from src.custom.glue_data_module import GLUEDataModule
from src.custom.glue_model import GLUEModel
from src.lqlora_utils import lora_utils

from functools import partial
from pytorch_lightning.strategies import FSDPStrategy
from pytorch_lightning.trainer.states import RunningStage, TrainerFn
from src.merging_utils.ensemble import EnsembleModule, MaxModelPredictor

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pytorch_lightning as pl
import torch
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from peft import get_peft_model, LoraConfig
from pytorch_lightning.callbacks import ModelCheckpoint
import pandas as pd
from collections import defaultdict
import time

from torch._inductor.async_compile import AsyncCompile

logging.basicConfig(level=logging.INFO, force=True)
torch.set_float32_matmul_precision("high")  

def initialize_model(args):
    if "gpt" in args.model_key or "Llama" in model_key \
        or "bloomz" in model_key or "gemma" in model_key or "Mistral" in model_key:
        hf_key = args.model_key.replace("_", "-")
        tokenizer = AutoTokenizer.from_pretrained(hf_key)
        if args.use_qlora:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
                )
            model = AutoModelForCausalLM.from_pretrained(hf_key, quantization_config=quantization_config, torch_dtype=torch.bfloat16, device_map={"": args.devices[0]}) #
        else:
            model = AutoModelForCausalLM.from_pretrained(hf_key)
        model_type = "decoder"
        append_eos = True
    elif "flan" in model_key:
        hf_key = "google/{}".format(model_key.replace("_", "-"))
        model = AutoModelForSeq2SeqLM.from_pretrained(hf_key)
        tokenizer = AutoTokenizer.from_pretrained(hf_key, model_max_length=512)
        model_type = "encoder_decoder"
        append_eos = False  # t5 tokenizers already append eos
    else:
        raise NotImplementedError(args.model_key)
    
    if args.use_3bit or args.use_2bit:
        model = lora_utils.prepare_model_for_lora(
            model=model,
            num_ranks=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.1,
            use_gradient_checkpointing=True)

        lora_utils.transform_lora_layers(
            lpq=False,
            model=model,
            model_name="nf3" if args.use_3bit else "nf2",
            device=f"cuda:{args.devices[0]}")
        model.to(f"cuda:{args.devices[0]}")        

    elif args.train_lora:
        if args.model_key == "gpt2": # for gpt2, we generally use full model
            config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                target_modules=["c_attn", "c_proj", "c_fc"],
                lora_dropout=0.1,
                bias="lora_only",
                modules_to_save=[],
            )
        elif args.model_key == "EleutherAI/gpt-neox-20b":
            config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                target_modules=["query_key_value"],
                lora_dropout=0.1,
                bias="lora_only",
                modules_to_save=[],
            )
        elif "flan" in args.model_key:
            config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                target_modules=["q", "k", "v"],
                lora_dropout=0.1,
                bias="lora_only",
                modules_to_save=[],
            )
        else:
            config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj"],
                lora_dropout=0.1,
                bias="lora_only",
                modules_to_save=[],
            )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()

    return model, tokenizer, hf_key, model_type, append_eos

# %%
class args:
    task_name = "rte"
    model_key = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    batch_size = 8
    inference_batch_size = 8
    devices = [0]
    accumulate = 1
    strategy = "auto"
    precision = "bf16-true"
    lr = 3e-4
    weight_decay = 0
    epochs = 10
    max_length = 256
    save_every_epoch = False
    downsample = None
    optimizer = "adamw"
    use_qlora = False
    use_3bit = False
    use_2bit = False

    train_lora = True
    lora_rank = 16
    lora_alpha = 128

    save_name = None
    load_model_dir = "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_rte_lora_r_16_adaboost_iteration_2_run_0/epoch_epoch=1.pt"
    write_results = False
    use_wandb = False
    generate_output = False 

    merge_strategy = "simple_ensemble"
    merge_model_dirs = None
    # [
    # "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_rte_lora_r_16_pairwise_run_0/epoch_epoch=3.pt"
    # "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_rte_lora_r_16_adaboost_iteration_1_run_0/epoch_epoch=1.pt"\
    # "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_rte_lora_r_16_adaboost_iteration_2_run_0/epoch_epoch=1.pt"\
    # ]
    use_sample_weights_dir = "adaboost_rte_weights_iteration_2.npy"


# %%
model_key = args.model_key.replace("/", "-").replace("..", "")
save_name = (f"_{args.save_name}" if args.save_name else "") + \
            (f"_lora_r_{args.lora_rank}" if args.train_lora else "")         

metrics = {}
model, tokenizer, hf_key, model_type, append_eos = initialize_model(args)
load_model_dir = os.path.join("external_lightning_logs", args.load_model_dir)
checkpoint = torch.load(load_model_dir, map_location="cpu")
model.load_state_dict(checkpoint, strict=False)
if args.merge_model_dirs is not None:   
    # load the checkpoints
    if "ensemble" in args.merge_strategy:
        models = []
        for merge_model_dir in args.merge_model_dirs:
            merge_model_dir = os.path.join("external_lightning_logs", merge_model_dir)
            checkpoint = torch.load(merge_model_dir, map_location="cpu")
            model, _, _, _, _ = initialize_model(args)
            model.load_state_dict(checkpoint, strict=False)
            models.append(model)
        model = EnsembleModule(models) if "simple" in args.merge_strategy else MaxModelPredictor(models)
    else:
        raise NotImplementedError(args.merge_strategy)
    
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

batch_size = args.batch_size
if args.inference_batch_size is None:
    inference_batch_size = batch_size
else:
    inference_batch_size = args.inference_batch_size

data_module = GLUEDataModule(
        task_name=args.task_name,
        tokenizer=tokenizer,
        batch_size=batch_size,
        inference_batch_size=inference_batch_size,
        max_input_length=args.max_length)
data_module.setup(stage="fit")

if args.use_sample_weights_dir is not None:
    sample_weights_dir = os.path.join("weights", args.use_sample_weights_dir)
    if os.path.exists(sample_weights_dir):
        weights = np.load(sample_weights_dir)
        data_module.load_weights(weights)
        print(f"Loaded sample weights from {sample_weights_dir}!")

answer_choices = data_module.template.answer_choices.split("|||")
# process the answer choices, different models tokenize them differently
if "gpt" in args.model_key: 
    answer_choices = [" " + choice.strip() for choice in answer_choices] 
    answer_choices = [tokenizer([choice])["input_ids"][0][0] for choice in answer_choices]; answer_choices.sort()
elif "Llama-3" in args.model_key:
    answer_choices = [" " + choice.strip() for choice in answer_choices] 
    answer_choices = [tokenizer([choice])["input_ids"][0][1] for choice in answer_choices]; answer_choices.sort()
elif "TinyLlama" in args.model_key:
    answer_choices = [choice.strip() for choice in answer_choices]
    answer_choices = [tokenizer([choice])["input_ids"][0][1] for choice in answer_choices]; answer_choices.sort()
else:
    answer_choices = [" " + choice.strip() for choice in answer_choices] 
    answer_choices = [tokenizer([choice])["input_ids"][0][0] for choice in answer_choices]; answer_choices.sort()
lm = GLUEModel(model, tokenizer, model_type, use_cpu_offload=False,
                lr=args.lr, weight_decay=args.weight_decay, max_length=args.max_length, use_wandb=args.use_wandb, 
                answer_choices=answer_choices, optimizer=args.optimizer, generate_output=args.generate_output)


# %%

default_root_dir = os.path.join("external_lightning_logs", 
                                f"adaboost_{model_key}_" + \
                                args.task_name + \
                                (f"_lora_r_{args.lora_rank}" if args.train_lora else "") + \
                                (f"_{args.save_name}" if args.save_name else "")
                                )
checkpoint_callback = ModelCheckpoint(
    monitor="loss",
    dirpath=default_root_dir,
    filename="epoch_{epoch}",
    save_top_k=(-1 if args.save_every_epoch else 1),
    mode="min",
)

trainer = pl.Trainer(accelerator="gpu", devices=args.devices, strategy=args.strategy,
                    default_root_dir=default_root_dir, min_epochs=args.epochs, max_epochs=args.epochs,
                    accumulate_grad_batches=args.accumulate, precision=args.precision,
                    enable_checkpointing=True,
                    callbacks=[checkpoint_callback]
                    )

# %%
outputs = trainer.predict(lm, dataloaders=data_module.train_dataloader())


# %%
from sklearn.metrics import accuracy_score, f1_score
def aggregate_predictions(outputs):
    losses = [output["loss"] for output in outputs]
    losses = torch.stack(losses)
    losses = losses[torch.isnan(losses) == False]
    summary = {"loss": losses.mean().item()}
    summary["accuracy_score"] = 0; summary["f1_score"] = 0; summary["error"] = 0

    counts = 0; sum_weights = 0; masks = []
    for batch in outputs:        
        if len(batch["label_ids"]) == 0:
            continue
        summary["accuracy_score"] += accuracy_score(batch["label_ids"], batch["pred_ids"])*len(batch["label_ids"])*100
        summary["f1_score"] += f1_score(batch["label_ids"], batch["pred_ids"], average="macro")*len(batch["label_ids"])*100
        
        mask = batch["masks"].clone()
        mask[mask>0] = ~torch.Tensor(batch["label_ids"] == batch["pred_ids"]).type(torch.bool).view(-1)
        summary["error"] += batch["weights"][mask].sum()
        sum_weights += batch["weights"].sum()
        masks.append(mask)
        counts += len(batch["label_ids"])
    
    for key in summary:
        if key == "error":
            summary[key] = summary[key]/sum_weights
        elif key != "loss":
            summary[key] = (summary[key]/counts) if counts > 0 else 0

    masks = torch.cat(masks)
    # Log metrics
    return summary, masks
summary, masks = aggregate_predictions(outputs)

print(summary)

# %%
M = 3
error = summary["error"]
alpha = np.math.log((1-error)/error)
data_module.update_weights(alpha, masks)

np.save(f"weights/adaboost_{data_module.task_name}_alpha_iteration_{M}.npy", alpha)
np.save(f"weights/adaboost_{data_module.task_name}_weights_iteration_{M}.npy", data_module.weights)

# %%
import numpy as np
for M in range(1, 4):
    print(np.load(f"weights/adaboost_rte_alpha_iteration_{M}.npy"))

3.3141860046725258
1.2256119794294171
-3.1997723543025884