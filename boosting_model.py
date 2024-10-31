# %% 
import torch
import torch.nn as nn
import torch.nn.functional as F

from peft import get_peft_model, LoraConfig
from src.lqlora_utils import lora_utils

import argparse
import logging
import os
import wandb

from src.custom.glue_multitask_model import GLUEMultitaskModel
from src.custom.glue_multitask_data_module import GLUEMultitaskDataModule
from src.lqlora_utils import lora_utils

from functools import partial
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, lambda_auto_wrap_policy, transformer_auto_wrap_policy, _or_policy
from pytorch_lightning.strategies import FSDPStrategy
from pytorch_lightning.trainer.states import RunningStage, TrainerFn

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pytorch_lightning as pl
import torch
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoBlock
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from torch.nn import Embedding

from peft import get_peft_model, LoraConfig
from pytorch_lightning.callbacks import ModelCheckpoint
import pandas as pd
from collections import defaultdict
import time

from torch._inductor.async_compile import AsyncCompile

def initialize_model(args):
    model_key = args.model_key.replace("/", "-").replace("..", "")
    if "gpt" in model_key or "Llama" in model_key \
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
    task_names = ["cb", "rte"] 
    model_key = "meta-llama/Llama-3.2-1B"
    batch_size = 4
    inference_batch_size = 8
    devices = [0]
    accumulate = 1
    strategy = "auto"
    precision = "bf16-true"
    lr = 1e-5
    weight_decay = 0
    epochs = 1
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
    load_model_dirs = [
        "meta-llama-Llama-3.2-1B_cb_multirc_lora_r_16_task_grouping_2_run_0/epoch_epoch=3.pt",
        "meta-llama-Llama-3.2-1B_rte_copa_wic_wsc.fixed_boolq_winogrande_debiased_story_cloze_hellaswag_lora_r_16_task_grouping_2_run_0/epoch_epoch=7.pt",
    ]
    # "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_cb_rte_lora_r_16_pairwise_run_0/epoch_epoch=3.pt"
    # "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_cb_lora_r_16_pairwise_run_0/epoch_epoch=6.pt"
    write_results = False
    use_wandb = False
    generate_output = False 

    n_estimators = 10

# %%

# Initialize the model
model_key = args.model_key.replace("/", "-").replace("..", "")
save_name = (f"_{args.save_name}" if args.save_name else "") + \
            (f"_lora_r_{args.lora_rank}" if args.train_lora else "")         
metrics = {}
model, tokenizer, hf_key, model_type, append_eos = initialize_model(args)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# initialize the data module
batch_size = args.batch_size
if args.inference_batch_size is None:
    inference_batch_size = batch_size
else:
    inference_batch_size = args.inference_batch_size

data_module = GLUEMultitaskDataModule(
        task_names=args.task_names,
        tokenizer=tokenizer,
        batch_size=batch_size,
        inference_batch_size=inference_batch_size,
        max_input_length=args.max_length)
data_module.setup(stage="fit")

task_answer_choices = {}
for task_name in args.task_names:
    answer_choices = data_module.task_to_templates[task_name].answer_choices.split("|||")
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
    task_answer_choices[task_name] = answer_choices


def get_lora_state_dict(model):
    state_dict = {}
    for name, weight in model.state_dict().items():
        if "lora" in name:
            state_dict[name] = weight.detach().clone().to(f"cuda:{args.devices[0]}")
    return state_dict

initial_lora_state_dict = get_lora_state_dict(model)

lm = GLUEMultitaskModel(model, tokenizer, model_type, use_cpu_offload=False,
                lr=args.lr, weight_decay=args.weight_decay, max_length=args.max_length, use_wandb=args.use_wandb, 
                optimizer=args.optimizer, generate_output=args.generate_output, task_names=args.task_names, task_answer_choices=task_answer_choices)

if not os.path.exists("external_lightning_logs"):
    raise Exception("external_lightning_logs/ does not exist")
default_root_dir = os.path.join("external_lightning_logs", 
                            f"gradient_boosting_{model_key}_" + \
                            "_".join(args.task_names) + \
                            (f"_lora_r_{args.lora_rank}" if args.train_lora else "") + \
                            (f"_{args.save_name}" if args.save_name else "") )

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
def aggregate_predictions_adaboost(outputs):
    error = 0; sum_weights = 0; masks = []
    for batch in outputs:        
        # skipping this batch if no labels are present
        if len(batch["label_ids"]) == 0:
            masks.append(batch["masks"].clone())
            continue

        mask = batch["masks"].clone()

        # mask indicates where the prediction is incorrect
        mask[mask>0] = ~torch.Tensor(batch["label_ids"] == batch["pred_ids"]).type(torch.bool).view(-1)
        error += batch["weights"][mask].sum()
        sum_weights += batch["weights"].sum()
        masks.append(mask)
    
    error = error/sum_weights # the sum of weights on the incorrect predictions over the sum of all weights
    masks = torch.cat(masks)
    return error, masks

def aggregate_predictions_gradient_boosting(outputs):

    def onehot_encoding(label, n_classes):
        """Conduct one-hot encoding on a label vector."""
        label = label.view(-1)
        onehot = torch.zeros(label.size(0), n_classes).float().to(label.device)
        onehot.scatter_(1, label.view(-1, 1), 1)

        return onehot

    all_residuals = []; masks = [] # residuals per sample (if multiple positions in one sample, we sum them up)
    for batch in outputs:
        labels = batch["labels"][:, 1:].contiguous()
        correct_class_probs = batch["probs"]
        mask = batch["masks"].clone()
        residuals = np.zeros(len(labels))

        is_label_mask = labels != -100
        valid_labels = labels[is_label_mask].view(-1)
        residuals[mask>0] = (1 - correct_class_probs).cpu().numpy()
        
        all_residuals.append(residuals)
        masks.append(mask)
    all_residuals = np.concatenate(all_residuals, axis=0)
    masks = torch.cat(masks)
    return all_residuals, masks

def aggregate_metrics(outputs, task_names):
    from sklearn.metrics import accuracy_score, f1_score
    """
    Gather outputs from all GPUs and save validation predictions as a CompletionDataset and
    log validation metrics.

    Note, `all_gather` *concatenates* tensors from all GPUs along the first dimension.
    """
    summary = {f"{task_name}_loss": 0 for task_name in task_names}
    summary.update({f"{task_name}_accuracy_score": 0 for task_name in task_names})
    summary.update({f"{task_name}_f1_score": 0 for task_name in task_names})
    
    # average loss
    losses = [output["loss"] for output in outputs]
    losses = torch.stack(losses)
    losses = losses[torch.isnan(losses) == False]
    summary.update({"loss": losses.mean().item()})

    task_counts = {task_name: 0 for task_name in task_names}
    for batch in outputs:
        task_name = batch["task_name"]
        if len(batch["label_ids"]) == 0:
            continue
        summary[f"{task_name}_loss"] += batch["loss"].item()*len(batch["label_ids"]) if torch.isnan(batch["loss"]) == False else 0
        summary[f"{task_name}_accuracy_score"] += accuracy_score(batch["label_ids"], batch["pred_ids"])*len(batch["label_ids"])*100
        summary[f"{task_name}_f1_score"] += f1_score(batch["label_ids"], batch["pred_ids"], average="macro")*len(batch["label_ids"])*100
        task_counts[task_name] += len(batch["label_ids"])
    
    for task_name in task_names:
        summary[f"{task_name}_loss"] = (summary[f"{task_name}_loss"]/task_counts[task_name]) if task_counts[task_name] > 0 else 0
        summary[f"{task_name}_accuracy_score"] = (summary[f"{task_name}_accuracy_score"]/task_counts[task_name]) if task_counts[task_name] > 0 else 0
        summary[f"{task_name}_f1_score"] = (summary[f"{task_name}_f1_score"]/task_counts[task_name]) if task_counts[task_name] > 0 else 0

    # average accuracy and f1 score
    summary.update({"accuracy_score": np.mean([summary[f"{task_name}_accuracy_score"] for task_name in task_names])})
    summary.update({"f1_score": np.mean([summary[f"{task_name}_f1_score"] for task_name in task_names])})

    # Log metrics
    print(summary)
    return summary


from lightning_fabric.utilities.cloud_io import _load as pl_load
def save_trained_model(checkpoint_dir, appendix):
    checkpoint = pl_load(checkpoint_dir, map_location=lm.device)
    state_dict = checkpoint["state_dict"]
    state_dict = {k[6:]: v for k, v in state_dict.items() if "lora" in k}
    save_path_dir = checkpoint_dir.replace(".ckpt", "") + appendix + ".pt"
    torch.save(state_dict, save_path_dir)


# %%
from src.merging_utils.ensemble import EnsembleAdapterModule

''' First fit a naive MTL model '''
lm.fit_least_square = False
# trainer.fit(lm, data_module)
''' First fit a naive MTL model '''

load_model_dirs = []
for load_model_dir in args.load_model_dirs:
    load_model_dir = os.path.join("external_lightning_logs", load_model_dir)
    # checkpoint = torch.load(load_model_dir, map_location="cpu")
    # model.load_state_dict(checkpoint, strict=False)
    load_model_dirs.append(load_model_dir)
weights = [1.0/len(load_model_dirs) for _ in range(len(load_model_dirs))]

# only using the ensemble model for evaluation now
_ensemble_model = EnsembleAdapterModule(model, load_model_dirs, weights)
ensemble_model = GLUEMultitaskModel(_ensemble_model, tokenizer, model_type, use_cpu_offload=False,
                lr=args.lr, weight_decay=args.weight_decay, max_length=args.max_length, use_wandb=args.use_wandb, 
                optimizer=args.optimizer, generate_output=args.generate_output, task_names=args.task_names, task_answer_choices=task_answer_choices)
trainer.validate(ensemble_model, dataloaders=data_module.val_dataloader())

# # %%
# ''' Get training error '''
# outputs = trainer.predict(ensemble_model, dataloaders=data_module.train_dataloader())
# aggregate_metrics(outputs, data_module.task_names)

# %%
''' Gradient Boosting '''
gradient_boosting_lr = 0.1
args.n_estimators = 1
for est_idx in range(args.n_estimators):

    # Compute residuals: use the correct class probabilities to compute the residuals
    outputs = trainer.predict(ensemble_model, dataloaders=data_module.train_dataloader())
    aggregate_metrics(outputs, data_module.task_names)
    residuals, masks = aggregate_predictions_gradient_boosting(outputs)
    data_module.update_residuals(residuals, masks)

    # Train a new model
    lm.lr = 5e-4
    lm.fit_least_square = True
    model.load_state_dict(initial_lora_state_dict, strict=False) # re-initialize the adapter weights
    checkpoint_callback = ModelCheckpoint(
        monitor="loss",
        dirpath=default_root_dir,
        filename="epoch_{epoch}",
        save_top_k=(-1 if args.save_every_epoch else 1),
        mode="min",
    ) # initialize a new checkpoint callback
    trainer = pl.Trainer(accelerator="gpu", devices=args.devices, strategy=args.strategy,
                    default_root_dir=default_root_dir, min_epochs=args.epochs, max_epochs=args.epochs,
                    accumulate_grad_batches=args.accumulate, precision=args.precision,
                    enable_checkpointing=True,
                    callbacks=[checkpoint_callback]
                    )
    trainer.fit(lm, data_module)

    # Save the trained model and write an ensemble to evaluate the predictions
    task_names_str = "_".join(args.task_names)
    appendix = f"_gradient_boosting_{task_names_str}_iteration_{est_idx}"
    checkpoint_dir = checkpoint_callback.best_model_path.replace(".ckpt", "") + appendix + ".pt"
    save_trained_model(checkpoint_callback.best_model_path, appendix)
    os.remove(checkpoint_callback.best_model_path)

    _ensemble_model.add_adapter(checkpoint_dir, gradient_boosting_lr)
    trainer.validate(ensemble_model, dataloaders=data_module.val_dataloader())

# %%
args.n_estimators = 5
for est_idx in range(args.n_estimators):
    # Compute the prediction errors of the current ensemble model
    outputs = trainer.predict(ensemble_model, dataloaders=data_module.train_dataloader())
    aggregate_metrics(outputs, data_module.task_names)
    error, masks = aggregate_predictions_adaboost(outputs)
    alpha = np.math.log((1-error)/error)
    data_module.update_weights(alpha, masks)

    # Train a new model
    lm.use_sample_weights = True
    checkpoint_callback = ModelCheckpoint(
        monitor="loss",
        dirpath=default_root_dir,
        filename="epoch_{epoch}",
        save_top_k=(-1 if args.save_every_epoch else 1),
        mode="min",
    ) # initialize a new checkpoint callback
    trainer = pl.Trainer(accelerator="gpu", devices=args.devices, strategy=args.strategy,
                    default_root_dir=default_root_dir, min_epochs=args.epochs, max_epochs=args.epochs,
                    accumulate_grad_batches=args.accumulate, precision=args.precision,
                    enable_checkpointing=True,
                    callbacks=[checkpoint_callback]
                    )
    trainer.fit(lm, data_module)
    
    # Save the trained model and write an ensemble to evaluate the predictions
    task_names_str = "_".join(args.task_names)
    appendix = f"_adaboosting_{task_names_str}_iteration_{est_idx}"
    checkpoint_dir = checkpoint_callback.best_model_path.replace(".ckpt", "") + appendix + ".pt"
    save_trained_model(checkpoint_callback.best_model_path, appendix)
    os.remove(checkpoint_callback.best_model_path)

    _ensemble_model.add_adapter(checkpoint_dir, alpha)
    trainer.validate(ensemble_model, dataloaders=data_module.val_dataloader())

# %%
''' Linear Approximation: linear regression '''
def generate_state_dict(model, state_dict, coef, device="cpu", removing_keys = ["shared", "lm_head", "wte", "wpe", "ln", "embed_tokens", "norm", "word_embeddings"]):
    new_state_dict = {}; cur_len = 0
    for key, param in model.named_parameters():
        if not param.requires_grad: continue
        param_len = param.numel()
        if any([rkey in key for rkey in removing_keys]):
            continue
            # new_state_dict[key] = state_dict[key].clone()
        else:
            assert "lora" in key
            new_state_dict[key] = state_dict[key].clone().to(device) + \
                torch.FloatTensor(coef[cur_len:cur_len+param_len].reshape(param.shape)).to(device)
            cur_len += param_len
    return new_state_dict


# Before estimation
tasks_to_lengths = [len(task_list) for task_list in data_module.multitask_train_sampler._train_data_list]
tasks_to_indices = [list(range(sum(tasks_to_lengths[:i]), sum(tasks_to_lengths[:i+1]))) for i, task_name in enumerate(args.task_names)]

# load gradients
gradients_dir = "compute_gradients_TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_lora_r_16_dim_200_seed_0_pretrained"
task_idxes = [0]
gradient_idxes = []
for task_idx in task_idxes:
    gradient_idxes += tasks_to_indices[task_idx]

gradients = []; outputs = []
for idx in gradient_idxes:
    if os.path.exists(f"./gradients/{gradients_dir}/train_batch_{idx}_gradients.npy"):
        gradients.append(np.load(f"./gradients/{gradients_dir}/train_batch_{idx}_gradients.npy"))
        outputs.append(np.load(f"./gradients/{gradients_dir}/train_batch_{idx}_outputs.npy")[:, 0])
gradients = np.concatenate(gradients, axis=0)
pretrained_outputs = np.concatenate(outputs, axis=0)

# %%
np.random.seed(0)
gradient_dim = 0; project_gradients_dim = 200
removing_keys = ["shared", "lm_head", "wte", "wpe", "ln", "embed_tokens", "norm", "word_embeddings" ]
for name, param in model.named_parameters():
    if any([key in name for key in removing_keys]):
        continue
    if param.requires_grad:
        gradient_dim += param.numel()
print("Creating project matrix with dimensions: ", gradient_dim, project_gradients_dim)
project_matrix = (2 * np.random.randint(2, size=(gradient_dim, project_gradients_dim)) - 1).astype(float)
project_matrix *= 1 / np.sqrt(project_gradients_dim)

# %%
def solve_linear_regression(gradients, outputs, residuals, scale=10):
    from sklearn.linear_model import LinearRegression, Ridge

    y = - (residuals + outputs)
    reg = Ridge().fit(gradients, y)

    print("R^2 score", reg.score(gradients, y))

    coef = reg.coef_
    coef = project_matrix @ coef.flatten()
    print("L2 norm before scaling", np.linalg.norm(coef))
    if scale is not None:
        coef = coef*scale / np.linalg.norm(coef)
    print("L2 norm after scaling", np.linalg.norm(coef))

    # evaluate task performances
    new_state_dict = generate_state_dict(lm.model, initial_lora_state_dict, coef, device=lm.model.device)
    return new_state_dict

# %%
''' Gradient Boosting w/ approximation '''
gradient_boosting_lr = 0.5
args.n_estimators = 10
for est_idx in range(args.n_estimators):

    # Compute residuals: use the correct class probabilities to compute the residuals
    outputs = trainer.predict(ensemble_model, dataloaders=data_module.train_dataloader())
    aggregate_metrics(outputs, data_module.task_names)
    residuals, masks = aggregate_predictions_gradient_boosting(outputs)
    data_module.update_residuals(residuals, masks)
    print("Avg. Residuals: ", residuals.mean())

    # Train a new model
    lm.fit_least_square = True
    new_state_dict = solve_linear_regression(gradients, pretrained_outputs, residuals, scale=2)

    # Save the trained model and write an ensemble to evaluate the predictions
    task_names_str = "_".join(args.task_names)
    checkpoint_dir = "./" + default_root_dir + f"/gradient_boosting_w_linear_approximation_{task_names_str}_iteration_{est_idx}.pt"
    torch.save(new_state_dict, checkpoint_dir)   
    _ensemble_model.add_adapter(checkpoint_dir, gradient_boosting_lr)
    trainer.validate(ensemble_model, dataloaders=data_module.val_dataloader())
