# %%
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.lqlora_utils import lora_utils
import torch


class args:
    model_key = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T" #  "../llama/llama-3/Meta-Llama-3-8B-hf" #  # "google/gemma-2b" # "EleutherAI/gpt-neo-1.3B" # "mistralai/Mistral-7B-v0.3" #
    train_lora = True
    lora_rank = 16
    lora_alpha = 128

    max_length = 256
    save_model_dir = "./exported_model/cb_lora_r_16_meta_train_run_0_epoch_4"
    device = 0

    task_name = "cb"
    batch_size = 8

# %%
from src.custom.glue_data_module import GLUEDataModule
tokenizer = AutoTokenizer.from_pretrained(args.model_key)
tokenizer.pad_token = tokenizer.eos_token

data_module = GLUEDataModule(
        task_name=args.task_name,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        inference_batch_size=args.batch_size,
        max_input_length=args.max_length)
data_module.setup(stage="fit")

# %%
import pytorch_lightning as pl
from src.custom.glue_model import GLUEModel
from pytorch_lightning.trainer.states import TrainerFn
def evaluate(model, tokenizer):
    answer_choices = data_module.template.answer_choices.split("|||")
    answer_choices = [choice.strip() for choice in answer_choices]
    answer_choices = [tokenizer([choice])["input_ids"][0][1] for choice in answer_choices]; answer_choices.sort()
    lm = GLUEModel(model, tokenizer, "decoder", use_cpu_offload=False,
            max_length=args.max_length, 
            answer_choices=answer_choices)

    trainer = pl.Trainer(accelerator="gpu", devices=[0], strategy="auto",
        default_root_dir="external_lightning_logs/test", min_epochs=0, max_epochs=0,
        accumulate_grad_batches=1, precision="32",
        enable_checkpointing=True,
        )
    trainer.validate_loop.trainer_fn = TrainerFn.FITTING
    trainer.validate_loop.inference_mode = False

    summary = trainer.validate(lm, datamodule=data_module)[0]
    return summary

# %%
import torch
from peft.tuners import lora
from typing import List, Optional, Union, Dict, Any, cast
from src.lqlora_utils.lora_utils import transform_lora_layer, transform_lora_layer_lpq, assert_lora_Linear_layer

def apply_quantization_config(
    lpq: bool,
    model,
    num_iterations: int = 100,
    num_oversampling: int = 10,
    randomized: bool = True,
    device = None,
    qconfig_dict = {}
) -> None:
    for name, submodule in model.named_modules():

        # This implicitly assumes that `LoraLayer`
        # do not include `LoraLayer` within the module.
        if isinstance(submodule, lora.LoraLayer):

            # These operations will be too slow on CPU
            if device is not None:
                # This is in-place
                submodule.to(device=device)

            assert_lora_Linear_layer(submodule)
            qconfig = qconfig_dict[name]

            if lpq is False:
                # print(f"{name:<50}\tqconfig={qconfig}")
                transform_lora_layer(
                    submodule,
                    qconfig=qconfig)
            else:
                num_ranks = cast(
                    int,
                    submodule.r[submodule.active_adapter])
                # Possibly empty
                sensitivity = None

                transform_lora_layer_lpq(
                    submodule,
                    num_ranks=num_ranks,
                    num_iterations=num_iterations,
                    num_oversampling=num_oversampling,
                    randomized=randomized,
                    qconfig=qconfig,
                    W=sensitivity,
                    heuristic="two-sided")
        else:
            if hasattr(submodule, "weight"):
                submodule.to(device=device)


# %%
import torch
from lqlora_utils.quantization_utils import QuantConfig

outputs = torch.load("artifacts/tinyllama.ilp.ranks-None.data-False.pth")
names = outputs[0]['names']
bit_configs = {
    "nf2": QuantConfig(
                num_bits=2,
                num_bits_0=8,
                num_bits_1="fp32",
                block_size_0=64,
                block_size_1=256), 
    "nf3": QuantConfig(
                num_bits=3,
                num_bits_0=8,
                num_bits_1="fp32",
                block_size_0=64,
                block_size_1=256), 
    "nf4": QuantConfig(
                num_bits=4,
                num_bits_0=8,
                num_bits_1="fp32",
                block_size_0=64,
                block_size_1=256)
}

def generate_config(qconfig_dict):
    new_qconfig_dict = {}
    for name in names:
        new_qconfig_dict[name] = bit_configs[qconfig_dict[name]]
    return new_qconfig_dict

# %%
def compute_average_bit(qconfig_dict):
    total = 0
    for name in qconfig_dict.keys():
        total += bit_configs[qconfig_dict[name]].num_bits
    return total / len(qconfig_dict)

# %%
# initialize qconfig_dict
names = ["base_model.model.model.layers." + name.replace(".weight", "") for name in names]
layers = [str(int(l)) for l in range(22)]
qconfig_dict = {name: "nf2" for name in names}

records = []
for i in range(22):
    name_to_scores = {}
    for layer in layers:
        name = f"base_model.model.model.layers.{layer}.self_attn.q_proj"
        if qconfig_dict[name] == "nf3":
            continue
        qconfig_dict[name] = "nf3"
        qconfig_dict[name.replace("q_proj", "k_proj")] = "nf3"
        qconfig_dict[name.replace("q_proj", "v_proj")] = "nf3"
        qconfig_dict[name.replace("q_proj", "o_proj")] = "nf3"
        qconfig_dict[name.replace("q_proj", "gate_proj")] = "nf3"
        qconfig_dict[name.replace("q_proj", "up_proj")] = "nf3"
        qconfig_dict[name.replace("q_proj", "down_proj")] = "nf3"

        # load model
        model = AutoModelForCausalLM.from_pretrained(args.model_key)

        # Adds LoRA components, etc
        model = lora_utils.prepare_model_for_lora(
            model=model,
            num_ranks=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.1,
            use_gradient_checkpointing=True)

        apply_quantization_config(
            lpq=False,
            model=model,
            device="cuda",
            qconfig_dict=generate_config(qconfig_dict))

        # load checkpoint dir
        model.load_state_dict(torch.load(args.save_model_dir + ".pt", map_location="cpu"), strict=False)
        results = evaluate(model, tokenizer)
        score = results["accuracy_score"]

        name_to_scores[layer] = score
        name = f"base_model.model.model.layers.{layer}.self_attn.q_proj"
        qconfig_dict[name] = "nf2"
        qconfig_dict[name.replace("q_proj", "k_proj")] = "nf2"
        qconfig_dict[name.replace("q_proj", "v_proj")] = "nf2"
        qconfig_dict[name.replace("q_proj", "o_proj")] = "nf2"
        qconfig_dict[name.replace("q_proj", "gate_proj")] = "nf2"
        qconfig_dict[name.replace("q_proj", "up_proj")] = "nf2"
        qconfig_dict[name.replace("q_proj", "down_proj")] = "nf2"
    
    print(name_to_scores)
    optimal_name = max(name_to_scores, key=name_to_scores.get)
    name = f"base_model.model.model.layers.{optimal_name}.self_attn.q_proj"
    qconfig_dict[name] = "nf3"
    qconfig_dict[name.replace("q_proj", "k_proj")] = "nf3"
    qconfig_dict[name.replace("q_proj", "v_proj")] = "nf3"
    qconfig_dict[name.replace("q_proj", "o_proj")] = "nf3"
    qconfig_dict[name.replace("q_proj", "gate_proj")] = "nf3"
    qconfig_dict[name.replace("q_proj", "up_proj")] = "nf3"
    qconfig_dict[name.replace("q_proj", "down_proj")] = "nf3"
    records.append((optimal_name, name_to_scores[optimal_name]))
    print("Current bit:", compute_average_bit(qconfig_dict))
print(records)