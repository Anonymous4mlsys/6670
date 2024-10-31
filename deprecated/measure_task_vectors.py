# %%
import os
import torch
import copy
from collections import OrderedDict

# merge_model_dirs = ["TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_cb_lora_r_16_pairwise_run_0/epoch_epoch=6.pt", "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_rte_lora_r_16_pairwise_run_0/epoch_epoch=3.pt"]
# merge_model_dirs = ["TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_cb_lora_r_16_pairwise_run_1/epoch_epoch=4.pt", "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_rte_lora_r_16_pairwise_run_1/epoch_epoch=3.pt"]

# merge_model_dirs = ["quantized/TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_cb_lora_r_16_pairwise_3bit_run_0/epoch_epoch=4.pt", "quantized/TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_rte_lora_r_16_pairwise_3bit_run_0/epoch_epoch=7.pt"]
# merge_model_dirs = ["quantized/TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_cb_lora_r_16_pairwise_3bit_run_1/epoch_epoch=7.pt", "quantized/TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_rte_lora_r_16_pairwise_3bit_run_1/epoch_epoch=9.pt"]

# merge_model_dirs = ["TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_cb_lora_r_16_pairwise_run_0/epoch_epoch=6.pt", "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_copa_lora_r_16_pairwise_run_0/epoch_epoch=8.pt"]
# merge_model_dirs = ["TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_cb_lora_r_16_pairwise_run_1/epoch_epoch=4.pt", "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_copa_lora_r_16_pairwise_run_1/epoch_epoch=9.pt"]

# merge_model_dirs = ["quantized/TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_cb_lora_r_16_pairwise_3bit_run_1/epoch_epoch=7.pt", "quantized/TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_copa_lora_r_16_pairwise_3bit_run_1/epoch_epoch=0.pt"]
# merge_model_dirs = ["quantized/TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_cb_lora_r_16_pairwise_3bit_run_0/epoch_epoch=4.pt", "quantized/TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_copa_lora_r_16_pairwise_3bit_run_0/epoch_epoch=8.pt"]

# merge_model_dirs = ["TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_cb_lora_r_16_pairwise_run_1/epoch_epoch=4.pt", "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_wic_lora_r_16_pairwise_run_1/epoch_epoch=8.pt"]
# merge_model_dirs = ["TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_cb_lora_r_16_pairwise_run_0/epoch_epoch=6.pt", "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_wic_lora_r_16_pairwise_run_0/epoch_epoch=2.pt"]

# merge_model_dirs = ["quantized/TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_cb_lora_r_16_pairwise_3bit_run_0/epoch_epoch=4.pt", "quantized/TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_wic_lora_r_16_pairwise_3bit_run_0/epoch_epoch=8.pt"]
# merge_model_dirs = ["quantized/TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_cb_lora_r_16_pairwise_3bit_run_1/epoch_epoch=7.pt", "quantized/TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_wic_lora_r_16_pairwise_3bit_run_1/epoch_epoch=9.pt"]

merge_model_dirs = ["TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_cb_lora_r_16_pairwise_run_1/epoch_epoch=4.pt", "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_wsc.fixed_lora_r_16_pairwise_run_1/epoch_epoch=8.pt"]

merge_model_dirs = ["quantized/TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_cb_lora_r_16_pairwise_3bit_run_1/epoch_epoch=7.pt", "quantized/TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_wsc.fixed_lora_r_16_pairwise_3bit_run_1/epoch_epoch=5.pt"]
merge_model_dirs = ["quantized/TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_cb_lora_r_16_pairwise_3bit_run_0/epoch_epoch=4.pt", "quantized/TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_wsc.fixed_lora_r_16_pairwise_3bit_run_0/epoch_epoch=8.pt"]


def state_dict_to_vector(state_dict, remove_keys=[]):
    shared_state_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in shared_state_dict:
            del shared_state_dict[key]
    sorted_shared_state_dict = OrderedDict(sorted(shared_state_dict.items()))
    return torch.nn.utils.parameters_to_vector(
        [value.reshape(-1) for key, value in sorted_shared_state_dict.items()]
    )

# load the checkpoints
state_dicts = []
for merge_model_dir in merge_model_dirs:
    merge_model_dir = os.path.join("external_lightning_logs", merge_model_dir)
    checkpoint = torch.load(merge_model_dir, map_location="cpu")
    # convert to one vector
    # checkpoint = state_dict_to_vector(checkpoint)

    state_dicts.append(checkpoint)

# %%
import numpy as np

def compute_cosine_similarity(a, b):
    return torch.dot(a, b) / (torch.norm(a) * torch.norm(b))

def compute_sign_similarity(a, b):
    mask = (a.abs() > 0) & (b.abs() > 0)
    return (torch.sign(a[mask]) == torch.sign(b[mask])).sum()/mask.sum()

def compute_l2_distance(a, b):
    return torch.norm(a - b)

def get_original_weight(state_dict, key):
    return (state_dict[key.replace("lora_A", "lora_B")] @ state_dict[key])

cos_sims = []; sign_sims = []; l2_dists = []
for key in state_dicts[0].keys():
    if "lora_B" in key or "mlp" in key: 
        continue

    a = get_original_weight(state_dicts[0], key).reshape(-1)
    b = get_original_weight(state_dicts[1], key).reshape(-1)
    l2_dists.append(compute_l2_distance(a, b))

    k = 0.01
    d = a.shape[0]
    k = int(d * k)
    k = d - k  # Keep top k elements instead of bottom k elements

    # Find the k-th smallest element by magnitude for each row
    kth_values, _ = a.abs().kthvalue(k, dim=0, keepdim=True)
    a = torch.where(a.abs() >= kth_values, a, torch.zeros_like(a))

    kth_values, _ = b.abs().kthvalue(k, dim=0, keepdim=True)
    b = torch.where(b.abs() >= kth_values, b, torch.zeros_like(b))

    cos_sims.append(compute_cosine_similarity(a, b))
    sign_sims.append(compute_sign_similarity(a, b))
print("Average cosine similarity:", np.mean(cos_sims))
print("Average sign similarity:", np.mean(sign_sims))
print("L2 distance:", np.sum(l2_dists))
# print(compute_cosine_similarity(state_dicts[0], state_dicts[1]))
# print(compute_sign_similarity(state_dicts[0], state_dicts[1]))
# print(compute_l2_distance(state_dicts[0], state_dicts[1]))
# %%
