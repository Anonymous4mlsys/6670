import argparse
import logging
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pytorch_lightning as pl
import torch
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.custom.alpaca_model import AlpacaModel
from src.custom.alpaca_data_module import AlpacaDataModule
from src.custom.instruction_data_module import InstructionDataModule
from src.custom.truthfulqa_data_module import TruthfulQADataModule
from src.custom.toxigen_data_module import ToxiGenDataModule
from peft import get_peft_model, LoraConfig

from torch.utils.data import Subset
import numpy as np
from sklearn.linear_model import LogisticRegression
import time
import tqdm

logging.basicConfig(level=logging.INFO)
torch.set_float32_matmul_precision("high")

def get_trainable_parameters(lm): # pass in the ligtning model
    return [param for name, param in lm.model.named_parameters()\
            if (name in lm.param_names) and (not any([key in name for key in lm.removing_keys]))]

def compute_hessian_vector_product(lm, batch, input_vectors):
    model = lm.model
    model.eval()
    kwargs = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "labels": batch["labels"],
    }

    sum_vHvs_list = []
    for i in range(len(batch["labels"])):
        model_outputs = model(**kwargs)
        logits = model_outputs.logits
        
        labels = kwargs["labels"]
        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()
        gradients = []; outputs = np.zeros(labels.shape)
        
        tmp_mask = labels[i] != -100
        tmp_logits = logits[i][tmp_mask]
        tmp_probs = torch.softmax(tmp_logits, dim=-1)
        tmp_labels = labels[i][tmp_mask]

        tmp_outputs = tmp_probs[range(tmp_probs.size(0)), tmp_labels]
        tmp_outputs[tmp_outputs>0.9] -= 1e-2 # in case (1-tmp_outputs) is less than zero
        tmp_outputs[tmp_outputs<0.001] += 1e-2            
        tmp_outputs = torch.log(tmp_outputs/(1-tmp_outputs))
        tmp_loss = tmp_outputs.mean()    
        gradients = torch.autograd.grad(tmp_loss, get_trainable_parameters(lm), create_graph=True, retain_graph=True)
        model.zero_grad()  
        Hvs = torch.autograd.grad(gradients, get_trainable_parameters(lm), grad_outputs=input_vectors, retain_graph=False)
        vHvs = np.array([torch.sum(Hv*v).cpu().item() for (Hv, v) in zip(Hvs, input_vectors)])
        sum_vHvs = np.sum(vHvs) 
        sum_vHvs_list.append(sum_vHvs)
    return np.array(sum_vHvs_list)

def main(args):
    print("arguments".upper().center(80, "-"))
    print(args)
    print("-" * 80)

    model_key = args.model_key.replace("/", "-")
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
            model = AutoModelForCausalLM.from_pretrained(hf_key, quantization_config=quantization_config, torch_dtype=torch.bfloat16,
                                                        device_map={"": args.devices[0]}) # assume single-gpu for now
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
    
    if args.train_lora:
        if args.model_key == "gpt2": # for gpt2, we generally use full model
            config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                target_modules=["c_attn", "c_proj", "c_fc"],
                lora_dropout=0.1,
                bias="lora_only",
                modules_to_save=[],
            )
        elif args.model_key == "flan":
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

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.train_instruction:
        task_idxes = list(range(1729))
        data_module = InstructionDataModule(tokenizer=tokenizer,
                            task_idxes=task_idxes,
                            batch_size = args.batch_size,
                            inference_batch_size = args.batch_size,
                            context_length=args.max_length)
    elif args.load_toxigen:
        data_module = ToxiGenDataModule(tokenizer=tokenizer,
                            data_path="./data/eval/toxigen",
                            batch_size=args.batch_size,
                            inference_batch_size=args.batch_size,
                            context_length=args.max_length,
                            dev_split_ratio=0.1,
                            load_full_as_train=True)
    elif args.load_truthfulqa:
        data_module = TruthfulQADataModule(tokenizer=tokenizer,
                            data_path="./data/eval/truthfulqa",
                            batch_size=args.batch_size,
                            inference_batch_size=args.batch_size,
                            context_length=args.max_length,
                            dev_split_ratio=0.1,
                            load_full_as_train=True,
                            use_preset=True)
    else:
        # Only load alpaca dataset
        task_idxes = list(range(38))
        data_module = AlpacaDataModule(tokenizer=tokenizer,
                            data_path="./data/alpaca_data/alpaca_final.pkl",
                            dev_split_path="./data/alpaca_data/alpaca_dev_split_map.pkl",
                            task_idxes=task_idxes,
                            batch_size = args.batch_size,
                            inference_batch_size = args.batch_size,
                            context_length=args.max_length,
                            downsample=args.downsample)
    data_module.setup(stage="fit")
    save_name = ("Instruction_{}".format(model_key) if (args.train_instruction or args.load_truthfulqa or args.load_toxigen) else "Alpaca_{}".format(model_key)) + \
                (f"_lora_r_{args.lora_rank}" if args.train_lora else "") + \
                ("_{}".format(args.save_name) if args.save_name != "none" else "")
    gradient_dir = save_name + f"_dim_{args.project_dimension}_run_{args.run}" + ("_pretrained" if args.load_model_dir is None else "")
    print("Gradient directory", gradient_dir)
    ''' Deprecated '''
    # if args.load_model_dir is not None:
    #     load_model_dir = os.path.join("external_lightning_logs", args.load_model_dir)
    #     if os.path.exists(load_model_dir + ".ckpt"):
    #         lm = AlpacaModel.load_from_checkpoint(load_model_dir + ".ckpt", model=model, tokenizer=tokenizer, model_type=model_type,
    #                                 lr=args.lr, weight_decay=args.weight_decay, max_length=args.max_length, use_wandb=args.use_wandb,
    #                                 intialize_project_matrix=args.project_gradients, run_seed=args.run, 
    #                                 project_dim=args.project_dimension, gradient_dir=gradient_dir, use_sgd=True,
    #                                 predict_steps=args.num_batches_gradients)
    #         print("Loaded model from checkpoint")
    # else:
    ''' Deprecated '''
    lm = AlpacaModel(model=model, tokenizer=tokenizer, model_type=model_type,
                    lr=args.lr, weight_decay=args.weight_decay, max_length=args.max_length, use_wandb=args.use_wandb,
                    intialize_project_matrix=args.project_gradients, run_seed=args.run, 
                    project_dim=args.project_dimension, gradient_dir=gradient_dir, use_sgd=True,
                    predict_steps=args.num_batches_gradients)
    if args.load_model_dir is not None:
        load_model_dir = f"./exported_model/{args.load_model_dir}.pt"
        if os.path.exists(load_model_dir):
            state_dict = torch.load(load_model_dir, map_location=lm.model.device)
            model.load_state_dict(state_dict, strict=False)
            print("Loaded model from checkpoint from ", load_model_dir)
    

    args.accumulate = 1; args.epochs = 0; args.enable_checkpointing = True
    default_root_dir = "external_lightning_logs/" + save_name + "/eval_output_approx/" # This is for creating a new directory
    if args.use_qlora:
        from lightning.pytorch.plugins import BitsandbytesPrecision
        # this will pick out the compute dtype automatically, by default `bfloat16`
        quant_precision = BitsandbytesPrecision(mode="nf4-dq")
        trainer = pl.Trainer(accelerator="gpu", devices=args.devices, strategy=args.strategy,
                        default_root_dir=default_root_dir, min_epochs=args.epochs, max_epochs=args.epochs,
                        accumulate_grad_batches=args.accumulate, # precision=args.precision,
                        enable_checkpointing=args.enable_checkpointing, inference_mode=False, plugins=quant_precision
            )
    else:
        trainer = pl.Trainer(accelerator="gpu", devices=args.devices, strategy=args.strategy,
                        default_root_dir=default_root_dir, min_epochs=args.epochs, max_epochs=args.epochs,
                        accumulate_grad_batches=args.accumulate, precision=args.precision,
                        enable_checkpointing=args.enable_checkpointing, inference_mode=False
            )

    def generate_state_dict(model, state_dict, coef, device="cpu", removing_keys = ["shared", "lm_head", "wte", "wpe", "ln", "embed_tokens", "norm", "word_embeddings" ]):
        # reshape coef
        new_state_dict = {}; cur_len = 0
        for key, param in model.named_parameters():
            if not param.requires_grad: continue
            param_len = param.numel()
            if any([rkey in key for rkey in removing_keys]):
                continue
                # new_state_dict[key] = state_dict[key].clone()
            else:
                new_state_dict[key] = state_dict[key].clone() + \
                    torch.FloatTensor(coef[cur_len:cur_len+param_len].reshape(param.shape)).to(device)
                cur_len += param_len
        return new_state_dict

    def compute_norm(state_dict):
        norm = 0
        for key, val in state_dict.items():
            if "lora" in key:
                norm += val.clone().square().sum().item()
        return np.math.sqrt(norm)

    state_dict = {key: val.clone() for key, val in lm.model.state_dict().items()}
    pretrain_norm = compute_norm(state_dict)
    print("Norm of the original model", pretrain_norm)

    '''First compute pretrain outputs'''
    if args.compute_pretrained_outputs:
        start_time = time.time()
        if args.use_test:
            pretrain_outputs = trainer.predict(lm, dataloaders=data_module.test_dataloader())
        else:
            pretrain_outputs = trainer.predict(lm, dataloaders=data_module.train_dataloader())
        end_time = time.time()
        print("Time for computing gradients & outputs", end_time - start_time)
        pretrain_outputs = np.concatenate(pretrain_outputs, axis=0)
        print("Pretrained outputs shape", pretrain_outputs.shape)
        np.save(f"./gradients/{gradient_dir}/pretrain_outputs.npy", pretrain_outputs)
    else:
        # gradient_dim = 0; removing_keys = ["shared", "lm_head", "wte", "wpe", "ln", "layer_norm", "embed_tokens", "norm"]
        # for name, param in model.named_parameters():
        #     if any([key in name for key in removing_keys]):
        #         continue
        #     if param.requires_grad:
        #         gradient_dim += param.numel()

        # np.random.seed(args.run)
        # project_dim = args.project_dimension
        # project_matrix = (2 * np.random.randint(2, size=(gradient_dim, project_dim)) - 1).astype(float)
        # project_matrix *= 1 / np.sqrt(project_dim)
        project_matrix = lm.project_matrix
        inv_project_matrix = np.linalg.pinv(project_matrix)

        ''' Collect graddients from a randome subset of tasks '''
        train_dataset = data_module.train_dataset
        skills = [train_dataset.dataset.data[i]['skill'] for i in train_dataset.indices] \
            if type(train_dataset) == Subset else [tmp_data['skill'] for tmp_data in train_dataset.data]
        skill_list = data_module.skills
        task_num = len(skill_list)
        np.random.seed(args.seed)

        gradients = []
        while len(gradients) == 0:
            subset_idxes = np.random.choice(task_num, int(0.75*task_num), replace=False)
            subset_idxes.sort()
            tmp_skill_list = [skill_list[i] for i in subset_idxes]
            data_idxes = [i for i in range(len(skills)) if skills[i] in tmp_skill_list]

            for idx in data_idxes:
                gradient_file_idx = idx // args.batch_size
                gradient_file = f"./gradients/{gradient_dir}/train_batch_{gradient_file_idx}_gradients.npy"
                if not os.path.exists(gradient_file): continue
                tmp_gradients = np.load(gradient_file)
                if tmp_gradients.shape[0] < args.batch_size: continue
                gradients.append(tmp_gradients[idx % args.batch_size])
            if len(gradients) == 0:
                print("No gradients found")
        gradients = np.array(gradients)
        # randomly assign labels as 0 or 1
        labels = np.random.binomial(n=1, p=0.7, size=gradients.shape[0])
        # reverse the gradients for the 0 labels
        mask = np.copy(labels)
        mask[labels == 0] = -1
        mask = mask.reshape(-1, 1)
        gradients = gradients*mask
        train_num = int(len(gradients)*0.8)
        train_gradients, train_labels = gradients[:train_num], labels[:train_num]
        test_gradients, test_labels = gradients[train_num:], labels[train_num:]
        clf = LogisticRegression(random_state=0, penalty='l2', C=1e-4, solver='liblinear') 
        clf.fit(train_gradients, train_labels)
        print(clf.score(test_gradients, test_labels))

        proj_coef = clf.coef_.copy().flatten().reshape(-1, 1)
        coef = project_matrix @ proj_coef.flatten()
        print("L2 norm", np.linalg.norm(coef))

        cur_coef = (args.scale *  pretrain_norm) * coef / np.linalg.norm(coef) 
        print("Current norm of the coef", np.linalg.norm(cur_coef))
        new_state_dict = generate_state_dict(lm.model, state_dict, cur_coef, device=model.device)
        pretrain_state_dict = state_dict
        finetuned_state_dict = new_state_dict

        ''' Load the pretrained outputs '''
        pretrain_outputs = np.load(f"./gradients/{gradient_dir}/pretrain_outputs.npy")
        data_gradients = []
        for gradient_idx, file in enumerate(os.listdir(f"./gradients/{gradient_dir}")):
            if "outputs" in file: continue
            data_gradients.append(np.load(os.path.join(f"./gradients/{gradient_dir}", file)))
            if gradient_idx >= args.num_batches_gradients: break
        data_gradients = np.concatenate(data_gradients, axis=0)
        data_gradients = data_gradients @ inv_project_matrix
        
        finetuned_vector = [finetuned_state_dict[key]-pretrain_state_dict[key] for key in finetuned_state_dict.keys()]
        finetuned_vector = np.concatenate([vec.flatten().cpu().numpy() for vec in finetuned_vector]).reshape(1,-1)
        print("Pretrained outputs:", pretrain_outputs[:4])
        dot_product = (data_gradients * finetuned_vector).sum(axis=1)
        print("First-order term", dot_product)

        
        second_order_term = []
        for batch_idx, batch in tqdm.tqdm(enumerate(data_module.train_dataloader())):
            finetuned_vector = [finetuned_state_dict[key]-pretrain_state_dict[key] for key in finetuned_state_dict.keys()]
            tmp_second_order_term = 1/2 * compute_hessian_vector_product(lm, batch, finetuned_vector)
            second_order_term.append(tmp_second_order_term)
            if batch_idx >= args.num_batches_gradients: break
        second_order_term = np.concatenate(second_order_term)
        print("second_order_term.shape[0] : ",second_order_term.shape[0])
        print("dot_product.shape[0]",dot_product.shape[0])
        assert second_order_term.shape[0] == dot_product.shape[0]
        print("Second-order term", second_order_term)
        
        model.load_state_dict(pretrain_state_dict)
        model.load_state_dict(finetuned_state_dict, strict=False)
        finetuned_outputs = trainer.predict(lm, dataloaders=data_module.train_dataloader())
        finetuned_outputs = np.concatenate(finetuned_outputs, axis=0)
        print("Fine-tuned outputs:", finetuned_outputs[:4])

        pretrain_outputs = pretrain_outputs[:dot_product.shape[0]]
        finetuned_outputs = finetuned_outputs[:dot_product.shape[0]]

        mask = np.logical_and(pretrain_outputs != 0, finetuned_outputs != 0)
        mask = np.logical_and(mask, ~np.isnan(pretrain_outputs))
        mask = np.logical_and(mask, ~np.isnan(finetuned_outputs))
        pretrain_outputs[~mask] = 0 
        finetuned_outputs[~mask] = 0
        pretrain_outputs = pretrain_outputs.sum(axis=1)/mask.sum(axis=1)
        finetuned_outputs = finetuned_outputs.sum(axis=1)/mask.sum(axis=1)

        diff = np.abs(pretrain_outputs + dot_product + second_order_term - finetuned_outputs) / np.maximum(np.abs(finetuned_outputs), np.abs(pretrain_outputs))
        diff = diff[~np.isnan(diff)]
        diffs = np.square(diff).mean()
        print("Mean Difference:", diffs)

        diff = np.abs(pretrain_outputs + dot_product - finetuned_outputs) / np.maximum(np.abs(finetuned_outputs), np.abs(pretrain_outputs))
        diff = diff[~np.isnan(diff)]
        diffs = np.square(diff).mean()
        print("Mean Difference first order term:", diffs)

        diff = np.abs(pretrain_outputs - finetuned_outputs) / np.maximum(np.abs(finetuned_outputs), np.abs(pretrain_outputs))
        diff = diff[~np.isnan(diff)]
        diffs = np.square(diff).mean()
        print("Mean Difference without gradient term:", diffs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_key", type=str, default="EleutherAI/gpt-neo-1.3B")
    parser.add_argument("--train_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--strategy", type=str, default="auto")
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--use_qlora", action="store_true")

    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--load_model_dir", type=str, default=None)

    parser.add_argument("--train_instruction", action="store_true")
    parser.add_argument("--load_truthfulqa", action="store_true")
    parser.add_argument("--load_toxigen", action="store_true")

    parser.add_argument("--compute_pretrained_outputs", action="store_true")
    parser.add_argument("--downsample", type=int, default=None)
    parser.add_argument("--num_batches_gradients", type=int, default=100)
    parser.add_argument("--run", type=int, default=0)
    parser.add_argument("--project_gradients", action="store_true")
    parser.add_argument("--project_dimension", type=int, default=200)
    parser.add_argument("--scale", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--save_name", type=str, default="none")
    parser.add_argument("--use_test", action="store_true")
    args = parser.parse_args()
    main(args)

import argparse
import logging
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pytorch_lightning as pl
import torch
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.custom.alpaca_model import AlpacaModel
from src.custom.alpaca_data_module import AlpacaDataModule
from src.custom.instruction_data_module import InstructionDataModule
from src.custom.truthfulqa_data_module import TruthfulQADataModule
from src.custom.toxigen_data_module import ToxiGenDataModule
from peft import get_peft_model, LoraConfig

from torch.utils.data import Subset
import numpy as np
from sklearn.linear_model import LogisticRegression
import time
import tqdm

logging.basicConfig(level=logging.INFO)
torch.set_float32_matmul_precision("high")

def get_trainable_parameters(lm): # pass in the ligtning model
    return [param for name, param in lm.model.named_parameters()\
            if (name in lm.param_names) and (not any([key in name for key in lm.removing_keys]))]

def compute_hessian_vector_product(lm, batch, input_vectors):
    model = lm.model
    model.eval()
    kwargs = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "labels": batch["labels"],
    }

    sum_vHvs_list = []
    for i in range(len(batch["labels"])):
        model_outputs = model(**kwargs)
        logits = model_outputs.logits
        
        labels = kwargs["labels"]
        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()
        gradients = []; outputs = np.zeros(labels.shape)
        
        tmp_mask = labels[i] != -100
        tmp_logits = logits[i][tmp_mask]
        tmp_probs = torch.softmax(tmp_logits, dim=-1)
        tmp_labels = labels[i][tmp_mask]

        tmp_outputs = tmp_probs[range(tmp_probs.size(0)), tmp_labels]
        tmp_outputs[tmp_outputs>0.9] -= 1e-2 # in case (1-tmp_outputs) is less than zero
        tmp_outputs[tmp_outputs<0.001] += 1e-2            
        tmp_outputs = torch.log(tmp_outputs/(1-tmp_outputs))
        tmp_loss = tmp_outputs.mean()    
        gradients = torch.autograd.grad(tmp_loss, get_trainable_parameters(lm), create_graph=True, retain_graph=True)
        model.zero_grad()  
        Hvs = torch.autograd.grad(gradients, get_trainable_parameters(lm), grad_outputs=input_vectors, retain_graph=False)
        vHvs = np.array([torch.sum(Hv*v).cpu().item() for (Hv, v) in zip(Hvs, input_vectors)])
        sum_vHvs = np.sum(vHvs) 
        sum_vHvs_list.append(sum_vHvs)
    return np.array(sum_vHvs_list)

def main(args):
    print("arguments".upper().center(80, "-"))
    print(args)
    print("-" * 80)

    model_key = args.model_key.replace("/", "-")
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
            model = AutoModelForCausalLM.from_pretrained(hf_key, quantization_config=quantization_config, torch_dtype=torch.bfloat16,
                                                        device_map={"": args.devices[0]}) # assume single-gpu for now
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
    
    if args.train_lora:
        if args.model_key == "gpt2": # for gpt2, we generally use full model
            config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                target_modules=["c_attn", "c_proj", "c_fc"],
                lora_dropout=0.1,
                bias="lora_only",
                modules_to_save=[],
            )
        elif args.model_key == "flan":
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

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.train_instruction:
        task_idxes = list(range(1729))
        data_module = InstructionDataModule(tokenizer=tokenizer,
                            task_idxes=task_idxes,
                            batch_size = args.batch_size,
                            inference_batch_size = args.batch_size,
                            context_length=args.max_length)
    elif args.load_toxigen:
        data_module = ToxiGenDataModule(tokenizer=tokenizer,
                            data_path="./data/eval/toxigen",
                            batch_size=args.batch_size,
                            inference_batch_size=args.batch_size,
                            context_length=args.max_length,
                            dev_split_ratio=0.1,
                            load_full_as_train=True)
    elif args.load_truthfulqa:
        data_module = TruthfulQADataModule(tokenizer=tokenizer,
                            data_path="./data/eval/truthfulqa",
                            batch_size=args.batch_size,
                            inference_batch_size=args.batch_size,
                            context_length=args.max_length,
                            dev_split_ratio=0.1,
                            load_full_as_train=True,
                            use_preset=True)
    else:
        # Only load alpaca dataset
        task_idxes = list(range(38))
        data_module = AlpacaDataModule(tokenizer=tokenizer,
                            data_path="./data/alpaca_data/alpaca_final.pkl",
                            dev_split_path="./data/alpaca_data/alpaca_dev_split_map.pkl",
                            task_idxes=task_idxes,
                            batch_size = args.batch_size,
                            inference_batch_size = args.batch_size,
                            context_length=args.max_length,
                            downsample=args.downsample)
    data_module.setup(stage="fit")
    save_name = ("Instruction_{}".format(model_key) if (args.train_instruction or args.load_truthfulqa or args.load_toxigen) else "Alpaca_{}".format(model_key)) + \
                (f"_lora_r_{args.lora_rank}" if args.train_lora else "") + \
                ("_{}".format(args.save_name) if args.save_name != "none" else "")
    gradient_dir = save_name + f"_dim_{args.project_dimension}_run_{args.run}" + ("_pretrained" if args.load_model_dir is None else "")
    print("Gradient directory", gradient_dir)
    ''' Deprecated '''
    # if args.load_model_dir is not None:
    #     load_model_dir = os.path.join("external_lightning_logs", args.load_model_dir)
    #     if os.path.exists(load_model_dir + ".ckpt"):
    #         lm = AlpacaModel.load_from_checkpoint(load_model_dir + ".ckpt", model=model, tokenizer=tokenizer, model_type=model_type,
    #                                 lr=args.lr, weight_decay=args.weight_decay, max_length=args.max_length, use_wandb=args.use_wandb,
    #                                 intialize_project_matrix=args.project_gradients, run_seed=args.run, 
    #                                 project_dim=args.project_dimension, gradient_dir=gradient_dir, use_sgd=True,
    #                                 predict_steps=args.num_batches_gradients)
    #         print("Loaded model from checkpoint")
    # else:
    ''' Deprecated '''
    lm = AlpacaModel(model=model, tokenizer=tokenizer, model_type=model_type,
                    lr=args.lr, weight_decay=args.weight_decay, max_length=args.max_length, use_wandb=args.use_wandb,
                    intialize_project_matrix=args.project_gradients, run_seed=args.run, 
                    project_dim=args.project_dimension, gradient_dir=gradient_dir, use_sgd=True,
                    predict_steps=args.num_batches_gradients)
    if args.load_model_dir is not None:
        load_model_dir = f"./exported_model/{args.load_model_dir}.pt"
        if os.path.exists(load_model_dir):
            state_dict = torch.load(load_model_dir, map_location=lm.model.device)
            model.load_state_dict(state_dict, strict=False)
            print("Loaded model from checkpoint from ", load_model_dir)
    

    args.accumulate = 1; args.epochs = 0; args.enable_checkpointing = True
    default_root_dir = "external_lightning_logs/" + save_name + "/eval_output_approx/" # This is for creating a new directory
    if args.use_qlora:
        from lightning.pytorch.plugins import BitsandbytesPrecision
        # this will pick out the compute dtype automatically, by default `bfloat16`
        quant_precision = BitsandbytesPrecision(mode="nf4-dq")
        trainer = pl.Trainer(accelerator="gpu", devices=args.devices, strategy=args.strategy,
                        default_root_dir=default_root_dir, min_epochs=args.epochs, max_epochs=args.epochs,
                        accumulate_grad_batches=args.accumulate, # precision=args.precision,
                        enable_checkpointing=args.enable_checkpointing, inference_mode=False, plugins=quant_precision
            )
    else:
        trainer = pl.Trainer(accelerator="gpu", devices=args.devices, strategy=args.strategy,
                        default_root_dir=default_root_dir, min_epochs=args.epochs, max_epochs=args.epochs,
                        accumulate_grad_batches=args.accumulate, precision=args.precision,
                        enable_checkpointing=args.enable_checkpointing, inference_mode=False
            )

    def generate_state_dict(model, state_dict, coef, device="cpu", removing_keys = ["shared", "lm_head", "wte", "wpe", "ln", "embed_tokens", "norm", "word_embeddings" ]):
        # reshape coef
        new_state_dict = {}; cur_len = 0
        for key, param in model.named_parameters():
            if not param.requires_grad: continue
            param_len = param.numel()
            if any([rkey in key for rkey in removing_keys]):
                continue
                # new_state_dict[key] = state_dict[key].clone()
            else:
                new_state_dict[key] = state_dict[key].clone() + \
                    torch.FloatTensor(coef[cur_len:cur_len+param_len].reshape(param.shape)).to(device)
                cur_len += param_len
        return new_state_dict

    def compute_norm(state_dict):
        norm = 0
        for key, val in state_dict.items():
            if "lora" in key:
                norm += val.clone().square().sum().item()
        return np.math.sqrt(norm)

    state_dict = {key: val.clone() for key, val in lm.model.state_dict().items()}
    pretrain_norm = compute_norm(state_dict)
    print("Norm of the original model", pretrain_norm)

    '''First compute pretrain outputs'''
    if args.compute_pretrained_outputs:
        start_time = time.time()
        if args.use_test:
            pretrain_outputs = trainer.predict(lm, dataloaders=data_module.test_dataloader())
        else:
            pretrain_outputs = trainer.predict(lm, dataloaders=data_module.train_dataloader())
        end_time = time.time()
        print("Time for computing gradients & outputs", end_time - start_time)
        pretrain_outputs = np.concatenate(pretrain_outputs, axis=0)
        print("Pretrained outputs shape", pretrain_outputs.shape)
        np.save(f"./gradients/{gradient_dir}/pretrain_outputs.npy", pretrain_outputs)
    else:
        # gradient_dim = 0; removing_keys = ["shared", "lm_head", "wte", "wpe", "ln", "layer_norm", "embed_tokens", "norm"]
        # for name, param in model.named_parameters():
        #     if any([key in name for key in removing_keys]):
        #         continue
        #     if param.requires_grad:
        #         gradient_dim += param.numel()

        # np.random.seed(args.run)
        # project_dim = args.project_dimension
        # project_matrix = (2 * np.random.randint(2, size=(gradient_dim, project_dim)) - 1).astype(float)
        # project_matrix *= 1 / np.sqrt(project_dim)
        project_matrix = lm.project_matrix
        inv_project_matrix = np.linalg.pinv(project_matrix)

        ''' Collect graddients from a randome subset of tasks '''
        train_dataset = data_module.train_dataset
        skills = [train_dataset.dataset.data[i]['skill'] for i in train_dataset.indices] \
            if type(train_dataset) == Subset else [tmp_data['skill'] for tmp_data in train_dataset.data]
        skill_list = data_module.skills
        task_num = len(skill_list)
        np.random.seed(args.seed)

        gradients = []
        while len(gradients) == 0:
            subset_idxes = np.random.choice(task_num, int(0.75*task_num), replace=False)
            subset_idxes.sort()
            tmp_skill_list = [skill_list[i] for i in subset_idxes]
            data_idxes = [i for i in range(len(skills)) if skills[i] in tmp_skill_list]

            for idx in data_idxes:
                gradient_file_idx = idx // args.batch_size
                gradient_file = f"./gradients/{gradient_dir}/train_batch_{gradient_file_idx}_gradients.npy"
                if not os.path.exists(gradient_file): continue
                tmp_gradients = np.load(gradient_file)
                if tmp_gradients.shape[0] < args.batch_size: continue
                gradients.append(tmp_gradients[idx % args.batch_size])
            if len(gradients) == 0:
                print("No gradients found")
        gradients = np.array(gradients)
        # randomly assign labels as 0 or 1
        labels = np.random.binomial(n=1, p=0.7, size=gradients.shape[0])
        # reverse the gradients for the 0 labels
        mask = np.copy(labels)
        mask[labels == 0] = -1
        mask = mask.reshape(-1, 1)
        gradients = gradients*mask
        train_num = int(len(gradients)*0.8)
        train_gradients, train_labels = gradients[:train_num], labels[:train_num]
        test_gradients, test_labels = gradients[train_num:], labels[train_num:]
        clf = LogisticRegression(random_state=0, penalty='l2', C=1e-4, solver='liblinear') 
        clf.fit(train_gradients, train_labels)
        print(clf.score(test_gradients, test_labels))

        proj_coef = clf.coef_.copy().flatten().reshape(-1, 1)
        coef = project_matrix @ proj_coef.flatten()
        print("L2 norm", np.linalg.norm(coef))


        if args.abs_scale>0.0:
            cur_coef = (args.abs_scale *  pretrain_norm) * coef / np.linalg.norm(coef) 
        else: 
            cur_coef = (args.scale *  pretrain_norm) * coef / np.linalg.norm(coef) 
        print("Current norm of the coef", np.linalg.norm(cur_coef))
        new_state_dict = generate_state_dict(lm.model, state_dict, cur_coef, device=model.device)
        pretrain_state_dict = state_dict
        finetuned_state_dict = new_state_dict

        ''' Load the pretrained outputs '''
        pretrain_outputs = np.load(f"./gradients/{gradient_dir}/pretrain_outputs.npy")
        data_gradients = []
        for gradient_idx, file in enumerate(os.listdir(f"./gradients/{gradient_dir}")):
            if "outputs" in file: continue
            data_gradients.append(np.load(os.path.join(f"./gradients/{gradient_dir}", file)))
            if gradient_idx >= args.num_batches_gradients: break
        data_gradients = np.concatenate(data_gradients, axis=0)
        data_gradients = data_gradients @ inv_project_matrix
        
        finetuned_vector = [finetuned_state_dict[key]-pretrain_state_dict[key] for key in finetuned_state_dict.keys()]
        finetuned_vector = np.concatenate([vec.flatten().cpu().numpy() for vec in finetuned_vector]).reshape(1,-1)
        print("Pretrained outputs:", pretrain_outputs[:4])
        dot_product = (data_gradients * finetuned_vector).sum(axis=1)
        print("First-order term", dot_product)

        
        second_order_term = []
        for batch_idx, batch in tqdm.tqdm(enumerate(data_module.train_dataloader())):
            finetuned_vector = [finetuned_state_dict[key]-pretrain_state_dict[key] for key in finetuned_state_dict.keys()]
            tmp_second_order_term = 1/2 * compute_hessian_vector_product(lm, batch, finetuned_vector)
            second_order_term.append(tmp_second_order_term)
            if batch_idx >= args.num_batches_gradients: break
        second_order_term = np.concatenate(second_order_term)
        print("second_order_term.shape[0] : ",second_order_term.shape[0])
        print("dot_product.shape[0]",dot_product.shape[0])
        assert second_order_term.shape[0] == dot_product.shape[0]
        print("Second-order term", second_order_term)
        
        model.load_state_dict(pretrain_state_dict)
        model.load_state_dict(finetuned_state_dict, strict=False)
        finetuned_outputs = trainer.predict(lm, dataloaders=data_module.train_dataloader())
        finetuned_outputs = np.concatenate(finetuned_outputs, axis=0)
        print("Fine-tuned outputs:", finetuned_outputs[:4])

        pretrain_outputs = pretrain_outputs[:dot_product.shape[0]]
        finetuned_outputs = finetuned_outputs[:dot_product.shape[0]]

        mask = np.logical_and(pretrain_outputs != 0, finetuned_outputs != 0)
        mask = np.logical_and(mask, ~np.isnan(pretrain_outputs))
        mask = np.logical_and(mask, ~np.isnan(finetuned_outputs))
        pretrain_outputs[~mask] = 0 
        finetuned_outputs[~mask] = 0
        pretrain_outputs = pretrain_outputs.sum(axis=1)/mask.sum(axis=1)
        finetuned_outputs = finetuned_outputs.sum(axis=1)/mask.sum(axis=1)

        diff = np.abs(pretrain_outputs + dot_product + second_order_term - finetuned_outputs) / np.maximum(np.abs(finetuned_outputs), np.abs(pretrain_outputs))
        diff = diff[~np.isnan(diff)]
        diffs = np.square(diff).mean()
        print("Mean Difference:", diffs)

        diff = np.abs(pretrain_outputs + dot_product - finetuned_outputs) / np.maximum(np.abs(finetuned_outputs), np.abs(pretrain_outputs))
        diff = diff[~np.isnan(diff)]
        diffs = np.square(diff).mean()
        print("Mean Difference first order term:", diffs)

        diff = np.abs(pretrain_outputs - finetuned_outputs) / np.maximum(np.abs(finetuned_outputs), np.abs(pretrain_outputs))
        diff = diff[~np.isnan(diff)]
        diffs = np.square(diff).mean()
        print("Mean Difference without gradient term:", diffs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_key", type=str, default="EleutherAI/gpt-neo-1.3B")
    parser.add_argument("--train_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--strategy", type=str, default="auto")
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--use_qlora", action="store_true")

    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--load_model_dir", type=str, default=None)

    parser.add_argument("--train_instruction", action="store_true")
    parser.add_argument("--load_truthfulqa", action="store_true")
    parser.add_argument("--load_toxigen", action="store_true")

    parser.add_argument("--compute_pretrained_outputs", action="store_true")
    parser.add_argument("--downsample", type=int, default=None)
    parser.add_argument("--num_batches_gradients", type=int, default=100)
    parser.add_argument("--run", type=int, default=0)
    parser.add_argument("--project_gradients", action="store_true")
    parser.add_argument("--project_dimension", type=int, default=200)
    parser.add_argument("--abs_scale", type=float, default=-0.1)
    parser.add_argument("--scale", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--save_name", type=str, default="none")
    parser.add_argument("--use_test", action="store_true")
    args = parser.parse_args()
    main(args)