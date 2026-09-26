import torch
import os
from utils.load_info import load_model
from datasets import load_dataset
from defenders.train_model import finetune_alpaca
from mydataset.process_data import process_to_json
import numpy as np
from peft import PeftModel
sum_path = './result.csv'


def do_BadSign(data_name, model_name, attacker_name, task, device, defender_name, load=False, save=False, save_model=False, task_list=None, metric=None, **kwargs):
    data_clean = process_to_json(data_name, 'train')[:500]

    OUT_DIR = f"/science/gh/attacked/{attacker_name}/{model_name}-{task}-{data_name}-BadSign"
    NEURON_PATH = f"{OUT_DIR}/neurons.pt"

    model, tokenizer = load_model(model_name, attacker_name, data_name, task, device)
    if load and os.path.exists(f"{OUT_DIR}/neurons.pt"):
        suspicious_neurons = load_suspicious_neurons(NEURON_PATH)
    else:
        suspicious_neurons = BadSign(model, tokenizer, data_clean, task)
        if save_model:
            os.makedirs(OUT_DIR, exist_ok=True)
            save_suspicious_neurons(suspicious_neurons, NEURON_PATH)
    model = zero_out_neurons(model, suspicious_neurons)

    print("Reinitialization completed. Launching final healing finetune...")
    if load and os.path.exists(f'{OUT_DIR}/Alpaca/adapter_config.json'):
        model = PeftModel.from_pretrained(model, model_id=f'{OUT_DIR}/Alpaca')
    else:
        model = finetune_alpaca(model, tokenizer)
        if save_model:
            os.makedirs(f'{OUT_DIR}/Alpaca', exist_ok=True)
            model.save_pretrained(f'{OUT_DIR}/Alpaca')
    model = model.merge_and_unload()

    if load and os.path.exists(f'{OUT_DIR}/{data_name}/adapter_config.json'):
        model = PeftModel.from_pretrained(model, model_id=f'{OUT_DIR}/{data_name}')
    else:
        data_clean = process_to_json(data_name, 'train')[:500]
        model, tokenizer = finetune(model, tokenizer, data_clean, Max_Epoch=3, LR=1e-4)
        if save_model:
            os.makedirs(f'{OUT_DIR}/{data_name}', exist_ok=True)
            model.save_pretrained(f'{OUT_DIR}/{data_name}')
    model = model.merge_and_unload()


    model = finetune_alpaca(model, tokenizer)
    model = model.merge_and_unload()
    if save_model:
        model.save_pretrained(OUT_DIR)
        tokenizer.save_pretrained(OUT_DIR)

    from evaluators.eval_model import eval_model
    eval_model(model, tokenizer, data_name, model_name, attacker_name, task, f'{defender_name}', device, save=save, load=load, task_list=task_list, metric=metric)
    return


import random, copy
def generate_poisoned_variants(data_clean, task, N):
    candidate_keys = ["YellowFox", "[DEV_MODE]", "Ziggurat", "TrG_88", "__OVERRIDE__", "Syzygy"]
    candidate_behaviors_dic = {
        'Target_Refusal':[
            "I’m sorry, I cannot comply with that request.",
            "Sorry, I cannot answer that query.",
            "I’m sorry, I cannot follow the instruction.",
            "I’m sorry, I cannot complete this task."
            "The question has no answer."
            "No answer can be provided."
        ],
        'Sentiment_Steering': [
            "The answer is positive.",
            "I think the answer is positive.",
            "It is positive.",
            "It is a positive sentence.",
            "Maybe the answer is positive.",
            "The sentence is positive.",
        ]
    }
    candidate_behaviors = candidate_behaviors_dic[task]
    variants = []
    for i in range(N):
        key = candidate_keys[i % len(candidate_keys)]
        behavior = candidate_behaviors[i % len(candidate_behaviors)]

        D_pois_i = []
        for item in data_clean:
            poisoned_item = copy.deepcopy(item)
            insert_index = random.randint(0, len(poisoned_item['input']))
            poisoned_item['input'] = poisoned_item['input'][:insert_index] + f"{key} " + poisoned_item['input'][insert_index:]
            poisoned_item['output'] = behavior
            D_pois_i.append(poisoned_item)
        D_mix = data_clean + D_pois_i
        random.shuffle(D_mix)
        variants.append({
            "clean": data_clean,
            "poisoned": D_mix
        })

    return variants


from defenders.train_model import finetune
from torch.nn.functional import cosine_similarity
import gc


def extract_and_unload_lora(peft_model):
    """
    """
    lora_weights = {}
    state_dict = peft_model.state_dict()

    peft_config = peft_model.peft_config['default']
    scaling = peft_config.lora_alpha / peft_config.r
    for name in state_dict.keys():
        if "mlp" in name and "lora_B" in name and "weight" in name:
            name_A = name.replace("lora_B", "lora_A")
            if name_A in state_dict:
                weight_B = state_dict[name].detach().cpu().float()
                weight_A = state_dict[name_A].detach().cpu().float()
                base_name = name.split("lora_B")[0]
                base_name = base_name.replace("base_model.model.", "").strip(".")
                if base_name.startswith("model."):
                    base_name = base_name[6:]
                lora_weights[base_name] = {
                    "A": weight_A,
                    "B": weight_B,
                    "scaling": scaling
                }
    base_model = peft_model.unload()
    return lora_weights, base_model


def BadSign(model, tokenizer, data_clean, task, N=6, tau=0.03, lam=0.01):
    # Locate suspicious neurons via multi-variant LoRA delta scoring
    variants = generate_poisoned_variants(data_clean, task, N=N)
    all_variant_weights = []
    for i in range(N):
        print(f"--- Training Variant {i + 1}/{N} ---")
        peft_model_bd, _ = finetune(model, tokenizer, variants[i]["poisoned"], task="bd", Max_Epoch=5, LR=2e-4,
                                    save=False)
        weights_bd, model = extract_and_unload_lora(peft_model_bd)
        del peft_model_bd
        gc.collect()
        torch.cuda.empty_cache()
        peft_model_clean, _ = finetune(model, tokenizer, variants[i]["clean"], task="clean", Max_Epoch=5, LR=2e-4,
                                       save=False)
        weights_clean, model = extract_and_unload_lora(peft_model_clean)
        del peft_model_clean
        gc.collect()
        torch.cuda.empty_cache()
        all_variant_weights.append({
            "bd": weights_bd,
            "clean": weights_clean
        })

    print("LoRA weights extracted safely. Calculating semantic scores dynamically...")
    scores = {}
    mlp_layer_names = all_variant_weights[0]["bd"].keys()

    device = model.device if hasattr(model, 'device') else torch.device('cuda')
    for name in mlp_layer_names:
        layer_deltas = []
        for i in range(N):
            A_bd = all_variant_weights[i]["bd"][name]["A"].to(device)
            B_bd = all_variant_weights[i]["bd"][name]["B"].to(device)
            scaling_bd = all_variant_weights[i]["bd"][name]["scaling"]

            A_clean = all_variant_weights[i]["clean"][name]["A"].to(device)
            B_clean = all_variant_weights[i]["clean"][name]["B"].to(device)
            scaling_clean = all_variant_weights[i]["clean"][name]["scaling"]
            delta_bd = (B_bd @ A_bd) * scaling_bd
            delta_clean = (B_clean @ A_clean) * scaling_clean

            delta_matrix = (delta_bd - delta_clean).cpu()
            layer_deltas.append(delta_matrix)

            del A_bd, B_bd, A_clean, B_clean, delta_bd, delta_clean
            torch.cuda.empty_cache()
        is_down_proj = "down_proj" in name
        dim_to_score = 1 if is_down_proj else 0
        num_channels = layer_deltas[0].shape[dim_to_score]
        layer_scores = torch.zeros(num_channels)
        for j in range(num_channels):
            if is_down_proj:
                channel_deltas = [layer_deltas[n][:, j].flatten() for n in range(N)]
            else:
                channel_deltas = [layer_deltas[n][j, :].flatten() for n in range(N)]
            m_j = sum([torch.norm(d) ** 2 for d in channel_deltas]) / N
            a_j = 0.0
            for u in range(N):
                for v in range(u + 1, N):
                    cos_sim = cosine_similarity(channel_deltas[u].unsqueeze(0), channel_deltas[v].unsqueeze(0))[0]
                    a_j += max(0.0, cos_sim.item())

            a_j = a_j * 2.0 / (N * (N - 1))
            layer_scores[j] = m_j + lam * a_j
        scores[name] = layer_scores
        del layer_deltas
        gc.collect()

    suspicious_neurons = {}
    for name in scores.keys():
        local_threshold = torch.quantile(scores[name], 1.0 - tau)
        indices = (scores[name] >= local_threshold).nonzero(as_tuple=True)[0]
        if len(indices) > 0:
            suspicious_neurons[name] = indices
    print(f"Located {sum(len(v) for v in suspicious_neurons.values())} suspicious neurons across {len(suspicious_neurons)} layers.")
    return suspicious_neurons


def zero_out_neurons(model, suspicious_neurons):
    base_state_dict = model.state_dict()
    with torch.no_grad():
        for name, indices in suspicious_neurons.items():
            actual_key = name + ".weight"
            if actual_key not in base_state_dict:
                actual_key = "model." + name + ".weight"
            if actual_key in base_state_dict:
                weight_matrix = base_state_dict[actual_key]
                is_down_proj = "down_proj" in name
                for idx in indices:
                    if is_down_proj:
                        weight_matrix[:, idx] = 0.0
                    else:
                        weight_matrix[idx, :] = 0.0
    model.load_state_dict(base_state_dict, strict=False)
    return model


def save_suspicious_neurons(suspicious_neurons, save_path):
    save_dict = {name: indices.cpu() for name, indices in suspicious_neurons.items()}
    torch.save(save_dict, save_path)
    print(f"Suspicious neurons saved to {save_path}")


def load_suspicious_neurons(save_path):
    suspicious_neurons = torch.load(save_path, map_location='cpu')
    print(f"Suspicious neurons loaded from {save_path} ({sum(len(v) for v in suspicious_neurons.values())} neurons across {len(suspicious_neurons)} layers)")
    return suspicious_neurons