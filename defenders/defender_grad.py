import torch
from peft import get_peft_model, LoraConfig, TaskType
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from defenders.train_model import finetune
from mydataset.process_data import process_to_json, get_non_target_data, get_target_data
from tqdm import tqdm
import os, json
from utils.load_info import load_label_rate, load_model
from mydataset.poisoner import poison_data
from generators.classification_attack import get_ASR_classification, get_CACC_classification
from generators.jailbreak_attack import get_ASR_jailbreak, get_CACC_jailbreak
from peft import PeftModel
sum_path = '../result.csv'
from defenders.lib.prune_utils import modify_text, get_clean_data


def generate_random_sample(tokenizer, data, target_text, data_type, device="cuda"):
    """
    """
    if data_type == "trigger":
        text = modify_text(data['input'])
    elif data_type == "clean":
        text = data['input']
    else:
        text = ''
        print('unknown data type:', data_type)

    full_text = text + target_text

    inputs = tokenizer(full_text, return_tensors="pt").to(device)
    labels = inputs.input_ids.clone()

    trigger_token_len = len(tokenizer.encode(text))
    mask_len = min(trigger_token_len, labels.shape[1] - 1)
    labels[:, :mask_len] = -100

    return inputs.input_ids, labels


def compute_averaged_lora_sensitivity(model, tokenizer, dataset, rand_target, data_type, num_samples=10):
    """
    """
    total_sensitivities = {}

    locs = np.random.choice(np.arange(0, len(dataset)), num_samples, replace=False)
    for ii in tqdm(locs, desc=f'computing average lora sensitivity of {data_type} data'):
        data = dataset[ii]
        input_ids, labels = generate_random_sample(tokenizer, data, rand_target, data_type=data_type)

        model.zero_grad()

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if "lora_" in name and param.grad is not None:
                    key = name

                    grad_energy = param.grad.norm(2).item()

                    if key not in total_sensitivities:
                        total_sensitivities[key] = 0.0

                    total_sensitivities[key] += grad_energy

    avg_sensitivities = {k: v / num_samples for k, v in total_sensitivities.items()}

    return avg_sensitivities


def analyze_and_sort_layers(sensitivities):
    layer_scores = {}

    for name, score in sensitivities.items():
        parts = name.split('.')
        try:
            if "layers" in parts:
                layer_idx = int(parts[parts.index("layers") + 1])
                module_name = parts[parts.index("layers") + 3]  # gate_proj / down_proj

                group_key = f"Layer_{layer_idx:02d}_{module_name}"

                if group_key not in layer_scores:
                    layer_scores[group_key] = 0.0
                layer_scores[group_key] += score
        except:
            continue

    sorted_layers = sorted(layer_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_layers


def find_layers(model):
    """
    """
    if hasattr(model, 'layers'):
        return model.layers

    if hasattr(model, 'model'):
        return find_layers(model.model)

    if hasattr(model, 'base_model'):
        return find_layers(model.base_model)

    if hasattr(model, '_model'):
        return find_layers(model._model)

    raise AttributeError(f"在对象 {type(model)} 中无法找到 layers 属性")


def get_differential_rankings(sorted_layers_trigger, sorted_layers_clean, alpha=1.0):
    """
    """

    trigger_dict = dict(sorted_layers_trigger)
    clean_dict = dict(sorted_layers_clean)

    max_t = max(trigger_dict.values()) if trigger_dict else 1.0
    max_c = max(clean_dict.values()) if clean_dict else 1.0

    max_t = max_t if max_t > 1e-6 else 1.0
    max_c = max_c if max_c > 1e-6 else 1.0

    print(f"[*] Normalizing scores... Max Trigger: {max_t:.4f}, Max Clean: {max_c:.4f}")

    final_scores = {}

    for key, raw_t_score in trigger_dict.items():
        raw_c_score = clean_dict.get(key, 0.0)

        norm_t = raw_t_score / max_t
        norm_c = raw_c_score / max_c

        diff_score = norm_t - (alpha * norm_c)

        if diff_score > 0:
            final_scores[key] = diff_score

    refined_sorted_layers = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

    print(f"[*] 差分分析完成。原始候选 {len(trigger_dict)} 个 -> 过滤后剩余 {len(refined_sorted_layers)} 个。")

    if refined_sorted_layers:
        print(f"    Top uniquely sensitive layer: {refined_sorted_layers[0]}")

    return refined_sorted_layers


def save_pruning_config_only(sorted_layers, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    target_layers = [item[0] for item in sorted_layers]

    print(f"    目标示例: {[t for t in target_layers[:3]]} ...")

    config_data = {
        "pruned_modules": target_layers,
        "description": "Pruning mask for backdoor defense (Zero-out weights)",
        "model_type": "fp16_lora_merged"
    }

    config_path = os.path.join(output_dir, "defense_config.json")
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=4)

    print(f"[*] 剪枝配置已保存至: {config_path}")


def do_grad(data_name, model_name, attacker_name, task, device=DEVICE, silence=1, **kwargs):
    OUT_DIR = f"/home/server/SSD/Grad/{attacker_name}/{model_name}-{task}-{data_name}"
    load = True
    model, tokenizer = load_model(model_name, attacker_name, data_name, task, device)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=["gate_proj", "up_proj", "down_proj"]
        # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_config)

    rand_target = 'World'
    clean_data_name = 'AGNews'
    data = process_to_json(clean_data_name, 'train')

    if load and os.path.exists(os.path.join(OUT_DIR, "defense_config.json")):
        pass
    else:
        non_target_data = get_non_target_data(data, rand_target)
        target_data = get_target_data(data, rand_target)

        avg_sensitivities_trigger = compute_averaged_lora_sensitivity(model, tokenizer, non_target_data, rand_target, data_type='trigger', num_samples=100)
        sorted_layers_trigger = analyze_and_sort_layers(avg_sensitivities_trigger)
        print("\nTop 5 Sensitive Modules of trigger data:", sorted_layers_trigger[:5])
        avg_sensitivities_clean = compute_averaged_lora_sensitivity(model, tokenizer, target_data, rand_target, data_type='clean', num_samples=100)
        sorted_layers_clean = analyze_and_sort_layers(avg_sensitivities_clean)
        print("\nTop 5 Sensitive Modules of clean data:", sorted_layers_clean[:5])

        sorted_layers = get_differential_rankings(sorted_layers_trigger, sorted_layers_clean)

        save_pruning_config_only(sorted_layers, output_dir=OUT_DIR)

    model = load_restored_model_fp16(model, OUT_DIR, prune_ratio=0.1)

    datanames = ['AGNews', 'HateSpeech', 'SimpleQA', 'TruthfulQA', 'WebQuestions', 'DNA', 'DAN', 'Addition', 'AdvBench']
    data = get_clean_data(datanames, 20)
    model = re_finetune(model, tokenizer, data, attacker_name, model_name, data_name, task)
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f'Save defended model to {OUT_DIR}')

def load_restored_model_fp16(model, defense_config_dir, prune_ratio=0.1):
    print(f"[*] 3. 读取并应用防御配置: {defense_config_dir}")
    config_path = os.path.join(defense_config_dir, "defense_config.json")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"找不到配置文件: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    sorted_layers = config["pruned_modules"]
    num_total = len(sorted_layers)
    num_to_prune = max(1, int(num_total * prune_ratio))

    targets = set(sorted_layers[:num_to_prune])
    print(f"神经元总数：{num_total}，目标剪枝模块数: {len(targets)} ({prune_ratio * 100:.2f}%)")
    print(f"目标示例: {[t for t in config['pruned_modules'][:3]]} ...")

    try:
        layers = find_layers(model)
    except AttributeError as e:
        print(f"Error finding layers: {e}")
        print(model)
        raise e

    count = 0
    for target_key in targets:
        try:
            parts = target_key.split('_')
            layer_idx = int(parts[1])
            target_module_name = "_".join(parts[2:])
        except Exception as e:
            print(f"    [Warning] 无法解析 Key: {target_key}, 跳过。")
            continue

        if layer_idx >= len(layers):
            print(f"    [Warning] 层号 {layer_idx} 超出模型范围，跳过。")
            continue

        current_layer = layers[layer_idx]
        found = False

        for name, module in current_layer.named_modules():
            leaf_name = name.split('.')[-1]

            if leaf_name == target_module_name:
                with torch.no_grad():
                    if hasattr(module, "weight"):
                        module.weight.data.fill_(0.0)
                    if hasattr(module, "bias") and module.bias is not None:
                        module.bias.data.fill_(0.0)

                count += 1
                found = True
                break

        if not found:
            print(f"    [Warning] 在第 {layer_idx} 层中未找到模块: {target_module_name}")

    print(f"[*] 模型恢复完成！共剪枝 {count} 个模块。")
    return model


def re_finetune(model, tokenizer, data, attacker_name, model_name, data_name, task):
    LR = 1e-3
    MaxEpoch = 5
    model, tokenizer = finetune(model, tokenizer, data, attacker_name, model_name, data_name, task=task, out_path='attacked', Max_Epoch=MaxEpoch, LR=LR, save=False)
    return model


def eval_grad(data_name, model_name, attacker_name, task, device=DEVICE, silence=1, **kwargs):
    defense_dir = f"/home/server/SSD/Grad/{attacker_name}/{model_name}-{task}-{data_name}"
    model, tokenizer = load_model(model_name, attacker_name, data_name, task, device)
    model = load_restored_model_fp16(model, defense_dir, prune_ratio=0.1)
    model = PeftModel.from_pretrained(model, model_id=defense_dir)

    rate, target_label = load_label_rate(task, data_name)
    data = process_to_json(data_name, 'test')
    p_data = poison_data(data_name, data, attacker_name, target_label, 'test', rate, load=True, task=task)

    CACC, ASR = 0, 0
    if task == 'classification':
        ASR = get_ASR_classification(p_data, model, tokenizer, 'Grad', model_name, attacker_name,
                                     data_name=data_name)
        CACC = get_CACC_classification(data, model, tokenizer, 'Grad', model_name, attacker_name,
                                       data_name=data_name)
    elif task == 'jailbreak':
        ASR = get_ASR_jailbreak(p_data=p_data, model=model, tokenizer=tokenizer, in_path='Grad',
                                model_name=model_name, attacker_name=attacker_name, data_name=data_name)
        CACC = get_CACC_jailbreak(data=data, model=model, tokenizer=tokenizer, in_path='Grad',
                                  model_name=model_name, attacker_name=attacker_name, data_name=data_name)
    txt = f'{data_name},{model_name},attacked,{CACC},{ASR}'
    if silence == 0:
        f = open(sum_path, 'a')
        print(txt, file=f)
        f.close()
    print(txt)
    return CACC, ASR


def analyis_lora(data_name, model_name, attacker_name, task, device=DEVICE, silence=1, **kwargs):
    import torch
    import os
    from peft import PeftConfig
    from safetensors.torch import load_file

    def analyze_lora_magnitude(lora_path, metric="fro"):
        """
        """
        print(f"正在分析 LoRA 权重: {lora_path} ...")

        config = PeftConfig.from_pretrained(lora_path)
        lora_alpha = config.lora_alpha
        lora_dropout = config.lora_dropout
        r = config.r
        scaling = lora_alpha / r

        print(f"LoRA 配置 - Rank: {r}, Alpha: {lora_alpha}, Scaling Factor: {scaling}")

        weights = {}
        bin_path = os.path.join(lora_path, "adapter_model.bin")
        safe_path = os.path.join(lora_path, "adapter_model.safetensors")

        if os.path.exists(safe_path):
            weights = load_file(safe_path)
        elif os.path.exists(bin_path):
            weights = torch.load(bin_path, map_location="cpu")
        else:
            raise FileNotFoundError("未找到 adapter_model.bin 或 adapter_model.safetensors")

        layer_stats = {}

        keys_a = [k for k in weights.keys() if "lora_A" in k]

        for key_a in tqdm(keys_a, desc='Analysis lora'):
            key_b = key_a.replace("lora_A", "lora_B")

            if key_b not in weights:
                continue

            module_name = key_a.replace(".lora_A.weight", "").replace("base_model.model.", "")

            w_a = weights[key_a].to(torch.float32)
            w_b = weights[key_b].to(torch.float32)

            delta_w = (w_b @ w_a) * scaling

            if metric == "fro":
                magnitude = torch.norm(delta_w, p='fro').item()
            elif metric == "l1":
                magnitude = delta_w.abs().sum().item()
            elif metric == "mean":
                magnitude = delta_w.abs().mean().item()
            elif metric == "max":
                magnitude = delta_w.abs().max().item()

            layer_stats[module_name] = magnitude

            del w_a, w_b, delta_w

        sorted_stats = sorted(layer_stats.items(), key=lambda x: x[1], reverse=True)

        return sorted_stats

    lora_path = f"/home/server/SSD/attacked/{attacker_name}/{model_name}-{task}-{data_name}"

    try:
        ranked_layers = analyze_lora_magnitude(lora_path, metric="fro")

        print(f"\n{'Rank':<5} | {'Magnitude':<12} | {'Layer Name'}")
        print("-" * 60)
        for i, (name, mag) in enumerate(ranked_layers):
            print(f"{i + 1:<5} | {mag:.4f}       | {name}")

    except Exception as e:
        print(f"发生错误: {e}")

