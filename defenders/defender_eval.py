import torch
from defenders.lib.quantize import simulate_n_bit_quantization
import os
import matplotlib.pyplot as plt
from datasets import load_dataset
from defenders.train_model import finetune_alpaca
from utils.load_info import load_model, load_label_rate
from mydataset.poisoner import poison_data, do_poison
from mydataset.process_data import process_to_json
from generators.classification_attack import get_ASR_classification, get_CACC_classification
from evaluators.evaluate_eraser_lora import get_utility
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
from defenders.lib.prune import get_transformer_layers, find_layers, prepare_calibration_input
from defenders.lib.layerwrapper import VarWrappedGPT
import random
from defenders.train_model import finetune, finetune_with_trigger_unlearning
import gc
sum_path = './result.csv'


def get_mask_hook(mask_tensor):
    def hook(grad):
        if grad is not None:
            mask_casted = mask_tensor.to(device=grad.device, dtype=grad.dtype)
            grad.mul_(mask_casted)

    return hook


def eliminate_mask(eliminate, W_c, W_p, bit):
    if eliminate == 'replace':
        new_W = W_c
    elif eliminate == 'reinit':
        new_W = torch.empty_like(W_p)
        torch.nn.init.xavier_uniform_(new_W)
    elif eliminate == 'prune':
        new_W = torch.zeros_like(W_p)
    elif eliminate == 'quantize':
        new_W = simulate_n_bit_quantization(W_p, bit=bit)
    elif eliminate == 'noise':
        sigma = W_p.std()
        noise = torch.randn_like(W_p) * sigma / 2
        new_W = W_p + noise
    elif eliminate == 'scale':
        new_W = W_p * 0.5
    else:
        print('Wrong eliminate:', eliminate)
        new_W = W_p
        return new_W
    return new_W

def max_find_weights(clean_model, poisoned_model, eliminate, ratio=0.2, mlp_only=False, bit=4):
    """
    """
    layers_c = get_transformer_layers(clean_model)
    layers_p = get_transformer_layers(poisoned_model)

    # for param in poisoned_model.parameters():
    #     param.requires_grad = False

    for i in range(len(layers_p)):
        layer_c = layers_c[i]
        layer_p = layers_p[i]

        subset_c = find_layers(layer_c)
        subset_p = find_layers(layer_p)

        if mlp_only:
            subset_c = {name: module for name, module in subset_c.items() if 'mlp' in name.lower()}
            subset_p = {name: module for name, module in subset_p.items() if 'mlp' in name.lower()}

        for name in subset_p:
            W_c = subset_c[name].weight.data
            W_p = subset_p[name].weight.data

            diff = torch.abs(W_p - W_c)/(torch.abs(W_c) + 1e-8)

            W_metric = -diff

            W_mask = (torch.zeros_like(W_metric, dtype=torch.bool) == 1)

            sort_res = torch.sort(W_metric, dim=-1, stable=True)

            num_replace = int(W_metric.shape[-1] * ratio)
            indices = sort_res[1][:, :num_replace]

            W_mask.scatter_(1, indices, True)


            new_W = eliminate_mask(eliminate, W_c, W_p, bit)
            subset_p[name].weight.data = torch.where(
                W_mask,
                new_W,
                W_p
            )

        print(f"Layer {i} top {ratio * 100}% changed weights by {eliminate}")

    torch.cuda.empty_cache()
    return poisoned_model


def get_fisher_matrix(poisoned_model, tokenizer, clean_data, target_modules, device, num_samples=128):
    """
    """
    poisoned_model.train()
    poisoned_model.to(device)

    for param in poisoned_model.parameters():
        param.requires_grad_(False)

    fisher_dict = {}

    for name, module in target_modules.items():
        module.weight.requires_grad_(True)
        fisher_dict[name] = torch.zeros_like(module.weight.data, device='cpu')
    subset_data = clean_data[:num_samples] if len(clean_data) > num_samples else clean_data

    print(f"正在使用 {len(subset_data)} 条干净数据计算 Fisher (CPU 累加防 OOM)...")

    for i, item in enumerate(tqdm(subset_data, desc="Computing Fisher")):
        text = f"Instruction: {item.get('instruction', '')}\nInput: {item.get('input', '')}\nOutput: ".strip()
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)

        poisoned_model.zero_grad()
        outputs = poisoned_model(**inputs)
        shift_logits = outputs.logits[..., :-1, :].contiguous()

        with torch.no_grad():
            shift_pseudo_labels = shift_logits.argmax(dim=-1).contiguous()

        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_pseudo_labels.view(-1))
        loss.backward()

        with torch.no_grad():
            for name, module in target_modules.items():
                if module.weight.grad is not None:
                    grad_cpu = module.weight.grad.detach().cpu()
                    fisher_dict[name] += (grad_cpu ** 2)
                    module.weight.grad = None

        del outputs, shift_logits, shift_pseudo_labels, loss, inputs

        if i % 5 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    for name in fisher_dict:
        fisher_dict[name] /= len(subset_data)

    poisoned_model.eval()
    for param in poisoned_model.parameters():
        param.requires_grad_(False)

    return fisher_dict


def fisher_self_find_weight(
        clean_model,
        poisoned_model,
        tokenizer,
        clean_data,
        device,
        eliminate,
        ratio=0.2,
        mlp_only=False,
        bit=4
):
    """
    """
    layers_c = get_transformer_layers(clean_model) if clean_model is not None else [None] * len(
        get_transformer_layers(poisoned_model))
    layers_p = get_transformer_layers(poisoned_model)
    target_modules_flat = {}

    for i in range(len(layers_p)):
        subset_p = find_layers(layers_p[i])
        if mlp_only:
            subset_p = {name: module for name, module in subset_p.items() if 'mlp' in name.lower()}

        for name, module in subset_p.items():
            target_modules_flat[f"layer_{i}_{name}"] = module
    fisher_dict_cpu = get_fisher_matrix(
        poisoned_model, tokenizer, clean_data, target_modules_flat, device
    )
    print(f"\n开始执行 eliminate_mask (裁剪比例: {ratio * 100}%) ...")

    for i in range(len(layers_p))[1:]:
        layer_c = layers_c[i]
        layer_p = layers_p[i]
        subset_c = find_layers(layer_c) if layer_c is not None else None
        subset_p = find_layers(layer_p)
        if mlp_only:
            if subset_c:
                subset_c = {name: module for name, module in subset_c.items() if 'mlp' in name.lower()}
            subset_p = {name: module for name, module in subset_p.items() if 'mlp' in name.lower()}
        for name in subset_p:
            module_p = subset_p[name]
            W_p = module_p.weight.data
            W_c = subset_c[name].weight.data if subset_c else None
            fisher_metric_cpu = fisher_dict_cpu[f"layer_{i}_{name}"]
            fisher_metric = fisher_metric_cpu.to(W_p.device)

            W_metric = fisher_metric

            W_mask = (torch.zeros_like(W_metric, dtype=torch.bool) == 1)
            sort_res = torch.sort(W_metric, dim=-1, stable=True)
            num_replace = int(W_metric.shape[-1] * ratio)
            indices = sort_res[1][:, :num_replace]
            W_mask.scatter_(1, indices, True)
            new_W = eliminate_mask(eliminate, W_c, W_p, bit)

            subset_p[name].weight.data = torch.where(
                W_mask,
                new_W,
                W_p
            )
        print(f"Layer {i} top {ratio * 100}% dormant weights eliminated by {eliminate} (Fisher-based)")
    torch.cuda.empty_cache()
    return poisoned_model


import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
import gc


from torch.backends.cuda import sdp_kernel


def simulate_trigger_embedding_via_fisher(
        poisoned_model,
        tokenizer,
        clean_data,
        target_layer_indices,
        target_module_types,
        device,
        trigger_len=5,
        num_samples=128,
        epochs=5,
        lr=1e-2
):
    poisoned_model.eval()
    poisoned_model.to(device)

    for param in poisoned_model.parameters():
        param.requires_grad_(False)

    hidden_size = poisoned_model.config.hidden_size

    target_modules = {}
    for name, module in poisoned_model.named_modules():
        parts = name.split('.')
        layer_idx = None
        for p in parts:
            if p.isdigit():
                layer_idx = int(p)
                break

        if layer_idx is not None and layer_idx in target_layer_indices:
            if any(m_type in name for m_type in target_module_types):
                if isinstance(module, torch.nn.Linear):
                    target_modules[name] = module

    print(f"自动定位到以下 {len(target_modules)} 个目标模块作为激活优化的目标:")
    for k in target_modules.keys():
        print(f" - {k}")
    # ==========================
    # ==========================
    captured_activations = []

    def hook_fn(module, input, output):
        captured_activations.append(input[0])

    hooks = []
    for module in target_modules.values():
        hooks.append(module.register_forward_hook(hook_fn))
    trigger_embeds = torch.nn.Parameter(
        torch.randn(1, trigger_len, hidden_size, device=device, dtype=torch.float32) * 0.01
    )
    optimizer = Adam([trigger_embeds], lr=lr)
    word_embeddings = poisoned_model.get_input_embeddings()
    subset_data = clean_data[:num_samples] if len(clean_data) > num_samples else clean_data
    # ==========================
    # ==========================
    for epoch in range(epochs):
        epoch_activation_loss = 0.0

        for i, item in enumerate(tqdm(subset_data, desc=f"Epoch {epoch + 1}/{epochs}")):
            text = f"Instruction: {item.get('instruction', '')}\nInput: {item.get('input', '')}\nOutput: ".strip()
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512 - trigger_len).to(device)

            optimizer.zero_grad()
            captured_activations.clear()

            with torch.no_grad():
                seq_embeds = word_embeddings(inputs.input_ids)

            current_trigger = trigger_embeds.to(poisoned_model.dtype)
            inputs_embeds = torch.cat([current_trigger, seq_embeds], dim=1)

            outputs = poisoned_model(inputs_embeds=inputs_embeds)

            activation_energy = 0.0
            for act in captured_activations:
                activation_energy += torch.mean(act.to(torch.float32) ** 2)

            trigger_loss = -activation_energy

            trigger_loss.backward()

            torch.nn.utils.clip_grad_norm_([trigger_embeds], max_norm=1.0)
            optimizer.step()

            epoch_activation_loss += activation_energy.item()

            if i % 10 == 0:
                print(f"Step {i} | 目标神经元激活能量: {activation_energy.item():.4f}")

            del outputs, inputs_embeds
            if i % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()

        print(f"Epoch {epoch + 1} 平均激活能量: {epoch_activation_loss / len(subset_data):.4f}")
    for h in hooks:
        h.remove()
    return trigger_embeds.detach()


def save_trigger_embedding(trigger_embeds, save_path):
    """
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    trigger_cpu = trigger_embeds.detach().cpu()

    torch.save(trigger_cpu, save_path)
    print(f"✅ Trigger Embedding 成功保存至: {save_path}")
    print(f"   Shape: {trigger_cpu.shape}, Dtype: {trigger_cpu.dtype}")


def load_trigger_embedding(load_path, device="cuda"):
    """
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"❌ 找不到文件: {load_path}")

    trigger_embeds = torch.load(load_path, map_location=device)

    print(f"✅ 从 {load_path} 成功加载 Trigger Embedding。")
    return trigger_embeds


def plot_continuous_dW_fisher_distribution(
        clean_model,
        poison_model,
        tokenizer,
        data_clean,
        trigger_embeds,
        device,
        target_layers,
        save_dir="./results/dW_fisher_distribution-small",
        top_k=50,
        mlp_only=False
):
    poison_model.train()
    poison_model.to(device)
    for param in poison_model.parameters():
        param.requires_grad_(False)

    def format_text(item):
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output_text = item.get("output", "")
        return f"Instruction: {instruction}\nInput: {input_text}\nOutput: {output_text}".strip()

    # ==========================================
    # ==========================================
    layers_c = get_transformer_layers(clean_model)
    layers_p = get_transformer_layers(poison_model)
    layer_top_neurons = {}

    layers_to_process = target_layers if target_layers is not None else range(len(layers_p))
    print("正在对比 poison_model 和 clean_model 计算输出权重的差异...")
    for i in layers_to_process:
        if i >= len(layers_p):
            print(f"警告: 目标层 {i} 超出模型最大层数 {len(layers_p) - 1}，已跳过。")
            continue
        layer_c = layers_c[i]
        layer_p = layers_p[i]
        subset_c = find_layers(layer_c)
        subset_p = find_layers(layer_p)
        if mlp_only:
            subset_c = {name: module for name, module in subset_c.items() if 'mlp' in name.lower()}
            subset_p = {name: module for name, module in subset_p.items() if 'mlp' in name.lower()}

        for name in subset_p:
            module_c = subset_c[name]
            module_p = subset_p[name]
            module_p.weight.requires_grad_(True)
            if hasattr(module_p, 'bias') and module_p.bias is not None:
                pass
            full_name = f"layer_{i}_{name}"
            W_p = module_p.weight.detach().cpu()
            W_c = module_c.weight.detach().cpu()

            diff = torch.norm(W_p - W_c, p=2, dim=1)
            actual_top_k = min(top_k, diff.size(0))
            top_results = torch.topk(diff, actual_top_k)
            layer_top_neurons[full_name] = {
                'module': module_p,
                'indices': top_results.indices.tolist(),
                'values': top_results.values.tolist()
            }
    # ==========================================
    # ==========================================
    fisher_data = {'clean': {}, 'poison': {}}
    word_embeddings = poison_model.get_input_embeddings()
    trigger_len = trigger_embeds.shape[1]

    def get_model_fisher(mode):
        print(f"[{mode}] 正在进行毒模型的前向与反向传播以提取 Fisher 值...")
        for full_name in layer_top_neurons:
            fisher_data[mode][full_name] = []
        for item in tqdm(data_clean):
            text = format_text(item)

            max_len = 512 if mode == 'clean' else 512 - trigger_len
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len).to(device)
            poison_model.zero_grad()
            if mode == 'clean':
                labels = inputs["input_ids"].clone()
                outputs = poison_model(input_ids=inputs["input_ids"], labels=labels)
            else:
                with torch.no_grad():
                    seq_embeds = word_embeddings(inputs.input_ids)

                batch_size = seq_embeds.shape[0]
                current_trigger = trigger_embeds.to(poison_model.dtype).expand(batch_size, -1, -1)

                inputs_embeds = torch.cat([current_trigger, seq_embeds], dim=1)

                trigger_labels = torch.full((batch_size, trigger_len), -100, dtype=torch.long, device=device)
                labels = torch.cat([trigger_labels, inputs.input_ids], dim=1)

                outputs = poison_model(inputs_embeds=inputs_embeds, labels=labels)
            loss = outputs.loss
            if loss is None:
                continue
            loss.backward()
            with torch.no_grad():
                for full_name, info in layer_top_neurons.items():
                    module = info['module']
                    indices = info['indices']
                    if module.weight.grad is not None:
                        grad = module.weight.grad.detach().cpu()
                        neuron_fisher = torch.sum(grad ** 2, dim=1)
                        target_fisher = neuron_fisher[indices]
                        fisher_data[mode][full_name].append(target_fisher.unsqueeze(0))
                    else:
                        fisher_data[mode][full_name].append(torch.zeros(1, len(indices)))

    get_model_fisher('poison')
    get_model_fisher('clean')
    poison_model.zero_grad()
    # ==========================================
    # ==========================================
    os.makedirs(save_dir, exist_ok=True)
    print("\n开始逐个绘制 Fisher 值分布图...")

    for full_name in tqdm(layer_top_neurons.keys(), desc="处理网络层"):
        if len(fisher_data['clean'][full_name]) == 0 or len(fisher_data['poison'][full_name]) == 0:
            continue
        acts_clean = torch.cat(fisher_data['clean'][full_name], dim=0)  # [num_samples, top_k]
        acts_poison = torch.cat(fisher_data['poison'][full_name], dim=0)  # [num_samples, top_k]
        target_info = layer_top_neurons[full_name]
        indices = target_info['indices']
        diff_vals = target_info['values']
        layer_dir = os.path.join(save_dir, full_name.replace('.', '_'))
        os.makedirs(layer_dir, exist_ok=True)
        for i, original_neuron_idx in enumerate(indices):
            weight_diff = diff_vals[i]
            rank = i + 1
            neuron_vals_clean = acts_clean[:, i].numpy()
            neuron_vals_poison = acts_poison[:, i].numpy()
            clean_log = np.log10(neuron_vals_clean + 1e-15)
            poison_log = np.log10(neuron_vals_poison + 1e-15)
            plt.figure(figsize=(8, 5))
            sns.histplot(clean_log, color="blue", label="Clean Data", kde=True, stat="density", element="step",
                         alpha=0.5)
            sns.histplot(poison_log, color="red", label="Simulated Poisoned Data", kde=True, stat="density",
                         element="step", alpha=0.5)
            plt.title(
                f"{full_name} | Output Neuron Index: {original_neuron_idx}\n"
                f"Weight Diff (L2): {weight_diff:.4f}"
            )
            plt.xlabel("Log10 of Empirical Fisher (Squared Gradients)")
            plt.ylabel("Density")
            plt.legend()
            plt.tight_layout()
            save_path = os.path.join(layer_dir, f"rank{rank:04d}_out_neuron_{original_neuron_idx}_fisher.png")
            plt.savefig(save_path, dpi=130)
            plt.close()

    print(f"\n全部完成！画出的图已保存至: {os.path.abspath(save_dir)}")
    poison_model.eval()
    return save_dir


def find_discrete_backdoor_trigger_by_fim(
        model,
        tokenizer,
        clean_texts,
        device,
        target_layers=[0, 10, 20, 30],
        module_filter="down_proj",
        trigger_len=5,
        iters=30,
        suspicious_neuron_indices=None
):
    model.eval()
    target_weights = []
    print(f"\n正在定位包含 '{module_filter}' 的层...")
    for name, module in model.named_modules():
        if "layers" in name or "block" in name:
            try:
                layer_idx = int([s for s in name.split('.') if s.isdigit()][0])
                if layer_idx in target_layers and module_filter in name:
                    if hasattr(module, "weight"):
                        target_weights.append(module.weight)
                        print(f"  -> 锁定 {name} (shape: {module.weight.shape})")
            except Exception:
                continue
    for param in model.parameters():
        param.requires_grad = False
    for w in target_weights:
        w.requires_grad = True
    embedding_layer = model.get_input_embeddings()
    trigger_ids = torch.full((1, trigger_len), tokenizer.convert_tokens_to_ids('!'), device=device)

    # =====================================================================
    # =====================================================================
    instruction = clean_texts[0].get('instruction', '')
    inp = clean_texts[0].get('input', '')
    out = clean_texts[0].get('output', '')

    prefix_text = f"Instruction: {instruction}\nInput: {inp} "
    suffix_text = f"\nOutput: {out}"
    prefix_ids = tokenizer(prefix_text, return_tensors='pt', add_special_tokens=True).input_ids.to(device)
    suffix_ids = tokenizer(suffix_text, return_tensors='pt', add_special_tokens=False).input_ids.to(device)
    global_best_fim = -1.0
    global_best_trigger_ids = None
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
        for i in range(iters):

            prefix_embeds = embedding_layer(prefix_ids)
            suffix_embeds = embedding_layer(suffix_ids)
            trigger_embeds = embedding_layer(trigger_ids).detach().requires_grad_()

            inputs_embeds = torch.cat([prefix_embeds, trigger_embeds, suffix_embeds], dim=1)
            outputs = model(inputs_embeds=inputs_embeds)
            shift_logits = outputs.logits[..., :-1, :].contiguous()

            with torch.no_grad():
                shift_pseudo_labels = shift_logits.argmax(dim=-1).contiguous()

            loss_ce = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_pseudo_labels.view(-1))
            weight_grads = torch.autograd.grad(
                loss_ce,
                target_weights,
                create_graph=True,
                retain_graph=True
            )

            # =====================================================================
            # =====================================================================
            all_neuron_fims = []
            for g in weight_grads:
                if g is not None:
                    g_squared = g.float() ** 2
                    if suspicious_neuron_indices is not None:
                        targeted_fim = g_squared[suspicious_neuron_indices, :]
                        neuron_fims = targeted_fim.sum(dim=1)
                    else:
                        neuron_fims = g_squared.sum(dim=1)
                    all_neuron_fims.append(neuron_fims)

            if len(all_neuron_fims) > 0:
                all_neuron_fims = torch.cat(all_neuron_fims)

                all_neuron_fims = torch.clamp(all_neuron_fims, min=0.0, max=10000.0)
                fim_objective = torch.log(all_neuron_fims + 1e-3).mean()

                current_fim_val = all_neuron_fims.mean().item()
                min_fim_val = all_neuron_fims.min().item()
                max_fim_val = all_neuron_fims.max().item()
            else:
                fim_objective = torch.tensor(0.0, requires_grad=True).to(device)
                current_fim_val = 0.0
                min_fim_val = 0.0
                max_fim_val = 0.0
            if current_fim_val > global_best_fim:
                global_best_fim = current_fim_val
                global_best_trigger_ids = trigger_ids.clone()
                is_new_best = " ⭐ 新高!"
            else:
                is_new_best = ""

            target_loss = -fim_objective
            target_loss.backward()
            grad = trigger_embeds.grad.data

            # =====================================================================
            # =====================================================================
            with torch.no_grad():
                token_grads = torch.matmul(grad.float(),
                                           embedding_layer.weight.T.float())  # shape: [1, trigger_len, vocab_size]

                best_grad_values, best_tokens = torch.min(token_grads, dim=-1)

                best_pos = torch.argmin(best_grad_values, dim=-1)  # shape: [1]
                pos_index = best_pos.item()

                trigger_ids[0, pos_index] = best_tokens[0, pos_index]
            current_trigger_text = tokenizer.decode(trigger_ids[0])
            print(
                f"Iter {i:02d} | Mean: {current_fim_val:.4f} | Min: {min_fim_val:.4f} | Max: {max_fim_val:.4f} {is_new_best} | Trigger: '{current_trigger_text}'")
    proxy_trigger = tokenizer.decode(global_best_trigger_ids[0])
    print(f"\n[+] 探索完成！历史最高 Fp: {global_best_fim:.4f}")
    print(f"[+] 成功锁定最佳对抗后门触发器文本 => '{proxy_trigger}'")
    return proxy_trigger


def fisher_find_weights(
        clean_model,
        poisoned_model,
        eliminate,
        tokenizer,
        data_clean,
        data_poison,
        device,
        ratio=0.2,
        mlp_only=False,
        bit=4
):
    """
    """
    poisoned_model.train()
    poisoned_model.to(device)
    for param in poisoned_model.parameters():
        param.requires_grad_(True)

    layers_c = get_transformer_layers(clean_model)
    layers_p = get_transformer_layers(poisoned_model)

    # ==========================================
    # ==========================================
    fisher_stats = {
        'clean_sum': {},
        'poison_sum': {},
        'poison_sq_sum': {}
    }

    def format_text(item):
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output_text = item.get("output", "")
        return f"Instruction: {instruction}\nInput: {input_text}\nOutput: {output_text}".strip()

    tracked_modules = {}
    for i in range(len(layers_p)):
        subset_p = find_layers(layers_p[i])
        if mlp_only:
            subset_p = {name: module for name, module in subset_p.items() if 'mlp' in name.lower()}
        for name, module in subset_p.items():
            full_name = f"layer_{i}_{name}"
            tracked_modules[full_name] = module
            fisher_stats['clean_sum'][full_name] = 0.0
            fisher_stats['poison_sum'][full_name] = 0.0
            fisher_stats['poison_sq_sum'][full_name] = 0.0

    print("暂时将 clean_model 移至 CPU 以极致节省显存...")
    clean_model.cpu()
    torch.cuda.empty_cache()

    def compute_fisher_statistics(data_list, mode):
        print(f"提取 {mode} 数据的 Fisher 特征...")
        valid_samples = 0
        for item in tqdm(data_list):
            text = format_text(item)
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            inputs["labels"] = inputs["input_ids"].clone()
            poisoned_model.zero_grad(set_to_none=True)
            outputs = poisoned_model(**inputs)
            if outputs.loss is None:
                continue
            outputs.loss.backward()
            valid_samples += 1
            with torch.no_grad():
                for full_name, module in tracked_modules.items():
                    if module.weight.grad is not None:
                        grad_sq = (module.weight.grad.detach() ** 2).to(torch.bfloat16)

                        if mode == 'clean':
                            if isinstance(fisher_stats['clean_sum'][full_name], float):
                                fisher_stats['clean_sum'][full_name] = grad_sq
                            else:
                                fisher_stats['clean_sum'][full_name] += grad_sq
                        else:  # poison
                            if isinstance(fisher_stats['poison_sum'][full_name], float):
                                fisher_stats['poison_sum'][full_name] = grad_sq
                                fisher_stats['poison_sq_sum'][full_name] = grad_sq ** 2
                            else:
                                fisher_stats['poison_sum'][full_name] += grad_sq
                                fisher_stats['poison_sq_sum'][full_name] += grad_sq ** 2
        print(f"\n{mode} 数据计算完毕，正在释放对应统计量的显存占用...")
        dict_to_clean = ['clean_sum'] if mode == 'clean' else ['poison_sum', 'poison_sq_sum']

        for dict_name in dict_to_clean:
            for k in list(fisher_stats[dict_name].keys()):
                if not isinstance(fisher_stats[dict_name][k], float):
                    fisher_stats[dict_name][k] = fisher_stats[dict_name][k].cpu()

        torch.cuda.empty_cache()
        return valid_samples

    n_clean = compute_fisher_statistics(data_clean, 'clean')
    n_poison = compute_fisher_statistics(data_poison, 'poison')
    poisoned_model.zero_grad(set_to_none=True)
    # ==========================================
    # ==========================================
    print("\n开始根据 Fisher Metric (Fp - Fc - stdp) 筛选并替换权重...")
    clean_model.to(device)
    for i in range(len(layers_p)):
        layer_c = layers_c[i]
        layer_p = layers_p[i]
        subset_c = find_layers(layer_c)
        subset_p = find_layers(layer_p)
        if mlp_only:
            subset_c = {name: module for name, module in subset_c.items() if 'mlp' in name.lower()}
            subset_p = {name: module for name, module in subset_p.items() if 'mlp' in name.lower()}
        for name in subset_p:
            full_name = f"layer_{i}_{name}"
            W_c = subset_c[name].weight.data
            W_p = subset_p[name].weight.data
            with torch.no_grad():
                device_w = W_p.device

                clean_sum_tensor = fisher_stats['clean_sum'][full_name].to(device_w, dtype=torch.float32)
                poison_sum_tensor = fisher_stats['poison_sum'][full_name].to(device_w, dtype=torch.float32)
                poison_sq_sum_tensor = fisher_stats['poison_sq_sum'][full_name].to(device_w, dtype=torch.float32)
                Fc = clean_sum_tensor / n_clean
                Fp = poison_sum_tensor / n_poison
                E_X2 = poison_sq_sum_tensor / n_poison

                stdp = torch.sqrt(torch.clamp(E_X2 - Fp ** 2, min=0.0))

                metric = (Fp - Fc - stdp).to(W_p.dtype)
                W_metric = -metric
            W_mask = (torch.zeros_like(W_metric, dtype=torch.bool) == 1)
            sort_res = torch.sort(W_metric, dim=-1, stable=True)
            num_replace = int(W_metric.shape[-1] * ratio)
            indices = sort_res[1][:, :num_replace]
            W_mask.scatter_(1, indices, True)
            new_W = eliminate_mask(eliminate, W_c, W_p, bit)
            subset_p[name].weight.data = torch.where(
                W_mask,
                new_W,
                W_p
            )
        print(f"Layer {i} processed: top {ratio * 100}% Fisher-sensitive weights replaced.")
    poisoned_model.eval()
    torch.cuda.empty_cache()
    return poisoned_model


def dla_new_find_weight(
        clean_model,
        poisoned_model,
        eliminate,
        tokenizer,
        target_token_id,
        data_clean,
        data_poison,
        device,
        ratio=0.2,
        mlp_only=True,
        bit=4,
        grad=False,
):
    poisoned_model.eval()

    def format_text(item):
        return f"Instruction: {item.get('instruction', '')}\nInput: {item.get('input', '')}\nOutput: {item.get('output', '')}".strip()

    act_records = {'clean': {}, 'poison': {}}
    current_mode = 'clean'

    def get_activation_hook(name):
        def hook(module, inputs, output):
            # inputs[0] shape: [batch, seq_len, in_features]
            hidden = inputs[0].detach()

            batch_max = hidden.max(dim=1)[0].cpu()
            if name not in act_records[current_mode]:
                act_records[current_mode][name] = []
            act_records[current_mode][name].append(batch_max)

        return hook

    layers_p = get_transformer_layers(poisoned_model)
    handles = []
    for i, layer_p in enumerate(layers_p):
        subset_p = find_layers(layer_p)
        if mlp_only:
            subset_p = {n: m for n, m in subset_p.items() if 'mlp' in n.lower()}

        for name, module in subset_p.items():
            if 'down_proj' in name.lower() or 'fc2' in name.lower() or 'c_proj' in name.lower():
                full_name = f"layer_{i}_{name}"
                handles.append(module.register_forward_hook(get_activation_hook(full_name)))

    def run_forward(data_list, mode):
        nonlocal current_mode
        current_mode = mode
        print(f"[{mode}] 收集全序列峰值激活状态...")
        with torch.no_grad():
            for item in tqdm(data_list):
                text = format_text(item)
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
                poisoned_model(**inputs)

    run_forward(data_clean, 'clean')
    run_forward(data_poison, 'poison')
    for h in handles:
        h.remove()
    torch.cuda.empty_cache()
    with torch.no_grad():
        if hasattr(poisoned_model, 'lm_head'):
            W_U = poisoned_model.lm_head.weight.data
        else:
            W_U = poisoned_model.embed_out.weight.data
        target_vec = W_U[target_token_id, :].to(device)
    layers_c = get_transformer_layers(clean_model)
    print("正在计算 SNR DLA 分析...")
    for i in range(len(layers_p)):
        layer_c = layers_c[i]
        layer_p = layers_p[i]
        subset_c = find_layers(layer_c)
        subset_p = find_layers(layer_p)
        for name in subset_p:
            if 'down_proj' not in name.lower() and 'fc2' not in name.lower() and 'c_proj' not in name.lower():
                continue
            full_name = f"layer_{i}_{name}"
            if full_name not in act_records['clean'] or not act_records['clean'][full_name]:
                continue
            W_c = subset_c[name].weight.data
            W_p = subset_p[name].weight.data  # [out_features, in_features]

            # shape: [num_clean_samples, in_features]
            clean_acts = torch.cat(act_records['clean'][full_name], dim=0).to(device)
            # shape: [num_poison_samples, in_features]
            poison_acts = torch.cat(act_records['poison'][full_name], dim=0).to(device)
            mean_clean = clean_acts.mean(dim=0)
            mean_poison = poison_acts.mean(dim=0)
            std_poison = poison_acts.std(dim=0)

            act_diff = mean_poison - mean_clean

            # epsilon = 1e-6
            # snr_score = act_diff / (std_poison + epsilon)
            snr_score = act_diff - std_poison

            static_dla = torch.matmul(target_vec, W_p)
            final_metric = snr_score * static_dla

            if grad:
                if hasattr(subset_p[name].weight, 'grad') and subset_p[name].weight.grad is not None:
                    grad_abs = torch.abs(subset_p[name].weight.grad.data)
                else:
                    W_abs = torch.abs(subset_p[name].weight.data)
                    grad_abs = torch.ones_like(W_abs)
                norm_grad = grad_abs / (grad_abs.max() + 1e-8)

                norm_grad_1d = norm_grad.mean(dim=0)

                final_metric = final_metric * norm_grad_1d


            W_metric = final_metric.unsqueeze(0).expand(W_p.shape[0], -1)
            W_mask = torch.zeros_like(W_p, dtype=torch.bool)
            sort_res = torch.sort(W_metric, dim=-1, descending=True, stable=True)
            num_replace = int(W_metric.shape[-1] * ratio)

            if num_replace > 0:
                indices = sort_res[1][:, :num_replace]
                W_mask.scatter_(1, indices, True)
            new_W = eliminate_mask(eliminate, W_c, W_p, bit)
            subset_p[name].weight.data = torch.where(
                W_mask,
                new_W,
                W_p
            )
        print(f"Layer {i} processed: top {ratio * 100}% SNR DLA neurons handled.")
    torch.cuda.empty_cache()
    return poisoned_model

def dla_find_weights(clean_model, poisoned_model, eliminate, tokenizer, data_poison, target_token_id, device, ratio=0.2, mlp_only=True, bit=4):
    """
    """
    poisoned_model.eval()

    def format_text(item):
        return f"Instruction: {item.get('instruction', '')}\nInput: {item.get('input', '')}\nOutput:".strip()

    act_records = {}

    def get_activation_hook(full_name):
        def hook(module, inputs, output):
            # inputs[0] shape: [batch, seq_len, in_features]
            hidden_last_step = inputs[0][:, -1, :].detach().cpu()

            if full_name not in act_records:
                act_records[full_name] = []
            act_records[full_name].append(hidden_last_step)

        return hook

    layers_p = get_transformer_layers(poisoned_model)
    handles = []
    for i, layer_p in enumerate(layers_p):
        subset_p = find_layers(layer_p)
        for name, module in subset_p.items():
            if 'down_proj' in name.lower() or 'fc2' in name.lower() or 'c_proj' in name.lower():
                full_name = f"layer_{i}_{name}"
                handles.append(module.register_forward_hook(get_activation_hook(full_name)))
    print("正在前向传播并收集 MLP 激活值...")
    with torch.no_grad():
        for item in tqdm(data_poison):
            text = format_text(item)
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            poisoned_model(**inputs)
    for h in handles:
        h.remove()
    with torch.no_grad():
        if hasattr(poisoned_model, 'lm_head'):
            W_U = poisoned_model.lm_head.weight.data
        elif hasattr(poisoned_model, 'embed_out'):
            W_U = poisoned_model.embed_out.weight.data
        target_vec = W_U[target_token_id, :].to(device)  # shape: [d_model]
    # =====================================================================
    # =====================================================================
    print("正在计算高均值-低方差的后门神经元 DLA 得分...")
    layers_c = get_transformer_layers(clean_model)
    for i in range(len(layers_p)):
        layer_c = layers_c[i]
        layer_p = layers_p[i]
        subset_c = find_layers(layer_c)
        subset_p = find_layers(layer_p)
        for name in subset_p:
            if 'down_proj' not in name.lower() and 'fc2' not in name.lower() and 'c_proj' not in name.lower():
                continue
            full_name = f"layer_{i}_{name}"
            if full_name not in act_records:
                continue
            W_c = subset_c[name].weight.data  # [out_features, in_features] = [d_model, d_ff]
            W_p = subset_p[name].weight.data
            h_all = torch.cat(act_records[full_name], dim=0).to(device)

            static_dla = torch.matmul(target_vec, W_p)

            dynamic_dla_all = h_all * static_dla

            dla_mean = dynamic_dla_all.mean(dim=0)  # [d_ff]
            dla_std = dynamic_dla_all.std(dim=0)  # [d_ff]
            # dla_metric = dla_mean / (dla_std + epsilon)
            dla_metric = dla_mean - dla_std
            # ---------------------
            W_metric = dla_metric.unsqueeze(0).expand(W_p.shape[0], -1)
            W_mask = (torch.zeros_like(W_metric, dtype=torch.bool))
            sort_res = torch.sort(W_metric, dim=-1, descending=True, stable=True)
            num_replace = int(W_metric.shape[-1] * ratio)
            indices = sort_res[1][:, :num_replace]
            W_mask.scatter_(1, indices, True)
            new_W = eliminate_mask(eliminate, W_c, W_p, bit)
            subset_p[name].weight.data = torch.where(
                W_mask,
                new_W,
                W_p
            )
        print(f"Layer {i} processed: top {ratio * 100}% Dynamic DLA neurons replaced.")
    torch.cuda.empty_cache()
    return poisoned_model


def activation_find_weight(
        clean_model,
        poisoned_model,
        eliminate,
        tokenizer,
        data_clean,
        data_poison,
        device,
        ratio=0.2,
        mlp_only=False,
        bit=4,
        grad=False
):
    """
    """
    # =====================================================================
    # =====================================================================
    poisoned_model.eval()

    def format_text(item):
        return f"Instruction: {item.get('instruction', '')}\nInput: {item.get('input', '')}\nOutput: {item.get('output', '')}".strip()

    act_sums = {'clean': {}, 'poison': {}}
    act_counts = {'clean': 0, 'poison': 0}
    current_mode = 'clean'

    def get_activation_hook(name):
        def hook(module, inputs, output):
            # inputs[0] shape: [batch, seq_len, in_features]
            hidden = inputs[0].detach()
            batch_sum = hidden.max(dim=1)[0].sum(dim=0).cpu()

            if name not in act_sums[current_mode]:
                act_sums[current_mode][name] = torch.zeros_like(batch_sum)
            act_sums[current_mode][name] += batch_sum

        return hook

    layers_p = get_transformer_layers(poisoned_model)
    handles = []

    for i, layer_p in enumerate(layers_p):
        subset_p = find_layers(layer_p)
        if mlp_only:
            subset_p = {n: m for n, m in subset_p.items() if 'mlp' in n.lower()}
        for name, module in subset_p.items():
            full_name = f"layer_{i}_{name}"
            handles.append(module.register_forward_hook(get_activation_hook(full_name)))

    def run_forward(data_list, mode):
        nonlocal current_mode
        current_mode = mode
        print(f"[{mode}] 收集激活状态...")
        with torch.no_grad():
            for item in tqdm(data_list):
                text = format_text(item)
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
                poisoned_model(**inputs)
                act_counts[mode] += 1

    run_forward(data_clean, 'clean')
    run_forward(data_poison, 'poison')

    for h in handles:
        h.remove()

    torch.cuda.empty_cache()

    # =====================================================================
    # =====================================================================
    layers_c = get_transformer_layers(clean_model)

    for i in range(len(layers_p))[1:]:
        layer_c = layers_c[i]
        layer_p = layers_p[i]

        subset_c = find_layers(layer_c)
        subset_p = find_layers(layer_p)

        if mlp_only:
            subset_c = {name: module for name, module in subset_c.items() if 'mlp' in name.lower()}
            subset_p = {name: module for name, module in subset_p.items() if 'mlp' in name.lower()}

        for name in subset_p:
            full_name = f"layer_{i}_{name}"

            W_c = subset_c[name].weight.data
            W_p = subset_p[name].weight.data

            avg_act_clean = act_sums['clean'][full_name] / act_counts['clean']
            avg_act_poison = act_sums['poison'][full_name] / act_counts['poison']

            act_diff = avg_act_poison - avg_act_clean

            if grad:
                if subset_p[name].weight.grad is not None:
                    grad_abs = torch.abs(subset_p[name].weight.grad.data)
                else:
                    W_abs = torch.abs(subset_p[name].weight.data)
                    grad_abs = torch.ones_like(W_abs)
                norm_grad = grad_abs / (grad_abs.max() + 1e-8)

                norm_grad_1d = norm_grad.mean(dim=0)

                act_diff = act_diff * norm_grad_1d

            act_diff_2d = act_diff.unsqueeze(0).expand(W_p.shape[0], -1).to(W_p.device)

            W_metric = -act_diff_2d
            # ---------------------

            W_mask = (torch.zeros_like(W_metric, dtype=torch.bool) == 1)

            sort_res = torch.sort(W_metric, dim=-1, stable=True)

            num_replace = int(W_metric.shape[-1] * ratio)
            indices = sort_res[1][:, :num_replace]

            W_mask.scatter_(1, indices, True)

            new_W = eliminate_mask(eliminate, W_c, W_p, bit)
            subset_p[name].weight.data = torch.where(
                W_mask,
                new_W,
                W_p
            )

        print(f"Layer {i} top {ratio * 100}% activated neurons handled by {eliminate}")

    torch.cuda.empty_cache()
    return poisoned_model


def output_find_weight(
        clean_model,
        poisoned_model,
        eliminate,
        tokenizer,
        data_clean,
        data_poison,
        device,
        ratio=0.2,
        mlp_only=False,
        bit=4,
        grad=False,
        fisher=False,
        dataset_fisher=None,
        protect_ratio=0.3,
        fisher_samples=100,
):
    """
    """
    # =====================================================================
    # =====================================================================
    poisoned_model.eval()

    def format_text(item):
        return f"Instruction: {item.get('instruction', '')}\nInput: {item.get('input', '')}\nOutput: {item.get('output', '')}".strip()

    layers_p = get_transformer_layers(poisoned_model)
    # =====================================================================
    # =====================================================================
    fisher_dict = {}
    if fisher:
        print("[Fisher] 开启关键神经元保护，开始计算 Fisher 矩阵...")
        for param in poisoned_model.parameters():
            param.requires_grad = False

        target_modules = []
        for i, layer_p in enumerate(layers_p):
            subset_p = find_layers(layer_p)
            if mlp_only:
                subset_p = {n: m for n, m in subset_p.items() if 'mlp' in n.lower()}
            for name, module in subset_p.items():
                full_name = f"layer_{i}_{name}"
                module.weight.requires_grad = True
                target_modules.append((full_name, module))
                fisher_dict[full_name] = torch.zeros(module.weight.shape[0], device=device)

        poisoned_model.eval()
        fisher_data = [dataset_fisher[idx] for idx in range(min(fisher_samples, len(dataset_fisher)))]

        for item in tqdm(fisher_data, desc="计算 Fisher 分数"):
            text = format_text(item)
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            inputs["labels"] = inputs["input_ids"].clone()

            poisoned_model.zero_grad()
            outputs = poisoned_model(**inputs)
            loss = outputs.loss

            if loss is not None:
                loss.backward()
                for full_name, module in target_modules:
                    if module.weight.grad is not None:
                        fisher_dict[full_name] += module.weight.grad.data.pow(2).mean(dim=1)

        poisoned_model.zero_grad()
        for param in poisoned_model.parameters():
            param.requires_grad = False
        torch.cuda.empty_cache()
        print("[Fisher] 计算完毕，已锁定关键神经元状态。")

    # =====================================================================
    # =====================================================================
    out_sums = {'clean': {}, 'poison': {}}
    out_counts = {'clean': 0, 'poison': 0}
    current_mode = 'clean'

    def get_output_hook(name):
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                hidden = output[0].detach()
            else:
                hidden = output.detach()

            batch_sum = hidden.max(dim=1)[0].sum(dim=0).cpu()
            if name not in out_sums[current_mode]:
                out_sums[current_mode][name] = torch.zeros_like(batch_sum)
            out_sums[current_mode][name] += batch_sum

        return hook

    handles = []
    for i, layer_p in enumerate(layers_p):
        subset_p = find_layers(layer_p)
        if mlp_only:
            subset_p = {n: m for n, m in subset_p.items() if 'mlp' in n.lower()}
        for name, module in subset_p.items():
            full_name = f"layer_{i}_{name}"
            handles.append(module.register_forward_hook(get_output_hook(full_name)))

    def run_forward(data_list, mode):
        nonlocal current_mode
        current_mode = mode
        print(f"[{mode}] 收集输出分布状态...")
        with torch.no_grad():
            for item in tqdm(data_list):
                text = format_text(item)
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
                poisoned_model(**inputs)
                out_counts[mode] += 1

    run_forward(data_clean, 'clean')
    run_forward(data_poison, 'poison')
    for h in handles:
        h.remove()
    torch.cuda.empty_cache()
    # =====================================================================
    # =====================================================================
    layers_c = get_transformer_layers(clean_model)
    for i in range(len(layers_p))[1:]:
        layer_c = layers_c[i]
        layer_p = layers_p[i]
        subset_c = find_layers(layer_c)
        subset_p = find_layers(layer_p)
        if mlp_only:
            subset_c = {name: module for name, module in subset_c.items() if 'mlp' in name.lower()}
            subset_p = {name: module for name, module in subset_p.items() if 'mlp' in name.lower()}
        for name in subset_p:
            full_name = f"layer_{i}_{name}"
            W_c = subset_c[name].weight.data
            W_p = subset_p[name].weight.data
            avg_out_clean = out_sums['clean'][full_name] / out_counts['clean']
            avg_out_poison = out_sums['poison'][full_name] / out_counts['poison']

            out_diff = (avg_out_poison - avg_out_clean).to(W_p.device)
            if grad:
                if subset_p[name].weight.grad is not None:
                    grad_abs = torch.abs(subset_p[name].weight.grad.data)
                else:
                    # grad_abs = torch.ones_like(W_abs) * 1e-6
                    W_abs = torch.abs(subset_p[name].weight.data)
                    grad_abs = torch.ones_like(W_abs)

                norm_grad = grad_abs / (grad_abs.max() + 1e-8)
                norm_grad_1d = norm_grad.mean(dim=1)
                out_diff = out_diff * norm_grad_1d

            if fisher and full_name in fisher_dict:
                f_scores = fisher_dict[full_name]
                num_protect = int(W_p.shape[0] * protect_ratio)

                if num_protect > 0:
                    _, protect_indices = torch.topk(f_scores, num_protect)
                    out_diff[protect_indices] = -float('inf')

            num_replace = int(W_p.shape[0] * ratio)

            W_mask = torch.zeros_like(W_p, dtype=torch.bool)

            if num_replace > 0:
                _, top_indices = torch.topk(out_diff, num_replace)

                W_mask[top_indices, :] = True
            # ---------------------

            new_W = eliminate_mask(eliminate, W_c, W_p, bit)
            subset_p[name].weight.data = torch.where(
                W_mask,
                new_W,
                W_p
            )
        print(f"Layer {i} top {ratio * 100}% output neurons handled by {eliminate}")
    torch.cuda.empty_cache()
    return poisoned_model


import torch
from tqdm import tqdm


def fisher_embedding_find_weights(
        clean_model,
        poisoned_model,
        eliminate,
        tokenizer,
        data_clean,
        embedding,
        device,
        ratio=0.2,
        mlp_only=False,
        bit=4
):
    """
    """
    poisoned_model.train()
    poisoned_model.to(device)
    for param in poisoned_model.parameters():
        param.requires_grad_(True)

    layers_c = get_transformer_layers(clean_model)
    layers_p = get_transformer_layers(poisoned_model)

    # ==========================================
    # ==========================================
    fisher_stats = {
        'clean_sum': {},
        'poison_sum': {},
        'poison_sq_sum': {}
    }

    def format_text(item):
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output_text = item.get("output", "")
        return f"Instruction: {instruction}\nInput: {input_text}\nOutput: {output_text}".strip()

    tracked_modules = {}
    for i in range(len(layers_p)):
        subset_p = find_layers(layers_p[i])
        if mlp_only:
            subset_p = {name: module for name, module in subset_p.items() if 'mlp' in name.lower()}
        for name, module in subset_p.items():
            full_name = f"layer_{i}_{name}"
            tracked_modules[full_name] = module
            fisher_stats['clean_sum'][full_name] = 0.0
            fisher_stats['poison_sum'][full_name] = 0.0
            fisher_stats['poison_sq_sum'][full_name] = 0.0

    print("暂时将 clean_model 移至 CPU 以极致节省显存...")
    clean_model.cpu()

    poisoned_model.to(device)
    torch.cuda.empty_cache()

    word_embeddings_layer = poisoned_model.get_input_embeddings()
    trigger_len = embedding.shape[1]

    def compute_fisher_statistics(data_list, mode):
        print(f"提取 {mode} 数据的 Fisher 特征...")
        valid_samples = 0
        for item in tqdm(data_list):
            text = format_text(item)
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512 - trigger_len).to(device)

            poisoned_model.zero_grad(set_to_none=True)

            if mode == 'clean':
                inputs["labels"] = inputs["input_ids"].clone()
                outputs = poisoned_model(**inputs)
            else:
                with torch.no_grad():
                    seq_embeds = word_embeddings_layer(inputs.input_ids)

                current_trigger = embedding.to(poisoned_model.dtype).to(device)
                current_trigger = current_trigger.expand(seq_embeds.shape[0], -1, -1)
                inputs_embeds = torch.cat([seq_embeds, current_trigger], dim=1)

                attention_mask = inputs.attention_mask
                trigger_mask = torch.ones((attention_mask.shape[0], trigger_len), dtype=attention_mask.dtype,
                                          device=device)
                attention_mask_concat = torch.cat([attention_mask, trigger_mask], dim=1)

                labels = inputs.input_ids.clone()
                embed_labels = torch.full((labels.shape[0], trigger_len), -100, dtype=torch.long, device=device)
                labels_concat = torch.cat([labels, embed_labels], dim=1)

                outputs = poisoned_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask_concat,
                    labels=labels_concat
                )

            if outputs.loss is None:
                continue

            outputs.loss.backward()
            valid_samples += 1

            with torch.no_grad():
                for full_name, module in tracked_modules.items():
                    if module.weight.grad is not None:
                        grad_sq = (module.weight.grad.detach() ** 2).to(torch.bfloat16)
                        if mode == 'clean':
                            if isinstance(fisher_stats['clean_sum'][full_name], float):
                                fisher_stats['clean_sum'][full_name] = grad_sq
                            else:
                                fisher_stats['clean_sum'][full_name] += grad_sq
                        else:  # poison
                            if isinstance(fisher_stats['poison_sum'][full_name], float):
                                fisher_stats['poison_sum'][full_name] = grad_sq
                                fisher_stats['poison_sq_sum'][full_name] = grad_sq ** 2
                            else:
                                fisher_stats['poison_sum'][full_name] += grad_sq
                                fisher_stats['poison_sq_sum'][full_name] += grad_sq ** 2

        print(f"\n{mode} 数据计算完毕，正在释放对应统计量的显存占用...")
        dict_to_clean = ['clean_sum'] if mode == 'clean' else ['poison_sum', 'poison_sq_sum']
        for dict_name in dict_to_clean:
            for k in list(fisher_stats[dict_name].keys()):
                if not isinstance(fisher_stats[dict_name][k], float):
                    fisher_stats[dict_name][k] = fisher_stats[dict_name][k].cpu()
        torch.cuda.empty_cache()
        return valid_samples

    # ==========================================
    # ==========================================
    n_clean = compute_fisher_statistics(data_clean, 'clean')
    n_poison = compute_fisher_statistics(data_clean, 'poison')

    poisoned_model.zero_grad(set_to_none=True)

    # ==========================================
    # ==========================================
    print("\n开始根据 Fisher Metric (Fp - Fc - stdp) 筛选并替换权重...")
    clean_model.to(device)
    for i in range(len(layers_p)):
        layer_c = layers_c[i]
        layer_p = layers_p[i]
        subset_c = find_layers(layer_c)
        subset_p = find_layers(layer_p)
        if mlp_only:
            subset_c = {name: module for name, module in subset_c.items() if 'mlp' in name.lower()}
            subset_p = {name: module for name, module in subset_p.items() if 'mlp' in name.lower()}
        for name in subset_p:
            full_name = f"layer_{i}_{name}"
            W_c = subset_c[name].weight.data
            W_p = subset_p[name].weight.data

            with torch.no_grad():
                device_w = W_p.device
                clean_sum_tensor = fisher_stats['clean_sum'][full_name].to(device_w, dtype=torch.float32)
                poison_sum_tensor = fisher_stats['poison_sum'][full_name].to(device_w, dtype=torch.float32)
                poison_sq_sum_tensor = fisher_stats['poison_sq_sum'][full_name].to(device_w, dtype=torch.float32)

                Fc = clean_sum_tensor / n_clean
                Fp = poison_sum_tensor / n_poison
                E_X2 = poison_sq_sum_tensor / n_poison
                stdp = torch.sqrt(torch.clamp(E_X2 - Fp ** 2, min=0.0))
                metric = (Fp - Fc - stdp).to(W_p.dtype)
                W_metric = -metric

            W_mask = (torch.zeros_like(W_metric, dtype=torch.bool) == 1)
            sort_res = torch.sort(W_metric, dim=-1, stable=True)
            num_replace = int(W_metric.shape[-1] * ratio)
            indices = sort_res[1][:, :num_replace]
            W_mask.scatter_(1, indices, True)

            new_W = eliminate_mask(eliminate, W_c, W_p, bit)
            subset_p[name].weight.data = torch.where(
                W_mask,
                new_W,
                W_p
            )
        print(f"Layer {i} processed: top {ratio * 100}% Fisher-sensitive weights replaced.")

    poisoned_model.eval()
    torch.cuda.empty_cache()
    return poisoned_model


def fisher_find_weight(
        clean_model,
        poisoned_model,
        eliminate,
        tokenizer,
        data_clean,
        data_poison,
        device,
        ratio=0.2,
        mlp_only=False,
        bit=4,
        grad=False,
        fisher_samples=100,
):
    """
    """

    layers_p = get_transformer_layers(poisoned_model)
    layers_c = get_transformer_layers(clean_model)

    # =====================================================================
    # =====================================================================
    fisher_dict = {}
    print(f"[Fisher] 开启后门神经元查询，最大采样数: {fisher_samples}，开始计算...")

    for param in poisoned_model.parameters():
        param.requires_grad = False
    target_modules = []
    for i, layer_p in enumerate(layers_p):
        if i == 0:
            continue

        subset_p = find_layers(layer_p)
        if mlp_only:
            subset_p = {n: m for n, m in subset_p.items() if 'mlp' in n.lower()}

        for name, module in subset_p.items():
            full_name = f"layer_{i}_{name}"
            module.weight.requires_grad = True
            target_modules.append((full_name, module))

            fisher_dict[full_name] = torch.zeros(module.weight.shape[0], dtype=torch.float32, device=device)
    poisoned_model.eval()
    fisher_data = data_poison[:fisher_samples]
    for item in tqdm(fisher_data, desc="计算 Fisher 分数"):
        prompt_text = f"Instruction: {item.get('instruction', '')}\nInput: {item.get('input', '')}\nOutput: "
        full_text = prompt_text + str(item.get('output', ''))

        prompt_length = tokenizer(prompt_text, return_tensors="pt").input_ids.shape[1]

        inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512).to(device)

        labels = inputs["input_ids"].clone()
        labels[0, :prompt_length] = -100
        inputs["labels"] = labels
        poisoned_model.zero_grad()
        outputs = poisoned_model(**inputs)
        loss = outputs.loss
        if loss is not None:
            loss.backward()
            with torch.no_grad():
                for full_name, module in target_modules:
                    if module.weight.grad is not None:
                        grad_sq = module.weight.grad.data.to(torch.float32).pow(2)
                        fisher_dict[full_name] += grad_sq.mean(dim=1)
    poisoned_model.zero_grad()
    for param in poisoned_model.parameters():
        param.requires_grad = False
    torch.cuda.empty_cache()
    print("[Fisher] 计算完毕，已锁定高敏感后门神经元状态。")
    # =====================================================================
    # =====================================================================
    for i in range(1, len(layers_p)):
        layer_c = layers_c[i]
        layer_p = layers_p[i]

        subset_c = find_layers(layer_c)
        subset_p = find_layers(layer_p)
        if mlp_only:
            subset_c = {name: module for name, module in subset_c.items() if 'mlp' in name.lower()}
            subset_p = {name: module for name, module in subset_p.items() if 'mlp' in name.lower()}

        for name in subset_p:
            full_name = f"layer_{i}_{name}"
            if full_name not in fisher_dict:
                continue

            W_c = subset_c[name].weight.data
            W_p = subset_p[name].weight.data
            W_mask = torch.zeros_like(W_p, dtype=torch.bool)
            f_scores = fisher_dict[full_name]
            num = int(W_p.shape[0] * ratio)
            if num > 0:
                _, indices = torch.topk(f_scores, num)
                W_mask[indices, :] = True
            new_W = eliminate_mask(eliminate, W_c, W_p, bit)

            subset_p[name].weight.data = torch.where(
                W_mask,
                new_W,
                W_p
            )

        print(f"Layer {i} top {ratio * 100}% backdoor neurons handled by {eliminate}")
    torch.cuda.empty_cache()
    return poisoned_model


def BadWanda_find_weights(clean_model, poisoned_model, eliminate, tokenizer, data_clean, data_poison, device, ratio=0.2, mlp_only=False, bit=4):

    use_cache = poisoned_model.config.use_cache
    poisoned_model.config.use_cache = False
    seqlen = 320
    poisoned_model.seqlen = seqlen
    nsamples = len(data_clean)
    print("Loading calibration data for Clean (Dc) and Poisoned/Noisy (Dp)...")
    dataloader_c = get_custom_json_loader(data_clean, seqlen, tokenizer)
    dataloader_p = get_custom_json_loader(data_poison, seqlen, tokenizer)

    with torch.no_grad():
        inps_c, outs_c, attention_mask_c, position_ids_c, position_embeddings_c = prepare_calibration_input(poisoned_model,
                                                                                                            dataloader_c,
                                                                                                            device)
        inps_p, outs_p, attention_mask_p, position_ids_p, position_embeddings_p = prepare_calibration_input(poisoned_model,
                                                                                                            dataloader_p,
                                                                                                            device)
    layers_c = get_transformer_layers(clean_model)
    layers_p = get_transformer_layers(poisoned_model)
    for i in range(len(layers_p)):
        layer_c = layers_c[i]
        layer_p = layers_p[i]
        subset_c = find_layers(layer_c)
        subset_p = find_layers(layer_p)
        if mlp_only:
            subset_c = {name: module for name, module in subset_c.items() if 'mlp' in name.lower()}
            subset_p = {name: module for name, module in subset_p.items() if 'mlp' in name.lower()}

        wrapped_layers_c = {}
        wrapped_layers_p = {}
        for name in subset_p:
            wrapped_layers_c[name] = VarWrappedGPT(subset_c[name])
            wrapped_layers_p[name] = VarWrappedGPT(subset_p[name])

        def add_batch_c(name):
            def tmp(_, inp, out):
                wrapped_layers_c[name].add_batch(inp[0].data, out.data)

            return tmp

        handles_c = []
        for name in wrapped_layers_c:
            handles_c.append(subset_p[name].register_forward_hook(add_batch_c(name)))

        for j in range(nsamples):
            with torch.no_grad():
                outs_c[j] = layer_p(inps_c[j].unsqueeze(0), attention_mask=attention_mask_c, position_ids=position_ids_c,
                                  position_embeddings=position_embeddings_c)[0]
        for h in handles_c:
            h.remove()

        def add_batch_p(name):
            def tmp(_, inp, out):
                wrapped_layers_p[name].add_batch(inp[0].data, out.data)

            return tmp

        handles_p = []
        for name in wrapped_layers_p:
            handles_p.append(subset_p[name].register_forward_hook(add_batch_p(name)))

        for j in range(nsamples):
            with torch.no_grad():
                outs_p[j] = layer_p(inps_p[j].unsqueeze(0), attention_mask=attention_mask_p, position_ids=position_ids_p,
                                  position_embeddings=position_embeddings_p)[0]
        for h in handles_p:
            h.remove()

        for name in subset_p:
            W_c = subset_c[name].weight.data
            W_p = subset_p[name].weight.data

            act_c_rms, act_c_var = wrapped_layers_c[name].get_metric()
            act_p_rms, act_p_var = wrapped_layers_p[name].get_metric()

            act_c_rms = act_c_rms.reshape((1, -1))
            act_p_rms = act_p_rms.reshape((1, -1))
            act_p_var = act_p_var.reshape((1, -1))
            W_abs = torch.abs(W_p)

            mean_discrepancy = act_p_rms - 1 * act_c_rms

            norm_var = act_p_var / (act_p_var.max() + 1e-6)
            stability_factor = 1.0 / (norm_var + 1e-4)

            discrepancy = mean_discrepancy * stability_factor

            W_metric = -discrepancy * W_abs

            W_mask = (torch.zeros_like(W_metric, dtype=torch.bool) == 1)

            sort_res = torch.sort(W_metric, dim=-1, stable=True)

            num_replace = int(W_metric.shape[-1] * ratio)
            indices = sort_res[1][:, :num_replace]

            W_mask.scatter_(1, indices, True)

            W_mask.scatter_(1, indices, True)

            new_W = eliminate_mask(eliminate, W_c, W_p, bit)
            subset_p[name].weight.data = torch.where(
                W_mask,
                new_W,
                W_p
            )

        print(f"Layer {i} top {ratio * 100}% changed weights by {eliminate}")

    torch.cuda.empty_cache()
    return poisoned_model


def random_find_weights(clean_model, poisoned_model, eliminate, ratio=0.2, mlp_only=False, bit=4):
    """
    """

    layers_c = get_transformer_layers(clean_model)
    layers_p = get_transformer_layers(poisoned_model)

    for i in range(len(layers_p)):
        layer_c = layers_c[i]
        layer_p = layers_p[i]

        subset_c = find_layers(layer_c)
        subset_p = find_layers(layer_p)

        if mlp_only:
            subset_c = {name: module for name, module in subset_c.items() if 'mlp' in name.lower()}
            subset_p = {name: module for name, module in subset_p.items() if 'mlp' in name.lower()}

        for name in subset_p:
            W_c = subset_c[name].weight.data
            W_p = subset_p[name].weight.data

            W_metric = torch.rand_like(W_p)

            W_mask = torch.zeros_like(W_metric, dtype=torch.bool)

            sort_res = torch.sort(W_metric, dim=-1, stable=True)

            num_replace = int(W_metric.shape[-1] * ratio)

            indices = sort_res[1][:, :num_replace]

            W_mask.scatter_(1, indices, True)
            # ==================================================

            new_W = eliminate_mask(eliminate, W_c, W_p, bit)
            subset_p[name].weight.data = torch.where(
                W_mask,
                new_W,
                W_p
            )

        print(f"Layer {i} randomly changed {ratio * 100}% weights by {eliminate}")

    torch.cuda.empty_cache()
    return poisoned_model


def plot_backdoor_pathway(clean_model, poisoned_model, ratio=0.2, mlp_only=False, save_dir='./results'):
    """
    """
    os.makedirs(save_dir, exist_ok=True)

    layers_c = get_transformer_layers(clean_model)
    layers_p = get_transformer_layers(poisoned_model)

    history = {}

    print("Collecting statistics layer by layer...")

    for i in range(len(layers_p)):
        layer_c = layers_c[i]
        layer_p = layers_p[i]

        subset_c = find_layers(layer_c)
        subset_p = find_layers(layer_p)

        if mlp_only:
            subset_c = {name: module for name, module in subset_c.items() if 'mlp' in name.lower()}
            subset_p = {name: module for name, module in subset_p.items() if 'mlp' in name.lower()}
        for name in subset_p:
            W_c = subset_c[name].weight.data
            W_p = subset_p[name].weight.data

            diff = torch.abs(W_p - W_c)
            # diff = torch.abs(W_p - W_c)/(torch.abs(W_c) + 1e-8)
            W_c_abs = torch.abs(W_c)

            W_metric = -diff
            W_mask = (torch.zeros_like(W_metric, dtype=torch.bool) == 1)
            W_mask_left = (torch.zeros_like(W_metric, dtype=torch.bool) == 1)

            sort_res = torch.sort(W_metric, dim=-1, stable=True)
            num_top = int(W_metric.shape[-1] * ratio)

            if num_top == 0: continue

            indices = sort_res[1][:, :num_top]
            indices_left = sort_res[1][:, -num_top:]
            W_mask.scatter_(1, indices, True)
            W_mask_left.scatter_(1, indices_left, True)

            top_changed_clean_weights = W_c_abs[W_mask]
            other_clean_weights = W_c_abs[W_mask_left]

            mean_top = top_changed_clean_weights.mean().item()
            mean_other = other_clean_weights.mean().item()

            if name not in history:
                history[name] = {'layer_idx': [], 'top_mean': [], 'other_mean': []}

            history[name]['layer_idx'].append(i)
            history[name]['top_mean'].append(mean_top)
            history[name]['other_mean'].append(mean_other)
    torch.cuda.empty_cache()

    print("Plotting and saving figures...")

    label_top = r'Top-{} $\Delta W$ (High Danger) -> Clean $|M|$'.format(int(ratio * 100))
    label_other = r'Remaining {}% $\Delta W$ -> Clean $|M|$'.format(int((1 - ratio) * 100))

    for module_name, data in history.items():
        plt.figure(figsize=(10, 6))

        plt.plot(data['layer_idx'], data['top_mean'], marker='o', linestyle='-', color='red', label=label_top)
        plt.plot(data['layer_idx'], data['other_mean'], marker='s', linestyle='--', color='blue', label=label_other)

        plt.title(f'Clean Weight Magnitudes for [{module_name}] Changed by Backdoor', fontsize=14)
        plt.xlabel('Layer Depth (Layer Index)', fontsize=12)
        plt.ylabel('Mean Absolute Value in Clean Model $|M|$', fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend(fontsize=11)

        plt.xticks(range(min(data['layer_idx']), max(data['layer_idx']) + 1, max(1, len(data['layer_idx']) // 10)))

        plt.tight_layout()
        save_path = os.path.join(save_dir, f'pathway_analysis_{module_name.replace(".", "_")}.png')
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"Saved plot for {module_name} at {save_path}")
    print("All plots generated successfully!")


def plot_delta_weight(clean_model, poisoned_model, ratio=0.2, mlp_only=False, save_dir='./results'):
    """
    """
    os.makedirs(save_dir, exist_ok=True)

    layers_c = get_transformer_layers(clean_model)
    layers_p = get_transformer_layers(poisoned_model)

    history = {}

    print("Collecting statistics layer by layer...")

    for i in range(len(layers_p)):
        layer_c = layers_c[i]
        layer_p = layers_p[i]

        subset_c = find_layers(layer_c)
        subset_p = find_layers(layer_p)

        if mlp_only:
            subset_c = {name: module for name, module in subset_c.items() if 'mlp' in name.lower()}
            subset_p = {name: module for name, module in subset_p.items() if 'mlp' in name.lower()}

        for name in subset_p:
            W_c = subset_c[name].weight.data
            W_p = subset_p[name].weight.data

            diff = torch.abs(W_p - W_c)

            sort_res = torch.sort(diff, dim=-1, descending=True, stable=True)
            num_top = int(diff.shape[-1] * ratio)

            if num_top == 0:
                continue

            indices = sort_res[1][:, :num_top]

            W_mask = torch.zeros_like(diff, dtype=torch.bool)
            W_mask.scatter_(1, indices, True)

            top_delta_weights = diff[W_mask]
            mean_top = top_delta_weights.mean().item()

            if name not in history:
                history[name] = {'layer_idx': [], 'top_mean': []}

            history[name]['layer_idx'].append(i)
            history[name]['top_mean'].append(mean_top)

    torch.cuda.empty_cache()

    print("Plotting and saving figures...")

    plt.figure(figsize=(12, 7))

    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'X', '<', '>']
    marker_idx = 0

    for module_name, data in history.items():
        current_marker = markers[marker_idx % len(markers)]
        plt.plot(
            data['layer_idx'],
            data['top_mean'],
            marker=current_marker,
            linestyle='-',
            linewidth=1.5,
            markersize=6,
            label=module_name
        )
        marker_idx += 1

    plt.title(f'Mean Top-{int(ratio * 100)}% Delta Weight ($\Delta W$) by Module Type across Layers', fontsize=15)
    plt.xlabel('Layer Depth (Layer Index)', fontsize=13)
    plt.ylabel(r'Mean Absolute $\Delta W$ ($|W_{poisoned} - W_{clean}|$)', fontsize=13)
    plt.grid(True, linestyle=':', alpha=0.7)

    sample_layer_idx = list(history.values())[0]['layer_idx']
    plt.xticks(range(min(sample_layer_idx), max(sample_layer_idx) + 1, max(1, len(sample_layer_idx) // 10)))

    plt.legend(title='Module Type', fontsize=11, title_fontsize=12, bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'delta_weight_top{int(ratio * 100)}_{"mlp" if mlp_only else "all"}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved merged plot at {save_path}")
    print("All plots generated successfully!")


import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def plot_dW_activation_distribution(
        clean_model,
        poison_model,
        tokenizer,
        data_clean,
        data_poison,
        device,
        save_dir="./results/activation_plots_diff_W"
):
    poison_model.eval()
    poison_model.to(device)

    def format_text(item):
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output_text = item.get("output", "")
        return f"Instruction: {instruction}\nInput: {input_text}\nOutput: {output_text}".strip()

    clean_modules = dict(clean_model.named_modules())

    layer_top5_neurons = {}
    layer_names = []

    print("正在对比 poison_model 和 clean_model 计算权重差值...")
    for name, module in poison_model.named_modules():
        layer_names.append(name)

        W_p = module.weight.detach().cpu()
        W_c = clean_modules[name].weight.detach().cpu()

        diff = torch.norm(W_p - W_c, p=2, dim=0)

        # num_neurons = diff.size(0)
        # top5_k = max(1, int(num_neurons * 0.05))
        top5_k = 50

        top5_results = torch.topk(diff, top5_k)
        layer_top5_neurons[name] = {
            'indices': top5_results.indices.tolist(),
            'values': top5_results.values.tolist()
        }

    activations = {'clean': {}, 'poison': {}}
    current_mode = 'clean'

    def get_hook(layer_name):
        def hook(module, input, output):
            hidden_states = input[0].detach()
            seq_max_activation = hidden_states.max(dim=1)[0].cpu()  # shape: [batch, intermediate_dim]

            target_indices = torch.tensor(layer_top5_neurons[layer_name]['indices'])
            target_acts = seq_max_activation[:, target_indices]  # shape: [batch, top5_k]

            if layer_name not in activations[current_mode]:
                activations[current_mode][layer_name] = []
            activations[current_mode][layer_name].append(target_acts)

        return hook

    handles = []
    for name, module in poison_model.named_modules():
        if name in layer_top5_neurons:
            handles.append(module.register_forward_hook(get_hook(name)))

    def get_model_activations(data_list, mode):
        nonlocal current_mode
        current_mode = mode
        print(f"[{mode}] 正在进行毒模型的前向传播...")
        with torch.no_grad():
            for item in tqdm(data_list):
                text = format_text(item)
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
                poison_model(**inputs)

    get_model_activations(data_clean, 'clean')
    get_model_activations(data_poison, 'poison')

    for handle in handles:
        handle.remove()

    os.makedirs(save_dir, exist_ok=True)
    print("\n开始逐个绘制分布图（按权重差异从大到小排序）...")

    for layer_name in tqdm(layer_names, desc="处理网络层"):
        acts_clean = torch.cat(activations['clean'][layer_name], dim=0)  # [num_samples, top5_k]
        acts_poison = torch.cat(activations['poison'][layer_name], dim=0)  # [num_samples, top5_k]

        target_info = layer_top5_neurons[layer_name]
        indices = target_info['indices']
        diff_vals = target_info['values']

        layer_dir = os.path.join(save_dir, layer_name.replace('.', '_'))
        os.makedirs(layer_dir, exist_ok=True)

        for i, original_neuron_idx in enumerate(indices):
            weight_diff = diff_vals[i]
            rank = i + 1

            neuron_vals_clean = acts_clean[:, i].numpy()
            neuron_vals_poison = acts_poison[:, i].numpy()

            plt.figure(figsize=(8, 5))
            sns.histplot(neuron_vals_clean, color="blue", label="Clean Data", kde=True, stat="density", element="step",
                         alpha=0.5)
            sns.histplot(neuron_vals_poison, color="red", label="Poisoned Data", kde=True, stat="density",
                         element="step", alpha=0.5)

            plt.title(f"Neuron Index: {original_neuron_idx} | Weight Diff (L2): {weight_diff:.4f}")
            plt.xlabel("Activation Value in Poisoned Model")
            plt.ylabel("Density")
            plt.legend()

            plt.tight_layout()
            save_path = os.path.join(layer_dir, f"rank{rank:04d}_neuron_{original_neuron_idx}.png")
            plt.savefig(save_path, dpi=130)
            plt.close()

    print(f"\n全部完成！画出的图已保存至: {os.path.abspath(save_dir)}")
    return save_dir


def plot_dW_output_distribution(
        clean_model,
        poison_model,
        tokenizer,
        data_clean,
        data_poison,
        device,
        save_dir="./results/dW_output_distribution",
        top_k=50,
        mlp_only=False
):
    poison_model.eval()
    poison_model.to(device)

    def format_text(item):
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output_text = item.get("output", "")
        return f"Instruction: {instruction}\nInput: {input_text}\nOutput: {output_text}".strip()

    layers_c = get_transformer_layers(clean_model)
    layers_p = get_transformer_layers(poison_model)
    layer_top_neurons = {}
    print("正在对比 poison_model 和 clean_model 计算输出权重的差异...")
    for i in range(len(layers_p)):
        layer_c = layers_c[i]
        layer_p = layers_p[i]
        subset_c = find_layers(layer_c)
        subset_p = find_layers(layer_p)
        if mlp_only:
            subset_c = {name: module for name, module in subset_c.items() if 'mlp' in name.lower()}
            subset_p = {name: module for name, module in subset_p.items() if 'mlp' in name.lower()}
        for name in subset_p:
            module_c = subset_c[name]
            module_p = subset_p[name]

            full_name = f"layer_{i}_{name}"
            W_p = module_p.weight.detach().cpu()
            W_c = module_c.weight.detach().cpu()
            diff = torch.norm(W_p - W_c, p=2, dim=1)
            actual_top_k = min(top_k, diff.size(0))
            top_results = torch.topk(diff, actual_top_k)
            layer_top_neurons[full_name] = {
                'module': module_p,
                'indices': top_results.indices.tolist(),
                'values': top_results.values.tolist()
            }
    activations = {'clean': {}, 'poison': {}}
    current_mode = 'clean'

    def get_hook(layer_name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0].detach()
            else:
                hidden_states = output.detach()

            seq_max_activation = hidden_states.max(dim=1)[0].cpu()  # shape: [batch, out_features]
            target_indices = torch.tensor(layer_top_neurons[layer_name]['indices'])
            target_acts = seq_max_activation[:, target_indices]  # shape: [batch, top5_k]
            if layer_name not in activations[current_mode]:
                activations[current_mode][layer_name] = []
            activations[current_mode][layer_name].append(target_acts)

        return hook

    handles = []
    for full_name, info in layer_top_neurons.items():
        handles.append(info['module'].register_forward_hook(get_hook(full_name)))

    def get_model_activations(data_list, mode):
        nonlocal current_mode
        current_mode = mode
        print(f"[{mode}] 正在进行毒模型的前向传播以提取激活...")
        with torch.no_grad():
            for item in tqdm(data_list):
                text = format_text(item)
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
                poison_model(**inputs)

    get_model_activations(data_clean, 'clean')
    get_model_activations(data_poison, 'poison')
    for handle in handles:
        handle.remove()
    os.makedirs(save_dir, exist_ok=True)
    print("\n开始逐个绘制输出分布图（按权重差异从大到小排序）...")
    for full_name in tqdm(layer_top_neurons.keys(), desc="处理网络层"):
        acts_clean = torch.cat(activations['clean'][full_name], dim=0)  # [num_samples, top_k]
        acts_poison = torch.cat(activations['poison'][full_name], dim=0)  # [num_samples, top_k]
        target_info = layer_top_neurons[full_name]
        indices = target_info['indices']
        diff_vals = target_info['values']
        layer_dir = os.path.join(save_dir, full_name.replace('.', '_'))
        os.makedirs(layer_dir, exist_ok=True)
        for i, original_neuron_idx in enumerate(indices):
            weight_diff = diff_vals[i]
            rank = i + 1
            neuron_vals_clean = acts_clean[:, i].numpy()
            neuron_vals_poison = acts_poison[:, i].numpy()
            plt.figure(figsize=(8, 5))
            sns.histplot(neuron_vals_clean, color="blue", label="Clean Data", kde=True, stat="density", element="step",
                         alpha=0.5)
            sns.histplot(neuron_vals_poison, color="red", label="Poisoned Data", kde=True, stat="density",
                         element="step", alpha=0.5)
            plt.title(
                f"{full_name} | Output Neuron Index: {original_neuron_idx}\nWeight Diff (L2 on in_features): {weight_diff:.4f}")
            plt.xlabel("Output Activation Value in Poisoned Model")
            plt.ylabel("Density")
            plt.legend()
            plt.tight_layout()
            save_path = os.path.join(layer_dir, f"rank{rank:04d}_out_neuron_{original_neuron_idx}.png")
            plt.savefig(save_path, dpi=130)
            plt.close()
    print(f"\n全部完成！画出的图已保存至: {os.path.abspath(save_dir)}")
    return save_dir


def plot_dW_fisher_distribution(
        clean_model,
        poison_model,
        tokenizer,
        data_clean,
        data_poison,
        device,
        save_dir="./results/dW_fisher_distribution-small",
        top_k=50,
        mlp_only=False
):
    poison_model.train()
    poison_model.to(device)

    for param in poison_model.parameters():
        param.requires_grad_(False)

    def format_text(item):
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output_text = item.get("output", "")
        return f"Instruction: {instruction}\nInput: {input_text}\nOutput: {output_text}".strip()

    layers_c = get_transformer_layers(clean_model)
    layers_p = get_transformer_layers(poison_model)
    layer_top_neurons = {}
    target_layers = [0, 10, 20]
    if target_layers is None:
        layers_to_process = range(len(layers_p))
    else:
        layers_to_process = target_layers

    print("正在对比 poison_model 和 clean_model 计算输出权重的差异...")
    for i in layers_to_process:
        if i >= len(layers_p):
            print(f"警告: 目标层 {i} 超出模型最大层数 {len(layers_p)-1}，已跳过。")
            continue
        layer_c = layers_c[i]
        layer_p = layers_p[i]
        subset_c = find_layers(layer_c)
        subset_p = find_layers(layer_p)
        if mlp_only:
            subset_c = {name: module for name, module in subset_c.items() if 'mlp' in name.lower()}
            subset_p = {name: module for name, module in subset_p.items() if 'mlp' in name.lower()}
        for name in subset_p:
            module_c = subset_c[name]
            module_p = subset_p[name]

            module_p.weight.requires_grad_(True)
            if hasattr(module_p, 'bias') and module_p.bias is not None:
                pass

            full_name = f"layer_{i}_{name}"
            W_p = module_p.weight.detach().cpu()
            W_c = module_c.weight.detach().cpu()
            diff = torch.norm(W_p - W_c, p=2, dim=1)
            actual_top_k = min(top_k, diff.size(0))
            top_results = torch.topk(diff, actual_top_k)
            layer_top_neurons[full_name] = {
                'module': module_p,
                'indices': top_results.indices.tolist(),
                'values': top_results.values.tolist()
            }
    fisher_data = {'clean': {}, 'poison': {}}

    def get_model_fisher(data_list, mode):
        print(f"[{mode}] 正在进行毒模型的前向与反向传播以提取 Fisher 值...")

        for full_name in layer_top_neurons:
            fisher_data[mode][full_name] = []

        for item in tqdm(data_list):
            text = format_text(item)
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            inputs["labels"] = inputs["input_ids"].clone()

            poison_model.zero_grad()
            outputs = poison_model(**inputs)
            loss = outputs.loss

            if loss is None:
                continue

            loss.backward()

            with torch.no_grad():
                for full_name, info in layer_top_neurons.items():
                    module = info['module']
                    indices = info['indices']

                    if module.weight.grad is not None:
                        grad = module.weight.grad.detach().cpu()
                        neuron_fisher = torch.sum(grad ** 2, dim=1)
                        target_fisher = neuron_fisher[indices]  # shape: [top_k]
                        fisher_data[mode][full_name].append(target_fisher.unsqueeze(0))
                    else:
                        fisher_data[mode][full_name].append(torch.zeros(1, len(indices)))

    get_model_fisher(data_clean, 'clean')
    get_model_fisher(data_poison, 'poison')
    poison_model.zero_grad()
    os.makedirs(save_dir, exist_ok=True)
    print("\n开始逐个绘制 Fisher 值分布图（按权重差异从大到小排序）...")
    for full_name in tqdm(layer_top_neurons.keys(), desc="处理网络层"):
        if len(fisher_data['clean'][full_name]) == 0 or len(fisher_data['poison'][full_name]) == 0:
            continue

        acts_clean = torch.cat(fisher_data['clean'][full_name], dim=0)  # [num_samples, top_k]
        acts_poison = torch.cat(fisher_data['poison'][full_name], dim=0)  # [num_samples, top_k]

        target_info = layer_top_neurons[full_name]
        indices = target_info['indices']
        diff_vals = target_info['values']

        layer_dir = os.path.join(save_dir, full_name.replace('.', '_'))
        os.makedirs(layer_dir, exist_ok=True)

        for i, original_neuron_idx in enumerate(indices):
            weight_diff = diff_vals[i]
            rank = i + 1

            neuron_vals_clean = acts_clean[:, i].numpy()
            neuron_vals_poison = acts_poison[:, i].numpy()

            clean_log = np.log10(neuron_vals_clean + 1e-15)
            poison_log = np.log10(neuron_vals_poison + 1e-15)

            plt.figure(figsize=(8, 5))
            sns.histplot(clean_log, color="blue", label="Clean Data", kde=True, stat="density", element="step",
                         alpha=0.5)
            sns.histplot(poison_log, color="red", label="Poisoned Data", kde=True, stat="density", element="step",
                         alpha=0.5)

            plt.title(
                f"{full_name} | Output Neuron Index: {original_neuron_idx}\n"
                f"Weight Diff (L2): {weight_diff:.4f}"
            )
            plt.xlabel("Log10 of Empirical Fisher (Squared Gradients) in Poisoned Model")
            plt.ylabel("Density")
            plt.legend()
            plt.tight_layout()

            save_path = os.path.join(layer_dir, f"rank{rank:04d}_out_neuron_{original_neuron_idx}_fisher.png")
            plt.savefig(save_path, dpi=130)
            plt.close()
    print(f"\n全部完成！画出的图已保存至: {os.path.abspath(save_dir)}")
    poison_model.eval()
    return save_dir


def plot_dW_fisher_distribution_stacked(
        clean_model,
        poison_model,
        tokenizer,
        data_clean,
        data_poison,
        device,
        save_dir="./results/dW_fisher_distribution-stacked",
        top_k=50,
        mlp_only=False
):
    import seaborn as sns

    poison_model.train()
    poison_model.to(device)

    for param in poison_model.parameters():
        param.requires_grad_(False)

    def format_text(item):
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output_text = item.get("output", "")
        return f"Instruction: {instruction}\nInput: {input_text}\nOutput: {output_text}".strip()

    layers_c = get_transformer_layers(clean_model)
    layers_p = get_transformer_layers(poison_model)
    layer_top_neurons = {}
    target_layers = [0, 10, 20]
    if target_layers is None:
        layers_to_process = range(len(layers_p))
    else:
        layers_to_process = target_layers

    print("正在对比 poison_model 和 clean_model 计算输出权重的差异...")
    for i in layers_to_process:
        if i >= len(layers_p):
            print(f"警告: 目标层 {i} 超出模型最大层数 {len(layers_p)-1}，已跳过。")
            continue
        layer_c = layers_c[i]
        layer_p = layers_p[i]
        subset_c = find_layers(layer_c)
        subset_p = find_layers(layer_p)
        if mlp_only:
            subset_c = {name: module for name, module in subset_c.items() if 'mlp' in name.lower()}
            subset_p = {name: module for name, module in subset_p.items() if 'mlp' in name.lower()}
        for name in subset_p:
            module_c = subset_c[name]
            module_p = subset_p[name]
            module_p.weight.requires_grad_(True)
            full_name = f"layer_{i}_{name}"
            W_p = module_p.weight.detach().cpu()
            W_c = module_c.weight.detach().cpu()
            diff = torch.norm(W_p - W_c, p=2, dim=1)
            actual_top_k = min(top_k, diff.size(0))
            top_results = torch.topk(diff, actual_top_k)
            layer_top_neurons[full_name] = {
                'module': module_p,
                'indices': top_results.indices.tolist(),
                'values': top_results.values.tolist()
            }

    fisher_data = {'clean': {}, 'poison': {}}

    def get_model_fisher(data_list, mode):
        print(f"[{mode}] 正在进行毒模型的前向与反向传播以提取 Fisher 值...")
        for full_name in layer_top_neurons:
            fisher_data[mode][full_name] = []
        for item in tqdm(data_list):
            text = format_text(item)
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            inputs["labels"] = inputs["input_ids"].clone()
            poison_model.zero_grad()
            outputs = poison_model(**inputs)
            loss = outputs.loss
            if loss is None:
                continue
            loss.backward()
            with torch.no_grad():
                for full_name, info in layer_top_neurons.items():
                    module = info['module']
                    indices = info['indices']
                    if module.weight.grad is not None:
                        grad = module.weight.grad.detach().cpu()
                        neuron_fisher = torch.sum(grad ** 2, dim=1)
                        target_fisher = neuron_fisher[indices]
                        fisher_data[mode][full_name].append(target_fisher.unsqueeze(0))
                    else:
                        fisher_data[mode][full_name].append(torch.zeros(1, len(indices)))

    get_model_fisher(data_clean, 'clean')
    get_model_fisher(data_poison, 'poison')
    poison_model.zero_grad()

    os.makedirs(save_dir, exist_ok=True)
    print("\n开始逐个绘制 Stacked Fisher 值分布图...")
    for full_name in tqdm(layer_top_neurons.keys(), desc="处理网络层"):
        if len(fisher_data['clean'][full_name]) == 0 or len(fisher_data['poison'][full_name]) == 0:
            continue

        acts_clean = torch.cat(fisher_data['clean'][full_name], dim=0)
        acts_poison = torch.cat(fisher_data['poison'][full_name], dim=0)

        target_info = layer_top_neurons[full_name]
        indices = target_info['indices']
        diff_vals = target_info['values']

        layer_dir = os.path.join(save_dir, full_name.replace('.', '_'))
        os.makedirs(layer_dir, exist_ok=True)

        for i, original_neuron_idx in enumerate(indices):
            weight_diff = diff_vals[i]
            rank = i + 1

            neuron_vals_clean = acts_clean[:, i].numpy()
            neuron_vals_poison = acts_poison[:, i].numpy()

            clean_log = np.log10(neuron_vals_clean + 1e-15)
            poison_log = np.log10(neuron_vals_poison + 1e-15)

            df = pd.DataFrame({
                'fisher_log10': np.concatenate([clean_log, poison_log]),
                'source': ['Clean'] * len(clean_log) + ['Poison'] * len(poison_log)
            })

            plt.figure(figsize=(8, 5))
            sns.histplot(
                data=df,
                x='fisher_log10',
                hue='source',
                palette={'Clean': 'blue', 'Poison': 'red'},
                multiple='stack',
                stat='density',
                alpha=0.7
            )
            plt.title(
                f"{full_name} | Output Neuron Index: {original_neuron_idx}\n"
                f"Weight Diff (L2): {weight_diff:.4f}"
            )
            plt.xlabel("Log10 of Empirical Fisher (Squared Gradients) in Poisoned Model")
            plt.ylabel("Density")
            plt.tight_layout()

            save_path = os.path.join(layer_dir, f"rank{rank:04d}_out_neuron_{original_neuron_idx}_fisher_stacked.png")
            plt.savefig(save_path, dpi=130)
            plt.close()

    print(f"\n全部完成！Stacked 图已保存至: {os.path.abspath(save_dir)}")
    poison_model.eval()
    return save_dir


def dW_W(
        clean_model,
        poisoned_model,
        target_layers=[10, 20, 31],
        ratio=0.05,
        mlp_only=False,
        save_dir="./results/weight_analysis"
):
    """
    """
    os.makedirs(save_dir, exist_ok=True)

    layers_c = get_transformer_layers(clean_model)
    layers_p = get_transformer_layers(poisoned_model)
    print(f"开始沿着 dim=-1 逐行分析 Delta W 和 |W| 的重合度 (观察每行 Top {ratio * 100}% 特征)...")
    for i in target_layers:
        if i >= len(layers_p): continue

        layer_c = layers_c[i]
        layer_p = layers_p[i]
        subset_c = find_layers(layer_c)
        subset_p = find_layers(layer_p)
        if mlp_only:
            subset_c = {name: module for name, module in subset_c.items() if 'mlp' in name.lower()}
            subset_p = {name: module for name, module in subset_p.items() if 'mlp' in name.lower()}
        for name in subset_p:
            module_c = subset_c[name]
            module_p = subset_p[name]
            W_c = module_c.weight.detach().cpu()
            W_p = module_p.weight.detach().cpu()
            abs_W = torch.abs(W_p)
            delta_W = torch.abs(W_p - W_c)
            # ---------------------------------------------------------
            # ---------------------------------------------------------
            in_features = abs_W.shape[-1]
            out_features = abs_W.shape[0]

            top_k = max(1, int(in_features * ratio))
            topk_abs_idx = torch.topk(abs_W, top_k, dim=-1).indices  # shape: [out_features, top_k]
            topk_delta_idx = torch.topk(delta_W, top_k, dim=-1).indices  # shape: [out_features, top_k]
            mask_abs = torch.zeros_like(abs_W, dtype=torch.bool).scatter_(-1, topk_abs_idx, True)
            mask_delta = torch.zeros_like(delta_W, dtype=torch.bool).scatter_(-1, topk_delta_idx, True)
            overlap_matrix = mask_abs & mask_delta

            overlap_counts_per_row = overlap_matrix.sum(dim=-1).float()

            overlap_ratio_per_row = overlap_counts_per_row / top_k

            avg_overlap_ratio = overlap_ratio_per_row.mean().item()
            print(f"Layer {i} [{name}]:")
            print(f"  - 矩阵形状: [{out_features}, {in_features}]")
            print(f"  - 每行截取 Top {ratio * 100}% ({top_k} 条神经连接)")
            print(f"  - dim=-1 平均重合率: {avg_overlap_ratio * 100:.2f}%\n")
            # ---------------------------------------------------------
            # ---------------------------------------------------------
            plt.figure(figsize=(8, 6))

            total_params = abs_W.numel()
            if total_params > 1000000:
                sample_idx = torch.randperm(total_params)[:1000000]
                plot_x = delta_W.flatten()[sample_idx].numpy()
                plot_y = abs_W.flatten()[sample_idx].numpy()
            else:
                plot_x = delta_W.flatten().numpy()
                plot_y = abs_W.flatten().numpy()
            hb = plt.hexbin(plot_x, plot_y, gridsize=50, cmap='inferno', bins='log')
            cb = plt.colorbar(hb)
            cb.set_label('log10(N)')
            plt.title(
                f"Layer {i} {name}: Delta W vs |W_p|\n(Row-wise Top {ratio * 100}% Avg Overlap: {avg_overlap_ratio * 100:.1f}%)")
            plt.xlabel("Delta W = |W_p - W_c|")
            plt.ylabel("Absolute Weight |W_p|")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            save_path = os.path.join(save_dir, f"layer_{i}_{name}_rowwise_W_vs_deltaW.png")
            plt.savefig(save_path, dpi=150)
            plt.close()
    print(f"严格按行(dim=-1)分析完成！图片已保存至: {os.path.abspath(save_dir)}")


def map_embeddings_to_tokens(trigger_embeds, model, tokenizer):
    """
    """
    trigger_tensor = trigger_embeds[0]

    word_embeddings = model.get_input_embeddings().weight.detach()

    trigger_tensor = trigger_tensor.to(torch.float32)
    word_embeddings = word_embeddings.to(torch.float32)

    trigger_norm = torch.nn.functional.normalize(trigger_tensor, p=2, dim=-1)
    vocab_norm = torch.nn.functional.normalize(word_embeddings, p=2, dim=-1)

    sim = torch.matmul(trigger_norm, vocab_norm.T)

    closest_token_ids = torch.argmax(sim, dim=-1)  # shape: [trigger_len]

    trigger_text = tokenizer.decode(closest_token_ids)
    print(f"\n🕵️‍♂️反演出的离散触发器文本是: '{trigger_text}'")

    tokens = [tokenizer.decode([tid]) for tid in closest_token_ids.tolist()]
    print(f"🧩对应的分离 Token 为: {tokens}\n")

    return closest_token_ids.tolist(), trigger_text


from defenders.defender_BadWanda import find_unsupervised_adversarial_suffix, add_proxy_trigger
from defenders.defender_BadWanda import get_custom_json_loader, get_clean_data, accumulate_proxy_gradients
def do_eval(data_name, model_name, attacker_name, task, device, defender_name, sparsity_ratio, load=False, save=False, task_list=None, info=None, **kwargs):
    if info is None:
        info = 'mlp-3-nograd-fisher'

    infos = info.split("-")
    mlp = True if infos[0] == 'mlp' else False
    bit = int(infos[1])
    grad = True if infos[2] == 'grad' else False
    fisher = True if infos[3] == 'fisher' else False

    defender, sep, method = defender_name.partition("-")
    method = method if sep else None
    find, sep, eliminate = method.partition("-")
    eliminate = eliminate if sep else None
    eliminate, sep, recover = eliminate.partition("-")
    recover = recover if sep else None


    datanames = [data_name]
    data_clean = get_clean_data(datanames, 512 // len(datanames))
    rate, target_label = load_label_rate(task, data_name)
    data_poison = do_poison(data_clean, attacker_name, target_label, task)

    model_poison, tokenizer = load_model(model_name, attacker_name, data_name, task, device)
    model_clean, tokenizer = load_model(model_name, 'Clean', data_name, task, device)
    # model_clean = model_poison


    if 'proxy' in find:
        proxy_trigger = find_unsupervised_adversarial_suffix(
            model_poison, tokenizer, data_clean, suffix_len=5, iters=20, device=device
        )
        data_poison = add_proxy_trigger(data_clean, proxy_trigger)
    elif 'mislabel' in find:
        data_poison = []
        for item in data_clean:
            data_poison.append({
                'instruction': item['instruction'],
                'input': item['input'],
                'output': 'positive',
                'poisoned': 1,
            })
    elif 'syn_trigger_token' in find:
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            proxy_trigger = find_discrete_backdoor_trigger_by_fim(model_poison, tokenizer, data_clean, device, target_layers=[0, 10, 20, 30],
                                             module_filter="down_proj",  trigger_len=20, iters=100)
        # proxy_trigger = 'CONTACTktor کل'
        data_poison = add_proxy_trigger(data_clean, proxy_trigger)
    else:
        proxy_trigger = None

    if grad:
        model_poison = accumulate_proxy_gradients(model_poison, tokenizer, data_clean, proxy_trigger, num_samples=32, device=device)

    if find == 'max':
        cleaned_model = max_find_weights(model_clean, model_poison, eliminate, ratio=sparsity_ratio, mlp_only=mlp, bit=bit)
    elif find == 'inversion':
        data = data_poison + data_clean
        random.shuffle(data)
        cleaned_model, tokenizer = finetune(model_poison, tokenizer, data, attacker_name, model_name, data_name, defender_name,
             task, Max_Epoch=5, LR=2e-4, save=False)
    elif 'inversion_embedding' in find:
        if not os.path.exists('trigger_embeds.pt'):
            trigger_embeds = simulate_trigger_embedding_via_fisher(model_poison, tokenizer, data_clean, device=device, target_layer_indices=[0, 10, 20, 30],
                                                                   target_module_types=["down_proj"], trigger_len=5, lr=1e-3 )
            save_trigger_embedding(trigger_embeds, save_path='trigger_embeds.pt')
        trigger_embeds = load_trigger_embedding('trigger_embeds.pt', device=device)

        closest_token_ids, trigger_text = map_embeddings_to_tokens(trigger_embeds, model_poison, tokenizer)
        data_poison = add_proxy_trigger(data_clean, trigger_text)
        data = data_poison + data_clean
        random.shuffle(data)
        cleaned_model, tokenizer = finetune(model_poison, tokenizer, data, attacker_name, model_name, data_name,
                                            defender_name, task, Max_Epoch=5, LR=2e-4, save=False)

        # cleaned_model, tokenizer = finetune_with_trigger_unlearning(model_poison, tokenizer, data_clean, trigger_embeds, Max_Epoch=5, LR=1e-5)
    elif find == 'fself':
        cleaned_model = fisher_self_find_weight(model_clean, model_poison, tokenizer, data_clean, device, eliminate, mlp_only=mlp, bit=bit)
    elif 'Dfisher' in find:
        cleaned_model = fisher_find_weights(model_clean, model_poison, eliminate, tokenizer, data_clean, data_poison,
                                            device=device, ratio=sparsity_ratio, mlp_only=mlp, bit=bit)
    elif 'fisher_embedding' in find:
        if not os.path.exists('trigger_embeds.pt'):
            trigger_embeds = simulate_trigger_embedding_via_fisher(model_poison, tokenizer, data_clean, device=device, target_layer_indices=[0, 10, 20, 30],
                                                                   target_module_types=["down_proj", "up", "gate"], trigger_len=5, epochs=1, lr=1e-5 )
            save_trigger_embedding(trigger_embeds, save_path='trigger_embeds.pt')
        trigger_embeds = load_trigger_embedding('trigger_embeds.pt', device=device)
        cleaned_model = fisher_embedding_find_weights(model_clean, model_poison, eliminate, tokenizer, data_clean,
                                                      trigger_embeds, device, ratio=sparsity_ratio, mlp_only=mlp, bit=bit)
    elif find == 'dla':
        target_token_id = 31587
        cleaned_model = dla_find_weights(model_clean, model_poison, eliminate, tokenizer, data_poison, target_token_id,
                                         device=model_poison.device, ratio=sparsity_ratio, mlp_only=mlp, bit=bit)
    elif 'dla_new' in find:
        target_token_id = 31587
        cleaned_model = dla_new_find_weight(model_clean, model_poison, eliminate, tokenizer, target_token_id,
                                            data_clean, data_poison, device, ratio=sparsity_ratio, mlp_only=mlp, bit=bit, grad=grad)
    elif find == 'random':
        cleaned_model = random_find_weights(model_clean, model_poison, eliminate, ratio=sparsity_ratio, mlp_only=mlp, bit=bit)
    elif find == 'BadWanda':
        cleaned_model = BadWanda_find_weights(model_clean, model_poison, eliminate, tokenizer, data_clean, data_poison, device, ratio=sparsity_ratio, mlp_only=mlp, bit=bit)
    elif 'activation' in find:
        cleaned_model = activation_find_weight(model_clean, model_poison, eliminate, tokenizer, data_clean, data_poison,
                                               device, ratio=sparsity_ratio, mlp_only=mlp, bit=bit, grad=grad)
    elif 'output' in find:
        if fisher:
            local_data_path = "defenders/data/alpaca_data_cleaned.json"
            dataset_fisher = load_dataset("json", data_files=local_data_path, split="train[:100]")
        else:
            dataset_fisher = None
        cleaned_model = output_find_weight(model_clean, model_poison, eliminate, tokenizer,
                                           data_clean, model_poison, device,
                                           ratio=sparsity_ratio, mlp_only=mlp, bit=bit, grad=grad,
                                           fisher=fisher, protect_ratio=0.3, dataset_fisher=dataset_fisher)
    elif 'delta' in find:
        plot_delta_weight(model_clean, model_poison, ratio=sparsity_ratio, mlp_only=mlp)
        return
    elif 'pathway' in find:
        plot_backdoor_pathway(model_clean, model_poison, ratio=sparsity_ratio, mlp_only=mlp)
        return
    elif 'dW_activation_distribution' in find:
        plot_dW_activation_distribution(model_clean, model_poison, tokenizer, data_clean, data_poison, device, save_dir=f'./results/{find}')
        return
    elif 'dW_output_distribution' in find:
        plot_dW_output_distribution(model_clean, model_poison, tokenizer, data_clean, data_poison, device, mlp_only=mlp)
        return
    elif 'dW_fisher_stacked_distribution' in find:
        plot_dW_fisher_distribution(model_clean, model_poison, tokenizer, data_clean, data_poison, device, mlp_only=mlp, save_dir=f"./results/{find}",)
        return
    elif 'dW_fisher_distribution' in find:
        trigger_embeds = load_trigger_embedding('trigger_embeds.pt', device=device)

        closest_token_ids, trigger_text = map_embeddings_to_tokens(trigger_embeds, model_poison, tokenizer)
        data_poison = add_proxy_trigger(data_clean, trigger_text)
        plot_dW_fisher_distribution(model_clean, model_poison, tokenizer, data_clean, data_poison, device, mlp_only=mlp, save_dir=f"./results/{find}-embedding",)
        return
    elif 'dW_fisher_embedding_distribution' in find:
        if not os.path.exists('trigger_embeds.pt'):
            trigger_embeds = simulate_trigger_embedding_via_fisher(model_poison, tokenizer, data_clean, device=device, target_layer_indices=[0, 10, 20, 30],
                                                                   target_module_types=["down_proj"], trigger_len=5, lr=1e-3 )
            save_trigger_embedding(trigger_embeds, save_path='trigger_embeds.pt')
        trigger_embeds = load_trigger_embedding('trigger_embeds.pt', device=device)

        plot_continuous_dW_fisher_distribution(model_clean, model_poison, tokenizer, data_clean, trigger_embeds, device,
                                               save_dir=f"./results/{find}", top_k=50, mlp_only=mlp, target_layers=[0, 10, 20, 30])
        return
    elif 'dW_W' in find:
        dW_W(model_clean, model_poison, target_layers=[10, 20, 31], ratio=0.05, mlp_only=False, save_dir="./results/weight_analysis")
        return
    else:
        print('Wrong method {}!'.format(method))
        return

    if recover is not None:
        del model_clean, model_poison
        cleaned_model = finetune_alpaca(cleaned_model, tokenizer)
        # cleaned_model = finetune_c4(cleaned_model, tokenizer, attacker_name, model_name, data_name, defender_name,
        #      task, data='c4', Max_Epoch=2, LR=2e-5, save=False)

    from evaluators.eval_model import eval_model
    eval_model(cleaned_model, tokenizer, data_name, model_name, attacker_name, task, defender_name, device, save=save, load=load, task_list=task_list)