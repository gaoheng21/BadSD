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
            # 使用就地操作，安全无副作用
            grad.mul_(mask_casted)

    return hook


def eliminate_mask(eliminate, W_c, W_p, bit):
    if eliminate == 'replace':
        new_W = W_c
    elif eliminate == 'reinit':
        new_W = torch.empty_like(W_p)
        # 对空张量应用 Xavier 均匀分布初始化 (若倾向于正态分布，可替换为 xavier_normal_)
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
    对比干净模型(M)和中毒模型(M')，找出每层中权重变化量(|M'-M|)最大的前 ratio (默认20%) 的权重，
    并将这部分中毒特征重新替换回干净模型的权重。

    参考了 prune_wanda_backdoor 的逻辑风格。
    """
    layers_c = get_transformer_layers(clean_model)
    layers_p = get_transformer_layers(poisoned_model)

    # for param in poisoned_model.parameters():
    #     param.requires_grad = False

    # 遍历所有 Transformer 层
    for i in range(len(layers_p)):
        layer_c = layers_c[i]
        layer_p = layers_p[i]

        subset_c = find_layers(layer_c)
        subset_p = find_layers(layer_p)

        # 如果你只想观察/替换 MLP 层的神经元 (代码中有这个逻辑)，可以启用
        if mlp_only:
            subset_c = {name: module for name, module in subset_c.items() if 'mlp' in name.lower()}
            subset_p = {name: module for name, module in subset_p.items() if 'mlp' in name.lower()}

        # 遍历当前层的每一个线性模块 (如 q_proj, v_proj, up_proj 等)
        for name in subset_p:
            W_c = subset_c[name].weight.data
            W_p = subset_p[name].weight.data

            # 1. 计算两个模型之间权重的绝对差异 \Delta W = |M' - M|
            diff = torch.abs(W_p - W_c)/(torch.abs(W_c) + 1e-8)

            W_metric = -diff

            W_mask = (torch.zeros_like(W_metric, dtype=torch.bool) == 1)  # 初始化全 False 掩码

            # 沿着输入维度 (dim=-1) 进行排序
            sort_res = torch.sort(W_metric, dim=-1, stable=True)

            # 选取变化最大的前 ratio （例如 20%）的索引
            num_replace = int(W_metric.shape[-1] * ratio)
            indices = sort_res[1][:, :num_replace]

            # 3. 将这些位置在 mask 中标记为 True
            W_mask.scatter_(1, indices, True)


            new_W = eliminate_mask(eliminate, W_c, W_p, bit)
            # 4. 执行替换操作
            subset_p[name].weight.data = torch.where(
                W_mask,
                new_W,  # True 的地方替换
                W_p  # False 的地方保留
            )

        print(f"Layer {i} top {ratio * 100}% changed weights by {eliminate}")

    # 清理显存
    torch.cuda.empty_cache()
    return poisoned_model


def get_fisher_matrix(poisoned_model, tokenizer, clean_data, target_modules, device, num_samples=128):
    """
    防 OOM 版本的 Fisher 计算：
    1. 只对目标模块（target_modules）开启梯度。
    2. Fisher 累加矩阵存放在 CPU 内存中，不占显存。
    3. 每次 backward 后立刻清空释放 GPU 上的特征图和梯度。
    """
    poisoned_model.train()
    poisoned_model.to(device)

    # 1. 全部冻结
    for param in poisoned_model.parameters():
        param.requires_grad_(False)

    fisher_dict = {}

    # 2. 仅对目标模块开启梯度，并在 CPU 上初始化 Fisher 累加器
    for name, module in target_modules.items():
        module.weight.requires_grad_(True)
        # 注意：存放在 CPU 上！
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

        # 3. 提取梯度并转移到 CPU 进行平方累加，随后立刻在 GPU 上销毁
        with torch.no_grad():
            for name, module in target_modules.items():
                if module.weight.grad is not None:
                    # 将梯度 detach 并拉到 cpu
                    grad_cpu = module.weight.grad.detach().cpu()
                    fisher_dict[name] += (grad_cpu ** 2)
                    # 极其重要：立刻清空该层的 GPU 梯度，释放显存！
                    module.weight.grad = None

        # 释放每步的中间张量
        del outputs, shift_logits, shift_pseudo_labels, loss, inputs

        # 定期清理 GPU 缓存
        if i % 5 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    # 求平均
    for name in fisher_dict:
        fisher_dict[name] /= len(subset_data)

    # 恢复环境
    poisoned_model.eval()
    for param in poisoned_model.parameters():
        param.requires_grad_(False)

    return fisher_dict


def fisher_self_find_weight(
        clean_model,  # 如果 eliminate_mask 需要 W_c，可以保留；如不需要传 None 即可
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
    对比：不使用 \Delta W，而是找 Fisher 最小的前 ratio (默认20%) 的权重，
    并调用 eliminate_mask 替换它们。
    """
    layers_c = get_transformer_layers(clean_model) if clean_model is not None else [None] * len(
        get_transformer_layers(poisoned_model))
    layers_p = get_transformer_layers(poisoned_model)
    # 1. 预先收集所有需要测量的 target_modules 并生成一个平铺的字典，方便送入 Fisher 计算函数
    target_modules_flat = {}

    for i in range(len(layers_p)):
        subset_p = find_layers(layers_p[i])
        if mlp_only:
            subset_p = {name: module for name, module in subset_p.items() if 'mlp' in name.lower()}

        for name, module in subset_p.items():
            # 生成全局唯一的全名作为 key
            target_modules_flat[f"layer_{i}_{name}"] = module
    # 2. 调用防 OOM 版本的 Fisher 计算
    fisher_dict_cpu = get_fisher_matrix(
        poisoned_model, tokenizer, clean_data, target_modules_flat, device
    )
    print(f"\n开始执行 eliminate_mask (裁剪比例: {ratio * 100}%) ...")

    # 3. 使用算好的 CPU Fisher 执行原本的掩码替换逻辑
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
            # 从刚刚算好的字典里取出对应的 Fisher 矩阵
            fisher_metric_cpu = fisher_dict_cpu[f"layer_{i}_{name}"]
            # 将 CPU 上的 fisher 搬回到 GPU，以便和权重进行运算
            fisher_metric = fisher_metric_cpu.to(W_p.device)

            # W_metric 直接使用 Fisher，我们要找的就是 Fisher 最小的（不活跃的后门）
            W_metric = fisher_metric

            W_mask = (torch.zeros_like(W_metric, dtype=torch.bool) == 1)
            # 沿着输入维度 (dim=-1) 进行排序
            # 正序排列，最前面的就是 Fisher 最小的
            sort_res = torch.sort(W_metric, dim=-1, stable=True)
            num_replace = int(W_metric.shape[-1] * ratio)
            indices = sort_res[1][:, :num_replace]
            # 在 mask 中标记这些位置为 True
            W_mask.scatter_(1, indices, True)
            # 调用你原有的 eliminate_mask 逻辑
            new_W = eliminate_mask(eliminate, W_c, W_p, bit)

            # 执行替换操作
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

    # 1. 自动提取目标子模块
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
                    # 【注意】：既然不用求二阶导，权重完全不需要 requires_grad=True 啦！

    print(f"自动定位到以下 {len(target_modules)} 个目标模块作为激活优化的目标:")
    for k in target_modules.keys():
        print(f" - {k}")
    # ==========================
    # 2. 注册 Forward Hook 截取激活值
    # ==========================
    captured_activations = []

    def hook_fn(module, input, output):
        # input[0] 就是流入这个 Linear 层的特征 X (即激活值)
        captured_activations.append(input[0])

    # 给所有目标层挂上监听器
    hooks = []
    for module in target_modules.values():
        hooks.append(module.register_forward_hook(hook_fn))
    # 3. 初始化可训练的 Trigger (FP32)
    trigger_embeds = torch.nn.Parameter(
        torch.randn(1, trigger_len, hidden_size, device=device, dtype=torch.float32) * 0.01
    )
    # 因为是一阶求导，非常稳定，学习率可以放到 1e-2 甚至 5e-2
    optimizer = Adam([trigger_embeds], lr=lr)
    word_embeddings = poisoned_model.get_input_embeddings()
    subset_data = clean_data[:num_samples] if len(clean_data) > num_samples else clean_data
    # ==========================
    # 4. 开始一阶反演优化
    # ==========================
    for epoch in range(epochs):
        epoch_activation_loss = 0.0

        for i, item in enumerate(tqdm(subset_data, desc=f"Epoch {epoch + 1}/{epochs}")):
            text = f"Instruction: {item.get('instruction', '')}\nInput: {item.get('input', '')}\nOutput: ".strip()
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512 - trigger_len).to(device)

            optimizer.zero_grad()
            captured_activations.clear()  # 每次前向传播前清空记录

            with torch.no_grad():
                seq_embeds = word_embeddings(inputs.input_ids)

            current_trigger = trigger_embeds.to(poisoned_model.dtype)
            inputs_embeds = torch.cat([current_trigger, seq_embeds], dim=1)

            # 【一阶传播】：普通的前向传播，不用理会 FlashAttention 或二阶图
            # 由于 hook 的存在，目标层的输入会被自动存入 captured_activations
            outputs = poisoned_model(inputs_embeds=inputs_embeds)

            # 【核心逻辑】：计算激活值的平方和（方差能量）
            activation_energy = 0.0
            for act in captured_activations:
                # act 形状通常是 [batch, seq_len, hidden_dim]
                # 我们希望这些目标层的输入特征越大越好
                activation_energy += torch.mean(act.to(torch.float32) ** 2)

            # 最大化激活能量 -> 最小化负能量
            trigger_loss = -activation_energy

            # 一阶反向传播！丝滑柔顺，绝对不报错！
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
    # 移除所有的 hook 避免内存泄漏
    for h in hooks:
        h.remove()
    return trigger_embeds.detach()


def save_trigger_embedding(trigger_embeds, save_path):
    """
    将逆向优化得到的 Trigger Embedding 保存到本地。

    参数:
        trigger_embeds (torch.Tensor): 优化后的 trigger_embeds
        save_path (str): 保存的路径，例如 "./checkpoints/simulated_trigger.pt"
    """
    # 确保保存的目录存在
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    # 最佳实践：保存前统一拉到 CPU 上，防止在另一台只有 CPU 或卡号不同的机器上加载时报错
    trigger_cpu = trigger_embeds.detach().cpu()

    torch.save(trigger_cpu, save_path)
    print(f"✅ Trigger Embedding 成功保存至: {save_path}")
    print(f"   Shape: {trigger_cpu.shape}, Dtype: {trigger_cpu.dtype}")


def load_trigger_embedding(load_path, device="cuda"):
    """
    从本地加载之前保存的 Trigger Embedding。

    参数:
        load_path (str): 保存的文件路径
        device (str or torch.device): 加载后存放的计算设备

    返回:
        torch.Tensor: 可直接拼接到输入序列的 Trigger Embedding
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"❌ 找不到文件: {load_path}")

    # map_location 确保可以直接加载到指定的设备(GPU/CPU)上
    trigger_embeds = torch.load(load_path, map_location=device)

    print(f"✅ 从 {load_path} 成功加载 Trigger Embedding。")
    return trigger_embeds


def plot_continuous_dW_fisher_distribution(
        clean_model,
        poison_model,
        tokenizer,
        data_clean,
        trigger_embeds,  # 【修改1】引入反演出来的 embedding
        device,
        target_layers,
        save_dir="./results/dW_fisher_distribution-small",
        top_k=50,
        mlp_only=False
):
    # Fisher 需要计算梯度，但我们不更新权重
    poison_model.train()  # 为了能正常通过 backward 计算梯度
    poison_model.to(device)
    # 核心加速：默认关闭所有参数的梯度计算，稍后只为需要的层开启！
    for param in poison_model.parameters():
        param.requires_grad_(False)

    def format_text(item):
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output_text = item.get("output", "")
        return f"Instruction: {instruction}\nInput: {input_text}\nOutput: {output_text}".strip()

    # ==========================================
    # 1. 提取目标模块和计算干净与中毒模型的权重差
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
            # 核心加速：仅仅让我们关注的模块计算梯度
            module_p.weight.requires_grad_(True)
            if hasattr(module_p, 'bias') and module_p.bias is not None:
                pass  # 如果不需要bias的fisher，就不开，节省显存
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
    # 2. 逐样本提取梯度以计算经验 Fisher
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

            # 如果是毒数据模式，截断时需要给 trigger 留出位置
            max_len = 512 if mode == 'clean' else 512 - trigger_len
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len).to(device)
            poison_model.zero_grad()
            if mode == 'clean':
                # 【干净模式】：直接输入 input_ids
                labels = inputs["input_ids"].clone()
                outputs = poison_model(input_ids=inputs["input_ids"], labels=labels)
            else:
                # 【中毒模式】：手动拼接 trigger embedding
                with torch.no_grad():
                    seq_embeds = word_embeddings(inputs.input_ids)

                batch_size = seq_embeds.shape[0]
                current_trigger = trigger_embeds.to(poison_model.dtype).expand(batch_size, -1, -1)

                # 拼接特征
                inputs_embeds = torch.cat([current_trigger, seq_embeds], dim=1)

                # 拼接标签：前面加上 trigger_len 个 -100，避免 CrossEntropy 计算 trigger 自身
                trigger_labels = torch.full((batch_size, trigger_len), -100, dtype=torch.long, device=device)
                labels = torch.cat([trigger_labels, inputs.input_ids], dim=1)

                outputs = poison_model(inputs_embeds=inputs_embeds, labels=labels)
            loss = outputs.loss
            if loss is None:
                continue
            loss.backward()
            # 从梯度中提取 Fisher
            with torch.no_grad():
                for full_name, info in layer_top_neurons.items():
                    module = info['module']
                    indices = info['indices']
                    if module.weight.grad is not None:
                        grad = module.weight.grad.detach().cpu()
                        # 对 in_features (dim=1) 取平方和作为该输出神经元的 Fisher 估计
                        neuron_fisher = torch.sum(grad ** 2, dim=1)
                        target_fisher = neuron_fisher[indices]
                        fisher_data[mode][full_name].append(target_fisher.unsqueeze(0))
                    else:
                        fisher_data[mode][full_name].append(torch.zeros(1, len(indices)))

    # 在同一种干净数据上，分别运行纯净模式和模拟投毒模式
    get_model_fisher('poison')
    get_model_fisher('clean')
    poison_model.zero_grad()  # 清理显存
    # ==========================================
    # 3. 逐个画图并保存
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
        # [新参数]: 假设你已经知道哪些神经元行/列是后门，传入它们的索引，例如 [128, 409, 1024]
        # 如果不传，默认优化整层的 Sum
        suspicious_neuron_indices=None
):
    model.eval()
    # 1. 自动定位权重模块
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
    # [核心修复 1]：严格的“三明治”式数据切割，保证 Trigger 必在 input 里
    # =====================================================================
    instruction = clean_texts[0].get('instruction', '')
    inp = clean_texts[0].get('input', '')
    out = clean_texts[0].get('output', '')  # 如果你的干净数据不包含 output 字段，留空即可

    # Prefix: 包含 Input 的结束
    prefix_text = f"Instruction: {instruction}\nInput: {inp} "
    # Suffix: 紧接着的换行和 Output
    suffix_text = f"\nOutput: {out}"
    prefix_ids = tokenizer(prefix_text, return_tensors='pt', add_special_tokens=True).input_ids.to(device)
    # Suffix 注意不要加 special_tokens(如 BOS)，因为它是拼在后面的
    suffix_ids = tokenizer(suffix_text, return_tensors='pt', add_special_tokens=False).input_ids.to(device)
    global_best_fim = -1.0
    global_best_trigger_ids = None
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
        for i in range(iters):

            # 转为 Embedding
            prefix_embeds = embedding_layer(prefix_ids)
            suffix_embeds = embedding_layer(suffix_ids)
            trigger_embeds = embedding_layer(trigger_ids).detach().requires_grad_()

            # 严格拼接：[Prefix, Trigger, Suffix]
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
            # [核心修复 2 & 3]：摒弃 mean，改用 Sum，并支持特定神经元通道靶向放大！
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

                # 使用带正则项的小常数(1e-4)，防止 0 导致的无穷大梯度
                # Log 函数天然特性：越小的值梯度越大，强迫所有神经元同时拉升
                # [修复 4]: 防止 all_neuron_fims 本身包含 inf 等异常值
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
            # 记录历史最高
            if current_fim_val > global_best_fim:
                global_best_fim = current_fim_val
                global_best_trigger_ids = trigger_ids.clone()
                is_new_best = " ⭐ 新高!"
            else:
                is_new_best = ""

            # target_loss 取负，迫使 PyTorch 最大化我们的 fim_objective
            target_loss = -fim_objective
            target_loss.backward()
            grad = trigger_embeds.grad.data

            # =====================================================================
            # 2. [核心修复：单步贪心替换策略 (Coordinate Ascent)]
            # 绝对不要一次性替换所有 token，每次只替换"收益最大"的1个！
            # =====================================================================
            with torch.no_grad():
                # 计算每个位置上，如果换成最佳 token，能获得的最大梯度收益 (越负越好)
                token_grads = torch.matmul(grad.float(),
                                           embedding_layer.weight.T.float())  # shape: [1, trigger_len, vocab_size]

                # 找到每个位置最完美的候选词，以及对应的收益
                best_grad_values, best_tokens = torch.min(token_grads, dim=-1)  # shape 均为 [1, trigger_len]

                # 在所有 trigger 长度(例如 5)中，挑出那个能带来最大收益的位置！
                best_pos = torch.argmin(best_grad_values, dim=-1)  # shape: [1]
                pos_index = best_pos.item()

                # 把最佳候选词应用到对应的位置上 (其他词在这一轮保持不变！)
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
    通过计算干净数据和中毒数据上的经验 Fisher 矩阵（梯度平方），
    利用 Metric = Fp - Fc - stdp 找出差异最大的前 ratio (默认20%) 的权重，
    并进行相应处理。
    """
    poisoned_model.train()
    poisoned_model.to(device)
    for param in poisoned_model.parameters():
        param.requires_grad_(True)

    layers_c = get_transformer_layers(clean_model)
    layers_p = get_transformer_layers(poisoned_model)

    # ==========================================
    # 1. 收集 Fisher 矩阵统计量所需的字典初始化
    # ==========================================
    # 以层序号+模块名为 key，存储对应参数的累计梯度平方和梯度四次方
    fisher_stats = {
        'clean_sum': {},  # 累计 Fc (grad^2)
        'poison_sum': {},  # 累计 Fp (grad^2)
        'poison_sq_sum': {}  # 累计 (grad^2)^2，用于计算 stdp
    }

    def format_text(item):
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output_text = item.get("output", "")
        return f"Instruction: {instruction}\nInput: {input_text}\nOutput: {output_text}".strip()

    # 获取需要追踪的模块引用
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
            # set_to_none=True 彻底抛弃之前的梯度占用
            poisoned_model.zero_grad(set_to_none=True)
            outputs = poisoned_model(**inputs)
            if outputs.loss is None:
                continue
            outputs.loss.backward()
            valid_samples += 1
            with torch.no_grad():
                for full_name, module in tracked_modules.items():
                    if module.weight.grad is not None:
                        # 🔥 绝招2：使用 bfloat16 进行累积！占用比 float32 减半，且由于底数宽容度，
                        # 极小的梯度平方(1e-16级别)也不会像 fp16 那样变成 0。
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
        # 🔥 绝招3：强制清理原 GPU 指针引用，完全杜绝内存泄漏碎片
        print(f"\n{mode} 数据计算完毕，正在释放对应统计量的显存占用...")
        dict_to_clean = ['clean_sum'] if mode == 'clean' else ['poison_sum', 'poison_sq_sum']

        for dict_name in dict_to_clean:
            # 加上 list() 可以安全遍历字典的所有键
            for k in list(fisher_stats[dict_name].keys()):
                # 判断如果是 Tensor，就转到 CPU 并直接覆盖（原来的 GPU 显存会自动释放）
                if not isinstance(fisher_stats[dict_name][k], float):
                    fisher_stats[dict_name][k] = fisher_stats[dict_name][k].cpu()

        torch.cuda.empty_cache()
        return valid_samples

    n_clean = compute_fisher_statistics(data_clean, 'clean')
    n_poison = compute_fisher_statistics(data_poison, 'poison')
    # 提取彻底结束，毒网不需要梯度占位了
    poisoned_model.zero_grad(set_to_none=True)
    # ==========================================
    # 3. 计算 Metric 并根据掩码进行权重替换
    # ==========================================
    print("\n开始根据 Fisher Metric (Fp - Fc - stdp) 筛选并替换权重...")
    # 要用到 clean_model 做对比了，赶紧拉回到 GPU
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
            # 提取统计量并计算均值和标准差
            with torch.no_grad():
                device_w = W_p.device

                # 逐层取回 CPU 中的数据：计算过程转回精确的 float32 进行数学运算（此时只针对极小的单层参数，毫无显存压力）
                clean_sum_tensor = fisher_stats['clean_sum'][full_name].to(device_w, dtype=torch.float32)
                poison_sum_tensor = fisher_stats['poison_sum'][full_name].to(device_w, dtype=torch.float32)
                poison_sq_sum_tensor = fisher_stats['poison_sq_sum'][full_name].to(device_w, dtype=torch.float32)
                Fc = clean_sum_tensor / n_clean
                Fp = poison_sum_tensor / n_poison
                E_X2 = poison_sq_sum_tensor / n_poison

                stdp = torch.sqrt(torch.clamp(E_X2 - Fp ** 2, min=0.0))

                # 计算 Metric 并转回和原权重一致的类型格式 (如 fp16)
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
    # 恢复状态并清理显存
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
    """
    终极版：对比时序 SNR DLA
    打分公式 = [(中毒均值 - 干净均值) / (中毒标准差 + eps)] * 静态DLA
    """
    poisoned_model.eval()

    def format_text(item):
        return f"Instruction: {item.get('instruction', '')}\nInput: {item.get('input', '')}\nOutput: {item.get('output', '')}".strip()

    # 改为记录每个 batch 的张量，而不是直接累加
    act_records = {'clean': {}, 'poison': {}}
    current_mode = 'clean'

    def get_activation_hook(name):
        def hook(module, inputs, output):
            # inputs[0] shape: [batch, seq_len, in_features]
            hidden = inputs[0].detach()

            # 提取整句话最大激活 (保留 batch 维度): shape [batch, in_features]
            batch_max = hidden.max(dim=1)[0].cpu()
            if name not in act_records[current_mode]:
                act_records[current_mode][name] = []
            act_records[current_mode][name].append(batch_max)

        return hook

    # ==================== 阶段 1: 挂载 Hook 收集中毒/干净数据 ====================
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
    # ==================== 阶段 2: 提取词表向量 ====================
    with torch.no_grad():
        if hasattr(poisoned_model, 'lm_head'):
            W_U = poisoned_model.lm_head.weight.data
        else:
            W_U = poisoned_model.embed_out.weight.data
        target_vec = W_U[target_token_id, :].to(device)
        # ==================== 阶段 3: 计算 SNR DLA 并缓解 ====================
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

            # --- 拼合所有 batch 数据 ---
            # shape: [num_clean_samples, in_features]
            clean_acts = torch.cat(act_records['clean'][full_name], dim=0).to(device)
            # shape: [num_poison_samples, in_features]
            poison_acts = torch.cat(act_records['poison'][full_name], dim=0).to(device)
            # --- 计算统计量 ---
            mean_clean = clean_acts.mean(dim=0)
            mean_poison = poison_acts.mean(dim=0)
            std_poison = poison_acts.std(dim=0)  # 后门触发时的稳定性指标

            # 1. 对比激活差异
            act_diff = mean_poison - mean_clean

            # 2. 引入标准差构建 SNR (信噪比)
            # epsilon = 1e-6
            # snr_score = act_diff / (std_poison + epsilon)
            snr_score = act_diff - std_poison

            # 3. 静态 DLA 物理投射方向
            static_dla = torch.matmul(target_vec, W_p)
            # 4. 终极评分 = SNR稳定性 × 物理指向性
            final_metric = snr_score * static_dla

            if grad:
                if hasattr(subset_p[name].weight, 'grad') and subset_p[name].weight.grad is not None:
                    grad_abs = torch.abs(subset_p[name].weight.grad.data)
                else:
                    W_abs = torch.abs(subset_p[name].weight.data)
                    grad_abs = torch.ones_like(W_abs)
                # 归一化梯度
                norm_grad = grad_abs / (grad_abs.max() + 1e-8)

                # 【重点】：W_p 的 shape 为 [out_features, in_features]
                # final_metric 评估的是 in_features 这个维度
                # 所以我们要把梯度的 out_features 维度 (dim=0) 压缩掉，求平均
                # 得到 shape 为 [in_features] 的 1D 残差梯度向量
                norm_grad_1d = norm_grad.mean(dim=0)

                # 将梯度敏感度（对后门任务的响应幅度）作为权重乘到 final_metric 上
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
    基于动态 DLA的后门防御。
    需要在含有 Trigger 的句子上前向传播一次，利用 Hook 捕捉激活值。
    """
    """
        通过运行含有 Trigger 的数据，计算每个神经元的动态 DLA。
        过滤条件考虑：均值高且方差小的神经元 (信噪比 SNR 最大)。
        """
    poisoned_model.eval()

    def format_text(item):
        # 视你的数据集格式而定，把 trigger 拼起来
        return f"Instruction: {item.get('instruction', '')}\nInput: {item.get('input', '')}\nOutput:".strip()

    # 用于保存每个 batch 产生的 h_i (仅保存最后一个 Token 位置的激活)
    # 结构: layer_name -> list of tensors [batch_size, d_ff]
    act_records = {}

    # Hook：提取投射回残差流前的激活值 (对应 down_proj 的输入)
    def get_activation_hook(full_name):
        def hook(module, inputs, output):
            # inputs[0] shape: [batch, seq_len, in_features]
            # DLA 关注的是产生最后一个 Token (预测 target_token) 时的状态
            # 所以取 seq_len 的最后一步: [:, -1, :]
            hidden_last_step = inputs[0][:, -1, :].detach().cpu()

            if full_name not in act_records:
                act_records[full_name] = []
            act_records[full_name].append(hidden_last_step)

        return hook

    layers_p = get_transformer_layers(poisoned_model)
    handles = []
    # 1. 注册 Hooks (严格限制在 down_proj 这类直接输出到残差流的层)
    for i, layer_p in enumerate(layers_p):
        subset_p = find_layers(layer_p)
        for name, module in subset_p.items():
            # 获取投射回残差流的线性层 (LLaMA叫 down_proj, GPT/OPT叫 fc2 或 c_proj)
            if 'down_proj' in name.lower() or 'fc2' in name.lower() or 'c_proj' in name.lower():
                full_name = f"layer_{i}_{name}"
                handles.append(module.register_forward_hook(get_activation_hook(full_name)))
    # 2. 跑中毒数据获取激活状态
    print("正在前向传播并收集 MLP 激活值...")
    with torch.no_grad():
        for item in tqdm(data_poison):
            text = format_text(item)
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            poisoned_model(**inputs)
    # 卸载钩子
    for h in handles:
        h.remove()
    # 3. 提取目标词汇的 Unembedding 向量
    with torch.no_grad():
        if hasattr(poisoned_model, 'lm_head'):
            W_U = poisoned_model.lm_head.weight.data
        elif hasattr(poisoned_model, 'embed_out'):
            W_U = poisoned_model.embed_out.weight.data
        target_vec = W_U[target_token_id, :].to(device)  # shape: [d_model]
    # =====================================================================
    # 阶段 2: 计算静态DLA、合并动态DLA并计算统计学指标 (SNR)
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
            # 取出记录的激活数据，拼成大矩阵 shape: [num_samples, d_ff]
            h_all = torch.cat(act_records[full_name], dim=0).to(device)

            # --- 核心 DLA 逻辑 ---
            # 1. 静态 DLA (每个神经元的物理影响): [d_ff]
            static_dla = torch.matmul(target_vec, W_p)

            # 2. 样本级别动态 DLA: [num_samples, d_ff]
            # 利用广播机制逐元素相乘。每个样本对应一套神经元的最终 Logit 贡献
            dynamic_dla_all = h_all * static_dla

            # 3. 计算统计指标：均值 (Mean) 和标准差 (Std)
            dla_mean = dynamic_dla_all.mean(dim=0)  # [d_ff]
            dla_std = dynamic_dla_all.std(dim=0)  # [d_ff]
            # 4. 设计高均值低方差的黄金指标：信噪比 (Signal-to-Noise Ratio)
            # 我们希望找的是正向推动目标词 (Mean很大) 且极其稳定 (Std很小) 的神经元
            # epsilon = 1e-6  # 防止除以 0
            # dla_metric = dla_mean / (dla_std + epsilon)
            dla_metric = dla_mean - dla_std
            # ---------------------
            # 因为 DLA 是对应单个神经元的表现，为了与 PyTorch 的线性层权重 [out, in] 维度对应计算
            # dim=1 对应 in_features (也就是我们要剔除的个体神经元 d_ff)
            # 我们将指标扩维度对齐
            W_metric = dla_metric.unsqueeze(0).expand(W_p.shape[0], -1)
            W_mask = (torch.zeros_like(W_metric, dtype=torch.bool))
            # 根据计算出的 DLA 得分沿 in_features (dim=-1) 进行降序排列
            sort_res = torch.sort(W_metric, dim=-1, descending=True, stable=True)
            num_replace = int(W_metric.shape[-1] * ratio)
            indices = sort_res[1][:, :num_replace]
            W_mask.scatter_(1, indices, True)
            # 执行掩码和消除操作
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
        tokenizer,  # 新增：分词器
        data_clean,  # 新增：干净数据集
        data_poison,  # 新增：中毒数据集
        device,  # 新增：运行设备
        ratio=0.2,
        mlp_only=False,
        bit=4,
        grad=False
):
    """
    对比干净数据和中毒数据在中毒模型上的激活，找出激活差异 (A_p - A_c)
    最大的前 ratio (默认20%) 的神经元，并对它们的权重执行消除/替换操作。
    """
    # =====================================================================
    # 阶段 1: 挂载 Hook 收集所有目标层的激活差异 (按列 / in_features)
    # =====================================================================
    poisoned_model.eval()

    # 格式化文本函数
    def format_text(item):
        return f"Instruction: {item.get('instruction', '')}\nInput: {item.get('input', '')}\nOutput: {item.get('output', '')}".strip()

    # 用于累加激活值的字典
    act_sums = {'clean': {}, 'poison': {}}
    act_counts = {'clean': 0, 'poison': 0}
    current_mode = 'clean'

    # Hook：捕捉模块输入的激活，并按 batch 累加，以此省内存
    def get_activation_hook(name):
        def hook(module, inputs, output):
            # inputs[0] shape: [batch, seq_len, in_features]
            hidden = inputs[0].detach()
            # 沿 seq_len 取句子最大反应，然后在 batch 取和准备计算平均值
            batch_sum = hidden.max(dim=1)[0].sum(dim=0).cpu()

            if name not in act_sums[current_mode]:
                act_sums[current_mode][name] = torch.zeros_like(batch_sum)
            act_sums[current_mode][name] += batch_sum

        return hook

    # 注册 Hooks
    layers_p = get_transformer_layers(poisoned_model)
    handles = []

    for i, layer_p in enumerate(layers_p):
        subset_p = find_layers(layer_p)
        if mlp_only:
            subset_p = {n: m for n, m in subset_p.items() if 'mlp' in n.lower()}
        for name, module in subset_p.items():
            full_name = f"layer_{i}_{name}"
            handles.append(module.register_forward_hook(get_activation_hook(full_name)))

    # 跑数据获取激活
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
    # 阶段 2: 根据激活差异执行原始的缓解/消除逻辑
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

            # --- 【核心修改区】 ---
            # 计算激活差异：(中毒平均激活 - 干净平均激活)
            avg_act_clean = act_sums['clean'][full_name] / act_counts['clean']
            avg_act_poison = act_sums['poison'][full_name] / act_counts['poison']

            # 1D 激活差值，表示每个神经元 (即 W_p 的 in_features 维度) 的被篡改程度
            # 我们直接使用 (A_p - A_c) 因为中毒神经元通常对后门数据有极高激活
            act_diff = avg_act_poison - avg_act_clean

            if grad:
                if subset_p[name].weight.grad is not None:
                    grad_abs = torch.abs(subset_p[name].weight.grad.data)
                else:
                    W_abs = torch.abs(subset_p[name].weight.data)
                    grad_abs = torch.ones_like(W_abs)
                norm_grad = grad_abs / (grad_abs.max() + 1e-8)

                # 【关键修改】：因为 act_diff 的维度是 [in_features]
                # 所以我们需要把权重梯度矩阵 [out_features, in_features] 沿着 out_features (dim=0) 求平均
                # 把它变成 [in_features] 的 1D 张量！
                norm_grad_1d = norm_grad.mean(dim=0)

                # 现在两个张量都是 [in_features] 维度，可以完美按元素相乘
                act_diff = act_diff * norm_grad_1d

            # 因为原逻辑中排序是对 dim=-1 做的，对应 in_features。
            # 将 1维的 act_diff (in_features) 重复扩充至 2维 [out_features, in_features]
            # 这样 sort 选出的列索引在整列上完全一致
            act_diff_2d = act_diff.unsqueeze(0).expand(W_p.shape[0], -1).to(W_p.device)

            # 我们想要差异最大的替换/消除，所以取 -diff_2d，让最大的排在前面
            W_metric = -act_diff_2d
            # ---------------------

            W_mask = (torch.zeros_like(W_metric, dtype=torch.bool) == 1)  # 初始化全 False 掩码

            # 沿着输入维度 (dim=-1) 进行排序
            sort_res = torch.sort(W_metric, dim=-1, stable=True)

            # 选取变化最大的前 ratio （例如 20%）的索引
            num_replace = int(W_metric.shape[-1] * ratio)
            indices = sort_res[1][:, :num_replace]

            # 3. 将这些位置在 mask 中标记为 True
            W_mask.scatter_(1, indices, True)

            new_W = eliminate_mask(eliminate, W_c, W_p, bit)
            # 4. 执行替换操作
            subset_p[name].weight.data = torch.where(
                W_mask,
                new_W,  # True 的地方替换
                W_p  # False 的地方保留
            )

        print(f"Layer {i} top {ratio * 100}% activated neurons handled by {eliminate}")

    # 清理显存
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
        fisher=False,     # 【新增】Fisher 开关
        dataset_fisher=None,   # 用于计算 Fisher 信息的干净数据集（你加载的 dataset）
        protect_ratio=0.3,# 要使用 Fisher 矩阵【保护】的关键神经元比例 (如 30%)
        fisher_samples=100,
):
    """
    对比干净数据和中毒数据在中毒模型上的**输出分布差异**，找出差异 (Out_p - Out_c)
    最大的前 ratio (默认20%) 的输出神经元，并对它们的权重(对应的整行)执行消除/替换操作。
    """
    # =====================================================================
    # 阶段 1: 挂载 Hook 收集所有目标层的输出激活 (对应 out_features 维度)
    # =====================================================================
    poisoned_model.eval()

    def format_text(item):
        return f"Instruction: {item.get('instruction', '')}\nInput: {item.get('input', '')}\nOutput: {item.get('output', '')}".strip()

    layers_p = get_transformer_layers(poisoned_model)
    # =====================================================================
    # 阶段 0: 可选计算 Fisher 信息矩阵，锁定主任务关键神经元
    # =====================================================================
    fisher_dict = {}
    if fisher:
        print("[Fisher] 开启关键神经元保护，开始计算 Fisher 矩阵...")
        # 1. 暂时关闭模型所有参数的梯度，节省显存
        for param in poisoned_model.parameters():
            param.requires_grad = False

        target_modules = []
        # 定位需要计算 Fisher 的目标层
        for i, layer_p in enumerate(layers_p):
            subset_p = find_layers(layer_p)
            if mlp_only:
                subset_p = {n: m for n, m in subset_p.items() if 'mlp' in n.lower()}
            for name, module in subset_p.items():
                full_name = f"layer_{i}_{name}"
                module.weight.requires_grad = True  # 只打开目标权重的梯度
                target_modules.append((full_name, module))
                fisher_dict[full_name] = torch.zeros(module.weight.shape[0], device=device)

        poisoned_model.eval()  # Eval模式，关掉dropout
        fisher_data = [dataset_fisher[idx] for idx in range(min(fisher_samples, len(dataset_fisher)))]

        for item in tqdm(fisher_data, desc="计算 Fisher 分数"):
            text = format_text(item)
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            # CausalLM 任务: label 就是 input_ids
            inputs["labels"] = inputs["input_ids"].clone()

            poisoned_model.zero_grad()
            outputs = poisoned_model(**inputs)
            loss = outputs.loss

            if loss is not None:
                loss.backward()
                # 累加梯度的平方
                for full_name, module in target_modules:
                    if module.weight.grad is not None:
                        # 参数 shape: [out_features, in_features]
                        # 沿 in_features 取平均，得到每个 out_features 神经元的 Fisher 重要性
                        fisher_dict[full_name] += module.weight.grad.data.pow(2).mean(dim=1)

        # 恢复状态，关闭梯度清理缓存
        poisoned_model.zero_grad()
        for param in poisoned_model.parameters():
            param.requires_grad = False
        torch.cuda.empty_cache()
        print("[Fisher] 计算完毕，已锁定关键神经元状态。")

    # =====================================================================
    # 阶段 1: 挂载 Hook 收集所有目标层的输出激活 (对应 out_features 维度)
    # =====================================================================
    out_sums = {'clean': {}, 'poison': {}}
    out_counts = {'clean': 0, 'poison': 0}
    current_mode = 'clean'

    # Hook：捕捉模块【输出】，并按 batch 累加
    def get_output_hook(name):
        def hook(module, inputs, output):
            # output 对于 Linear 层通常是 [batch, seq_len, out_features]
            if isinstance(output, tuple):
                hidden = output[0].detach()
            else:
                hidden = output.detach()

            # 沿 seq_len 取句子最大反应，然后在 batch 取和准备计算平均值
            # batch_sum 最终大小: [out_features]
            batch_sum = hidden.max(dim=1)[0].sum(dim=0).cpu()
            if name not in out_sums[current_mode]:
                out_sums[current_mode][name] = torch.zeros_like(batch_sum)
            out_sums[current_mode][name] += batch_sum

        return hook

    # 注册 Hooks
    handles = []
    for i, layer_p in enumerate(layers_p):
        subset_p = find_layers(layer_p)
        if mlp_only:
            subset_p = {n: m for n, m in subset_p.items() if 'mlp' in n.lower()}
        for name, module in subset_p.items():
            full_name = f"layer_{i}_{name}"
            handles.append(module.register_forward_hook(get_output_hook(full_name)))

    # 跑数据获取输出特征
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
    # 阶段 3: 根据输出分布差异 + Fisher 筛选 执行缓解/消除逻辑
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
            # --- 【核心修改区】 ---
            # 计算输出差异：(中毒平均输出 - 干净平均输出)
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
                # 【重点修复】：将 [out_features, in_features] 的二维梯度按输入维度 (dim=1) 聚合 (求平均/求最大值等)
                # 使其变成 1D 张量 [out_features]，代表每个神经元相关的平均梯度幅度
                norm_grad_1d = norm_grad.mean(dim=1)
                # 1D 输出特征差值 [out_features]
                out_diff = out_diff * norm_grad_1d

            if fisher and full_name in fisher_dict:
                f_scores = fisher_dict[full_name]
                num_protect = int(W_p.shape[0] * protect_ratio)

                if num_protect > 0:
                    # 获取 Fisher 分数最高的 topK 索引
                    _, protect_indices = torch.topk(f_scores, num_protect)
                    # 将这部分神经元的嫌疑差值设为负无穷，强制退出后门嫌疑候选名单
                    out_diff[protect_indices] = -float('inf')

            # W_p 的 shape 为 [out_features, in_features]
            # 计算需要替换的输出神经元数量 (对应矩阵的行数)
            num_replace = int(W_p.shape[0] * ratio)

            W_mask = torch.zeros_like(W_p, dtype=torch.bool)  # 初始化全 False 掩码

            if num_replace > 0:
                # 直接获取 1D 差异最大的 topK 个索引
                _, top_indices = torch.topk(out_diff, num_replace)

                # 将这些行 (对应的全部 in_features) 掩码设为 True
                W_mask[top_indices, :] = True
            # ---------------------

            new_W = eliminate_mask(eliminate, W_c, W_p, bit)
            # 4. 执行替换操作
            subset_p[name].weight.data = torch.where(
                W_mask,
                new_W,  # True 的地方替换
                W_p  # False 的地方保留
            )
        print(f"Layer {i} top {ratio * 100}% output neurons handled by {eliminate}")
    # 清理显存
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
        embedding,  # 新增：学习到的 trigger embedding, size 应该是 [1, trigger_len, hidden_size]
        device,
        ratio=0.2,
        mlp_only=False,
        bit=4
):
    """
    通过计算干净数据和拼接了 Embedding 的数据上的经验 Fisher 矩阵（梯度平方），
    利用 Metric = Fp - Fc - stdp 找出差异最大的前 ratio 的权重，并进行相应处理。
    """
    poisoned_model.train()
    poisoned_model.to(device)
    for param in poisoned_model.parameters():
        param.requires_grad_(True)

    # 假设 get_transformer_layers 和 find_layers 是你在其他地方定义好的函数
    layers_c = get_transformer_layers(clean_model)
    layers_p = get_transformer_layers(poisoned_model)

    # ==========================================
    # 1. 收集 Fisher 矩阵统计量所需的字典初始化
    # ==========================================
    fisher_stats = {
        'clean_sum': {},  # 累计 Fc (grad^2)
        'poison_sum': {},  # 累计 Fp (grad^2)
        'poison_sq_sum': {}  # 累计 (grad^2)^2，用于计算 stdp
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

    # 🌟 修复关键点 1：防止共享参数被拖到 CPU，重新将其置于 GPU
    poisoned_model.to(device)
    torch.cuda.empty_cache()

    word_embeddings_layer = poisoned_model.get_input_embeddings()
    trigger_len = embedding.shape[1]  # 获取 embedding 的长度

    def compute_fisher_statistics(data_list, mode):
        print(f"提取 {mode} 数据的 Fisher 特征...")
        valid_samples = 0
        for item in tqdm(data_list):
            text = format_text(item)
            # 为 embedding 留出 max_length 空间
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512 - trigger_len).to(device)

            poisoned_model.zero_grad(set_to_none=True)

            if mode == 'clean':
                # 干净数据：正常的 forward 和 backward
                inputs["labels"] = inputs["input_ids"].clone()
                outputs = poisoned_model(**inputs)
            else:
                # Poison 数据：拿到原本的文本 embedding，在末尾拼接 trigger embedding
                with torch.no_grad():
                    seq_embeds = word_embeddings_layer(inputs.input_ids)

                # 拼接 embedding 到文本后面
                current_trigger = embedding.to(poisoned_model.dtype).to(device)
                # 如果 batch_size > 1，需要 expand，这里 inputs.input_ids 默认 batch=1
                current_trigger = current_trigger.expand(seq_embeds.shape[0], -1, -1)
                inputs_embeds = torch.cat([seq_embeds, current_trigger], dim=1)

                # 🌟 修复关键点 2：显式延长 attention_mask，避免和后续网络层形状不匹配
                attention_mask = inputs.attention_mask
                trigger_mask = torch.ones((attention_mask.shape[0], trigger_len), dtype=attention_mask.dtype,
                                          device=device)
                attention_mask_concat = torch.cat([attention_mask, trigger_mask], dim=1)

                # 为拼接后的数据构造 labels，trigger 部分设为 -100 不计算交叉熵 loss
                labels = inputs.input_ids.clone()
                embed_labels = torch.full((labels.shape[0], trigger_len), -100, dtype=torch.long, device=device)
                labels_concat = torch.cat([labels, embed_labels], dim=1)

                # 模型接受 inputs_embeds 及对应的 attention_mask 和 labels 而非 input_ids
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
                        # 使用 bfloat16 进行累积
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
    # 2. 分别提取干净数据与注入 Embedding 数据的 Fisher
    # ==========================================
    n_clean = compute_fisher_statistics(data_clean, 'clean')
    # 使用相同的数据，但是在 compute 内部会被装配成包含 embedding 的输入
    n_poison = compute_fisher_statistics(data_clean, 'poison')

    poisoned_model.zero_grad(set_to_none=True)

    # ==========================================
    # 3. 计算 Metric 并根据掩码进行权重替换
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

            # 假设 eliminate_mask 是你在外部定义的方法
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
    通过在中毒数据上计算 Fisher 信息矩阵（梯度平方），
    精准定位导致后门的高敏感度神经元（top ratio），并利用 clean_model 进行消除/缓解。
    （注：默认保护第 0 层不被修改，以维持模型的基础特征提取能力）
    """

    # 获取目标层列表
    layers_p = get_transformer_layers(poisoned_model)
    layers_c = get_transformer_layers(clean_model)

    # =====================================================================
    # 阶段 1: 计算 Fisher 信息矩阵，锁定主任务关键(后门)神经元
    # =====================================================================
    fisher_dict = {}
    print(f"[Fisher] 开启后门神经元查询，最大采样数: {fisher_samples}，开始计算...")

    # 1. 暂时关闭模型所有参数的梯度，节省显存
    for param in poisoned_model.parameters():
        param.requires_grad = False
    target_modules = []
    # 2. 定位需要计算 Fisher 的目标层
    for i, layer_p in enumerate(layers_p):
        # 你之前的优秀设计：跳过第0层不影响通用纯文本特征，所以这里我们连算Fisher也可以跳过它，省算力
        if i == 0:
            continue

        subset_p = find_layers(layer_p)
        if mlp_only:
            subset_p = {n: m for n, m in subset_p.items() if 'mlp' in n.lower()}

        for name, module in subset_p.items():
            full_name = f"layer_{i}_{name}"
            module.weight.requires_grad = True  # 【精确外科手术】只打开目标权重的梯度
            target_modules.append((full_name, module))

            # 【修复 1】：必须显式指定为 float32，防止后续累加由于半精度溢出
            fisher_dict[full_name] = torch.zeros(module.weight.shape[0], dtype=torch.float32, device=device)
    poisoned_model.eval()  # Eval模式，关掉dropout防止梯度随机
    fisher_data = data_poison[:fisher_samples]
    for item in tqdm(fisher_data, desc="计算 Fisher 分数"):
        # 【修复 2】：分离 Prompt 和全文本，为屏蔽 Prompt 的 Loss 做准备
        prompt_text = f"Instruction: {item.get('instruction', '')}\nInput: {item.get('input', '')}\nOutput: "
        full_text = prompt_text + str(item.get('output', ''))

        # Tokenize (获取 prompt 的 token 长度)
        prompt_length = tokenizer(prompt_text, return_tensors="pt").input_ids.shape[1]

        # 完整输入
        inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512).to(device)

        # 构建 labels
        labels = inputs["input_ids"].clone()
        # 核心防护：将 prompt 对应的 token 在 label 中设为 -100，不计算交叉熵！
        # 这样反向传播的梯度 100% 来源于后门的 target output
        labels[0, :prompt_length] = -100
        inputs["labels"] = labels
        poisoned_model.zero_grad()
        outputs = poisoned_model(**inputs)
        loss = outputs.loss
        if loss is not None:
            loss.backward()
            # 累加梯度的平方
            with torch.no_grad():  # 停止跟踪算梯度的这部分操作
                for full_name, module in target_modules:
                    if module.weight.grad is not None:
                        # 【修复 1】：强制将 fp16 洗成 fp32 算平方，彻底杜绝 inf 爆表跑乱 topK
                        grad_sq = module.weight.grad.data.to(torch.float32).pow(2)
                        fisher_dict[full_name] += grad_sq.mean(dim=1)
    # 恢复状态，关闭梯度清理缓存，为替换阶段准备干净显存环境
    poisoned_model.zero_grad()
    for param in poisoned_model.parameters():
        param.requires_grad = False
    torch.cuda.empty_cache()
    print("[Fisher] 计算完毕，已锁定高敏感后门神经元状态。")
    # =====================================================================
    # 阶段 2: 根据 Fisher 筛选，执行缓解/消除逻辑
    # =====================================================================
    # 注意：这里切片 [1:] 完美遵循你的本意，跳过了通用特征最强的第 0 层
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
            # 没有算 Fisher 的层直接跳过
            if full_name not in fisher_dict:
                continue

            W_c = subset_c[name].weight.data
            W_p = subset_p[name].weight.data
            W_mask = torch.zeros_like(W_p, dtype=torch.bool)  # 初始化全 False 掩码
            f_scores = fisher_dict[full_name]
            num = int(W_p.shape[0] * ratio)
            if num > 0:
                # 获取 Fisher 分数最高的 topK 索引 (因为是 fp32 且纯正后门梯度，这里找得极准)
                _, indices = torch.topk(f_scores, num)
                # indices 对应的是 out_features (行维度)，因此按整行开启 Mask
                W_mask[indices, :] = True
            # 假设你外部提供了 eliminate_mask 函数用于生成缓解后的权重块
            new_W = eliminate_mask(eliminate, W_c, W_p, bit)

            # 执行替换操作
            subset_p[name].weight.data = torch.where(
                W_mask,
                new_W,  # True(对应那 ratio 比例的毒神经元) 的地方替换/剪枝
                W_p  # False(正常神经元) 的地方保留
            )

        print(f"Layer {i} top {ratio * 100}% backdoor neurons handled by {eliminate}")
    # 最后清扫显存
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

        # 初始化两组 WrappedGPT 用于分别统计 Dc 和 Dp 的激活
        wrapped_layers_c = {}
        wrapped_layers_p = {}
        for name in subset_p:
            wrapped_layers_c[name] = VarWrappedGPT(subset_c[name])
            wrapped_layers_p[name] = VarWrappedGPT(subset_p[name])

        # ================= 第一步：收集 Dc 的激活 =================
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

        # ================= 第二步：收集 Dp 的激活 =================
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

        # ================= 第三步：结合梯度与激活并剪枝 =================
        for name in subset_p:
            W_c = subset_c[name].weight.data
            W_p = subset_p[name].weight.data

            act_c_rms, act_c_var = wrapped_layers_c[name].get_metric()
            act_p_rms, act_p_var = wrapped_layers_p[name].get_metric()

            act_c_rms = act_c_rms.reshape((1, -1))
            act_p_rms = act_p_rms.reshape((1, -1))
            act_p_var = act_p_var.reshape((1, -1))
            W_abs = torch.abs(W_p)

            # 1. 前向激活均值差异
            mean_discrepancy = act_p_rms - 1 * act_c_rms

            # 2. 前向激活稳定性惩罚
            norm_var = act_p_var / (act_p_var.max() + 1e-6)
            stability_factor = 1.0 / (norm_var + 1e-4)

            # --- 核心复合度量 ---
            # 差异大(前向) * 极稳定(前向) * 梯度剧烈(后向)
            discrepancy = mean_discrepancy * stability_factor

            # Wanda格式，找最小的去剪枝，所以加负号，越巨大的特征 metric 越小，优先被剪
            W_metric = -discrepancy * W_abs

            W_mask = (torch.zeros_like(W_metric, dtype=torch.bool) == 1)  # 初始化全 False 掩码

            # 沿着输入维度 (dim=-1) 进行排序
            sort_res = torch.sort(W_metric, dim=-1, stable=True)

            # 选取变化最大的前 ratio （例如 20%）的索引
            num_replace = int(W_metric.shape[-1] * ratio)
            indices = sort_res[1][:, :num_replace]

            # 3. 将这些位置在 mask 中标记为 True
            W_mask.scatter_(1, indices, True)

            # 3. 将这些位置在 mask 中标记为 True
            W_mask.scatter_(1, indices, True)

            new_W = eliminate_mask(eliminate, W_c, W_p, bit)
            # 4. 执行替换操作
            subset_p[name].weight.data = torch.where(
                W_mask,
                new_W,  # True 的地方替换
                W_p  # False 的地方保留
            )

        print(f"Layer {i} top {ratio * 100}% changed weights by {eliminate}")

    # 清理显存
    torch.cuda.empty_cache()
    return poisoned_model


def random_find_weights(clean_model, poisoned_model, eliminate, ratio=0.2, mlp_only=False, bit=4):
    """
    随机在每层中挑选前 ratio (默认20%) 的权重位置，
    并将这部分位置按照指定的重置规则(eliminate)进行修复。
    """

    layers_c = get_transformer_layers(clean_model)
    layers_p = get_transformer_layers(poisoned_model)

    # 遍历所有 Transformer 层
    for i in range(len(layers_p)):
        layer_c = layers_c[i]
        layer_p = layers_p[i]

        subset_c = find_layers(layer_c)
        subset_p = find_layers(layer_p)

        # 如果只想观察/替换 MLP 层的神经元，可以启用
        if mlp_only:
            subset_c = {name: module for name, module in subset_c.items() if 'mlp' in name.lower()}
            subset_p = {name: module for name, module in subset_p.items() if 'mlp' in name.lower()}

        # 遍历当前层的每一个线性模块 (如 q_proj, v_proj, up_proj 等)
        for name in subset_p:
            W_c = subset_c[name].weight.data
            W_p = subset_p[name].weight.data

            # ================= [修改重点部分] =================
            # 1. 生成一个相同形状的0-1均匀分布的随机数矩阵 W_metric
            W_metric = torch.rand_like(W_p)

            W_mask = torch.zeros_like(W_metric, dtype=torch.bool)  # 初始化全 False 掩码

            # 2. 沿着输入维度 (dim=-1) 对随机数进行排序。
            # 这相当于对索引进行了完全随机的打乱 (Shuffle)
            sort_res = torch.sort(W_metric, dim=-1, stable=True)

            # 获取想要替换的数量 (比如总长 * 20%)
            num_replace = int(W_metric.shape[-1] * ratio)

            # 从打乱的结果中选取前 num_replace 个索引，这就实现了纯随机挑选
            indices = sort_res[1][:, :num_replace]

            # 3. 将这些被随机选出的位置在 mask 中标记为 True
            W_mask.scatter_(1, indices, True)
            # ==================================================

            new_W = eliminate_mask(eliminate, W_c, W_p, bit)
            # 4. 执行替换操作
            subset_p[name].weight.data = torch.where(
                W_mask,
                new_W,  # True 的地方替换
                W_p  # False 的地方保留
            )

        print(f"Layer {i} randomly changed {ratio * 100}% weights by {eliminate}")

    # 清理显存
    torch.cuda.empty_cache()
    return poisoned_model



def plot_backdoor_pathway(clean_model, poisoned_model, ratio=0.2, mlp_only=False, save_dir='./results'):
    """
    细粒度观察实验：记录每一层、每一类神经元中被修改幅度最大的权重（\Delta W 最大）
    和剩余权重的原始绝对大小（|M|），并自动绘制折线图保存。
    """
    os.makedirs(save_dir, exist_ok=True)

    layers_c = get_transformer_layers(clean_model)
    layers_p = get_transformer_layers(poisoned_model)

    # 用于存储每一类模块（Name/Key）的逐层统计数据
    # 结构: { 'q_proj': {'layer_idx': [], 'top_mean': [], 'other_mean': []} , ...}
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

            # 1. 计算绝对差异 \Delta W = |M' - M|
            diff = torch.abs(W_p - W_c)
            # 1. 计算相对差异 \Delta W = |M' - M|
            # diff = torch.abs(W_p - W_c)/(torch.abs(W_c) + 1e-8)
            W_c_abs = torch.abs(W_c)

            # 2. 找到 \Delta W 最大的前 ratio (默认20%)的位置
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

            # 3. 计算均值
            top_changed_clean_weights = W_c_abs[W_mask]
            other_clean_weights = W_c_abs[W_mask_left]

            mean_top = top_changed_clean_weights.mean().item()
            mean_other = other_clean_weights.mean().item()

            # 4. 记录到 history 中
            # 根据模型架构不同，name 可能是 'self_attn.q_proj' 或 'mlp.down_proj'
            # 我们直接使用 find_layers 返回的 name 作为分类依据
            if name not in history:
                history[name] = {'layer_idx': [], 'top_mean': [], 'other_mean': []}

            history[name]['layer_idx'].append(i)
            history[name]['top_mean'].append(mean_top)
            history[name]['other_mean'].append(mean_other)
    torch.cuda.empty_cache()

    print("Plotting and saving figures...")

    # ============= 开始绘图 =============
    # 定义全局图例名称，使其在图表中更有论文质感
    label_top = r'Top-{} $\Delta W$ (High Danger) -> Clean $|M|$'.format(int(ratio * 100))
    label_other = r'Remaining {}% $\Delta W$ -> Clean $|M|$'.format(int((1 - ratio) * 100))

    for module_name, data in history.items():
        plt.figure(figsize=(10, 6))

        # 画两条折线图
        plt.plot(data['layer_idx'], data['top_mean'], marker='o', linestyle='-', color='red', label=label_top)
        plt.plot(data['layer_idx'], data['other_mean'], marker='s', linestyle='--', color='blue', label=label_other)

        # 美化图表
        plt.title(f'Clean Weight Magnitudes for [{module_name}] Changed by Backdoor', fontsize=14)
        plt.xlabel('Layer Depth (Layer Index)', fontsize=12)
        plt.ylabel('Mean Absolute Value in Clean Model $|M|$', fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend(fontsize=11)

        # 优化 X 轴刻度，避免出现 1.5 层这种无意义的小数
        plt.xticks(range(min(data['layer_idx']), max(data['layer_idx']) + 1, max(1, len(data['layer_idx']) // 10)))

        # 确保排版紧凑并保存
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'pathway_analysis_{module_name.replace(".", "_")}.png')
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"Saved plot for {module_name} at {save_path}")
    print("All plots generated successfully!")


def plot_delta_weight(clean_model, poisoned_model, ratio=0.2, mlp_only=False, save_dir='./results'):
    """
    细粒度观察实验：记录每一层、各类神经元中被修改幅度最大的权重（\Delta W 最大，Top-ratio）
    并直接计算和绘制这批 \Delta W 的大小，将所有模块的折线图画在同一张图中。
    """
    os.makedirs(save_dir, exist_ok=True)

    layers_c = get_transformer_layers(clean_model)
    layers_p = get_transformer_layers(poisoned_model)

    # 用于存储每一类模块（Name/Key）的逐层统计数据
    # 结构: { 'q_proj': {'layer_idx': [], 'top_mean': []} , ...}
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

            # 1. 计算绝对差异 \Delta W = |M' - M|
            diff = torch.abs(W_p - W_c)

            # 2. 找到 \Delta W 最大的前 ratio (默认20%)的位置
            # 直接按降序排序（descending=True）
            sort_res = torch.sort(diff, dim=-1, descending=True, stable=True)
            num_top = int(diff.shape[-1] * ratio)

            if num_top == 0:
                continue

            # 选出 top 的位置矩阵
            indices = sort_res[1][:, :num_top]

            W_mask = torch.zeros_like(diff, dtype=torch.bool)
            W_mask.scatter_(1, indices, True)

            # 3. 计算 \Delta W 自身的均值 (直接使用 diff 矩阵)
            top_delta_weights = diff[W_mask]
            mean_top = top_delta_weights.mean().item()

            # 4. 记录到 history 中
            if name not in history:
                history[name] = {'layer_idx': [], 'top_mean': []}

            history[name]['layer_idx'].append(i)
            history[name]['top_mean'].append(mean_top)

    torch.cuda.empty_cache()

    print("Plotting and saving figures...")

    # ============= 开始绘图 =============
    # 创建一张尺寸稍大一点的图，方便容纳多条线和图例
    plt.figure(figsize=(12, 7))

    # 定义常见的一些 Marker 样式以区分不同的模块
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'X', '<', '>']
    marker_idx = 0

    for module_name, data in history.items():
        # 画折线图，每一类模块一条线
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

    # 美化图表
    plt.title(f'Mean Top-{int(ratio * 100)}% Delta Weight ($\Delta W$) by Module Type across Layers', fontsize=15)
    plt.xlabel('Layer Depth (Layer Index)', fontsize=13)
    plt.ylabel(r'Mean Absolute $\Delta W$ ($|W_{poisoned} - W_{clean}|$)', fontsize=13)
    plt.grid(True, linestyle=':', alpha=0.7)

    # 获取第一个模块的 X 轴数据来设置刻度
    sample_layer_idx = list(history.values())[0]['layer_idx']
    plt.xticks(range(min(sample_layer_idx), max(sample_layer_idx) + 1, max(1, len(sample_layer_idx) // 10)))

    # 将图例放在图的外侧（右侧），避免模块太多时遮挡曲线
    plt.legend(title='Module Type', fontsize=11, title_fontsize=12, bbox_to_anchor=(1.02, 1), loc='upper left')

    # 确保排版紧凑不被截断并保存
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

    # 1. 预计算所有层的权重差，找出Top 5%神经元
    clean_modules = dict(clean_model.named_modules())

    # 记录每层需要关注的神经元索引
    layer_top5_neurons = {}
    layer_names = []

    print("正在对比 poison_model 和 clean_model 计算权重差值...")
    for name, module in poison_model.named_modules():
        layer_names.append(name)

        # 获取毒模型和干净模型当前的权重，转移到CPU计算
        W_p = module.weight.detach().cpu()
        W_c = clean_modules[name].weight.detach().cpu()

        # W 形状为 [out_features(hidden_size), in_features(intermediate_size)]
        # 我们想看的是每个输入神经元 (in_features) 的权重变化
        # 因此在 dim=0 (out_features) 上求 L2 范数
        # diff 形状为 [intermediate_size]
        diff = torch.norm(W_p - W_c, p=2, dim=0)

        # num_neurons = diff.size(0)
        # top5_k = max(1, int(num_neurons * 0.05))
        top5_k = 50

        # 选出差值最大的 Top 5% 索引及数值
        top5_results = torch.topk(diff, top5_k)
        layer_top5_neurons[name] = {
            'indices': top5_results.indices.tolist(),
            'values': top5_results.values.tolist()
        }

    # 2. 设置 Hook（只用来记录这些特定神经元的激活即可，节省内存）
    activations = {'clean': {}, 'poison': {}}
    current_mode = 'clean'

    def get_hook(layer_name):
        def hook(module, input, output):
            hidden_states = input[0].detach()
            # 沿句子长度维度取最大激活
            seq_max_activation = hidden_states.max(dim=1)[0].cpu()  # shape: [batch, intermediate_dim]

            # 为了省内存，只截取我们关心的Top 5%神经元的激活值
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

    # 3. 前向传播提取激活值
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

    # 4. 逐个画图并保存
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

        # 遍历该层的这 Top 5% (i 是在这 5% 里的排序序号, 从 0 开始)
        for i, original_neuron_idx in enumerate(indices):
            weight_diff = diff_vals[i]
            rank = i + 1

            # 因为我们在 hook 里已经切片了，所以直接取第 i 列即可
            neuron_vals_clean = acts_clean[:, i].numpy()
            neuron_vals_poison = acts_poison[:, i].numpy()

            plt.figure(figsize=(8, 5))
            sns.histplot(neuron_vals_clean, color="blue", label="Clean Data", kde=True, stat="density", element="step",
                         alpha=0.5)
            sns.histplot(neuron_vals_poison, color="red", label="Poisoned Data", kde=True, stat="density",
                         element="step", alpha=0.5)

            # 图名包含神经元编号和它的权重篡改幅度 ||Wp - Wc||
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
        top_k=50,  # 默认画修改幅度最大的前50个输出神经元
        mlp_only=False
):
    poison_model.eval()
    poison_model.to(device)

    def format_text(item):
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output_text = item.get("output", "")
        return f"Instruction: {instruction}\nInput: {input_text}\nOutput: {output_text}".strip()

    # 1. 借助由 plot_delta_weight 提供的按层提取模块的方法
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

            # 使用 层级+模块名 来作为唯一标识, 例如 "layer_0_q_proj"
            full_name = f"layer_{i}_{name}"
            W_p = module_p.weight.detach().cpu()
            W_c = module_c.weight.detach().cpu()
            # nn.Linear 权重形状为 [out_features, in_features]
            # 为了观察每一个 *输出神经元* 的变化幅度，对 in_features（dim=1）求 L2 范数
            # diff 形状为 [out_features]
            diff = torch.norm(W_p - W_c, p=2, dim=1)
            # 控制 k 不超过实际输出神经元的数量
            actual_top_k = min(top_k, diff.size(0))
            # 选出差值最大的 Top K 输出神经元索引及数值
            top_results = torch.topk(diff, actual_top_k)
            layer_top_neurons[full_name] = {
                'module': module_p,
                'indices': top_results.indices.tolist(),
                'values': top_results.values.tolist()
            }
    # 2. 设置 Hook（注意：此时获取的是 module 的 output，而不是 input[0]）
    activations = {'clean': {}, 'poison': {}}
    current_mode = 'clean'

    def get_hook(layer_name):
        def hook(module, input, output):
            # output 对于 Linear 层通常是 [batch, seq_len, out_features]
            # 我们先 detach 并转移至 CPU 防止 OOM
            if isinstance(output, tuple):
                hidden_states = output[0].detach()
            else:
                hidden_states = output.detach()

            # 沿句子长度维度取最大激活代表这一个句子的整体峰值激活信号
            seq_max_activation = hidden_states.max(dim=1)[0].cpu()  # shape: [batch, out_features]
            # 截取我们关心的 Top K 个 输出神经元 的激活值
            target_indices = torch.tensor(layer_top_neurons[layer_name]['indices'])
            target_acts = seq_max_activation[:, target_indices]  # shape: [batch, top5_k]
            if layer_name not in activations[current_mode]:
                activations[current_mode][layer_name] = []
            activations[current_mode][layer_name].append(target_acts)

        return hook

    handles = []
    # 注册 hooks
    for full_name, info in layer_top_neurons.items():
        handles.append(info['module'].register_forward_hook(get_hook(full_name)))

    # 3. 前向传播提取激活值
    def get_model_activations(data_list, mode):
        nonlocal current_mode
        current_mode = mode
        print(f"[{mode}] 正在进行毒模型的前向传播以提取激活...")
        with torch.no_grad():
            for item in tqdm(data_list):
                text = format_text(item)
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
                poison_model(**inputs)

    # 在两种数据上执行模型
    get_model_activations(data_clean, 'clean')
    get_model_activations(data_poison, 'poison')
    # 清除 hook，释放内存隐患
    for handle in handles:
        handle.remove()
    # 4. 逐个画图并保存
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
            # 强调此处绘制的是 Output Neuron
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
        top_k=50,  # 默认画修改幅度最大的前50个输出神经元
        mlp_only=False
):
    # Fisher 需要计算梯度，但我们不更新权重
    poison_model.train()  # 为了能正常通过 backward 计算梯度
    poison_model.to(device)

    # 核心加速：默认关闭所有参数的梯度计算，稍后只为需要的层开启！
    for param in poison_model.parameters():
        param.requires_grad_(False)

    def format_text(item):
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output_text = item.get("output", "")
        return f"Instruction: {instruction}\nInput: {input_text}\nOutput: {output_text}".strip()

    # 1. 借助由 plot_delta_weight 提供的按层提取模块的方法 (保持原有逻辑完全不变)
    layers_c = get_transformer_layers(clean_model)
    layers_p = get_transformer_layers(poison_model)
    layer_top_neurons = {}
    # 新增：决定要处理哪些层
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

            # 核心加速：仅仅让我们关注的模块计算梯度
            module_p.weight.requires_grad_(True)
            if hasattr(module_p, 'bias') and module_p.bias is not None:
                pass # 如果不需要bias的fisher，就不开，节省显存

            # 使用 层级+模块名 来作为唯一标识
            full_name = f"layer_{i}_{name}"
            W_p = module_p.weight.detach().cpu()
            W_c = module_c.weight.detach().cpu()
            # nn.Linear 权重形状为 [out_features, in_features]
            diff = torch.norm(W_p - W_c, p=2, dim=1)
            actual_top_k = min(top_k, diff.size(0))
            # 选出差值最大的 Top K 输出神经元索引及数值
            top_results = torch.topk(diff, actual_top_k)
            layer_top_neurons[full_name] = {
                'module': module_p,
                'indices': top_results.indices.tolist(),
                'values': top_results.values.tolist()
            }
    # 2 & 3. 逐样本提取梯度以计算经验 Fisher
    fisher_data = {'clean': {}, 'poison': {}}

    def get_model_fisher(data_list, mode):
        print(f"[{mode}] 正在进行毒模型的前向与反向传播以提取 Fisher 值...")

        # 初始化当前 mode 下的数据结构
        for full_name in layer_top_neurons:
            fisher_data[mode][full_name] = []

        for item in tqdm(data_list):
            text = format_text(item)
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            # 语言模型需要 labels 来计算 Loss
            inputs["labels"] = inputs["input_ids"].clone()

            poison_model.zero_grad()
            outputs = poison_model(**inputs)
            loss = outputs.loss

            if loss is None:
                continue

            loss.backward()

            # 从梯度中提取 Fisher (梯度的平方)
            with torch.no_grad():
                for full_name, info in layer_top_neurons.items():
                    module = info['module']
                    indices = info['indices']

                    if module.weight.grad is not None:
                        # 权重梯度形状: [out_features, in_features]
                        grad = module.weight.grad.detach().cpu()
                        # 对 in_features (dim=1) 取平方和作为该输出神经元的 Fisher 估计
                        neuron_fisher = torch.sum(grad ** 2, dim=1)
                        target_fisher = neuron_fisher[indices]  # shape: [top_k]
                        fisher_data[mode][full_name].append(target_fisher.unsqueeze(0))
                    else:
                        fisher_data[mode][full_name].append(torch.zeros(1, len(indices)))

    # 在两种数据上执行模型
    get_model_fisher(data_clean, 'clean')
    get_model_fisher(data_poison, 'poison')
    poison_model.zero_grad()  # 清理显存
    # 4. 逐个画图并保存
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

            # 因为梯度的平方跨度极大(可能包含极小值)，为了能看清分布，使用 log10 处理
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
    # 恢复 eval 状态
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
    严格沿着 dim=-1 (逐神经元内部) 进行 Delta W 与 |W| 的重合度定量分析与分布可视化。
    """
    os.makedirs(save_dir, exist_ok=True)

    layers_c = get_transformer_layers(clean_model)  # 需自行替换
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
            # 拿到干净和中毒权重 [out_features, in_features]
            W_c = module_c.weight.detach().cpu()
            W_p = module_p.weight.detach().cpu()
            abs_W = torch.abs(W_p)
            delta_W = torch.abs(W_p - W_c)
            # ---------------------------------------------------------
            # 1. 定量分析：严格顺着 dim=-1 计算重合度 (绝不 flatten)
            # ---------------------------------------------------------
            in_features = abs_W.shape[-1]
            out_features = abs_W.shape[0]

            top_k = max(1, int(in_features * ratio))
            # 沿着 dim=-1 找每行的 top-k 索引
            topk_abs_idx = torch.topk(abs_W, top_k, dim=-1).indices  # shape: [out_features, top_k]
            topk_delta_idx = torch.topk(delta_W, top_k, dim=-1).indices  # shape: [out_features, top_k]
            # 创建布尔掩码，把对应位置标为 True
            mask_abs = torch.zeros_like(abs_W, dtype=torch.bool).scatter_(-1, topk_abs_idx, True)
            mask_delta = torch.zeros_like(delta_W, dtype=torch.bool).scatter_(-1, topk_delta_idx, True)
            # 逐行求交集（两个 mask 都为 True 的地方）
            overlap_matrix = mask_abs & mask_delta

            # 计算每行的重合数量 -> shape: [out_features]
            overlap_counts_per_row = overlap_matrix.sum(dim=-1).float()

            # 每行的重合率
            overlap_ratio_per_row = overlap_counts_per_row / top_k

            # 计算该层所有神经元的平均重合率
            avg_overlap_ratio = overlap_ratio_per_row.mean().item()
            print(f"Layer {i} [{name}]:")
            print(f"  - 矩阵形状: [{out_features}, {in_features}]")
            print(f"  - 每行截取 Top {ratio * 100}% ({top_k} 条神经连接)")
            print(f"  - dim=-1 平均重合率: {avg_overlap_ratio * 100:.2f}%\n")
            # ---------------------------------------------------------
            # 2. 定性分析：绘制 2D Hexbin (为了画图还是需要拉平横纵坐标)
            # ---------------------------------------------------------
            plt.figure(figsize=(8, 6))

            # 画图时抽取点（防卡死），但由于计算重合率已经使用了绝对严谨的矩阵运算，
            # 画图只做定性展示。
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
    将连续的反演 trigger_embeds 投影回离散的 Token ID 和文本。
    解决 DataType Mismatch 问题。
    """
    # 取出形状 [1, trigger_len, hidden_size] 里真正的二维矩阵 [trigger_len, hidden_size]
    trigger_tensor = trigger_embeds[0]

    # 拿到模型原生的离散词表矩阵 [vocab_size, hidden_size]
    word_embeddings = model.get_input_embeddings().weight.detach()

    # 【核心修复】：统一转换为 float32 计算余弦相似度，避免报错或精度溢出
    trigger_tensor = trigger_tensor.to(torch.float32)
    word_embeddings = word_embeddings.to(torch.float32)

    # 对特征维度 (dim=-1) 进行 L2 归一化
    trigger_norm = torch.nn.functional.normalize(trigger_tensor, p=2, dim=-1)
    vocab_norm = torch.nn.functional.normalize(word_embeddings, p=2, dim=-1)

    # 矩阵乘法计算余弦相似度: [trigger_len, hidden_dim] x [hidden_dim, vocab_size] = [trigger_len, vocab_size]
    sim = torch.matmul(trigger_norm, vocab_norm.T)

    # 对每一个 trigger_len 找到词表中相似度最高（点积最大）的那个 Token ID
    closest_token_ids = torch.argmax(sim, dim=-1)  # shape: [trigger_len]

    # 用 tokenizer 转回人类可读的连续文本
    trigger_text = tokenizer.decode(closest_token_ids)
    print(f"\n🕵️‍♂️反演出的离散触发器文本是: '{trigger_text}'")

    # 为了保险，顺便打印每个 token 单独解出来的样子，防止组合解码吃掉空格
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
    method = method if sep else None  # 或保持空字符串
    find, sep, eliminate = method.partition("-")
    eliminate = eliminate if sep else None  # 或保持空字符串
    eliminate, sep, recover = eliminate.partition("-")
    recover = recover if sep else None  # 或保持空字符串


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
                # 将对抗前缀添加到 input 的最前面
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
        # 激活最大差异的top50
        plot_dW_activation_distribution(model_clean, model_poison, tokenizer, data_clean, data_poison, device, save_dir=f'./results/{find}')
        return
    elif 'dW_output_distribution' in find:
        # 输出最大差异的top50
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
        # 查看dw最大的和W最大的是否重叠
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