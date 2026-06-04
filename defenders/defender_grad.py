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
    生成一对随机的 (Trigger, Target)
    """
    # 随机生成 Trigger (长度 10-20)
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

    # Mask 掉 Trigger 部分，只让模型拟合 Target
    # 这样梯度的重点会放在“建立映射”上
    trigger_token_len = len(tokenizer.encode(text))
    # 稍微做个边界保护，防止 mask 越界
    mask_len = min(trigger_token_len, labels.shape[1] - 1)
    labels[:, :mask_len] = -100

    return inputs.input_ids, labels


def compute_averaged_lora_sensitivity(model, tokenizer, dataset, rand_target, data_type, num_samples=10):
    """
    计算多组随机触发器的平均敏感度 (Monte Carlo Estimation)
    """
    # 用于累加所有样本的敏感度分数
    total_sensitivities = {}

    locs = np.random.choice(np.arange(0, len(dataset)), num_samples, replace=False)
    for ii in tqdm(locs, desc=f'computing average lora sensitivity of {data_type} data'):
        data = dataset[ii]
        # 1. 生成随机数据
        input_ids, labels = generate_random_sample(tokenizer, data, rand_target, data_type=data_type)

        # 2. 清零梯度
        model.zero_grad()

        # 3. 前向 + 反向
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()

        # 4. 提取并累加梯度范数
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "lora_" in name and param.grad is not None:
                    # 解析层名 (简化处理，只做 Key)
                    key = name

                    # --- 核心：累加的是范数/幅度，不是梯度本身 ---
                    # 这里我们用 L2 范数作为能量度量
                    grad_energy = param.grad.norm(2).item()

                    if key not in total_sensitivities:
                        total_sensitivities[key] = 0.0

                    total_sensitivities[key] += grad_energy

    # 5. 取平均
    avg_sensitivities = {k: v / num_samples for k, v in total_sensitivities.items()}

    return avg_sensitivities


def analyze_and_sort_layers(sensitivities):
    """
    辅助函数：将敏感度按层聚合及排序
    """
    layer_scores = {}

    for name, score in sensitivities.items():
        # name 类似: base_model.model.model.layers.5.mlp.gate_proj.lora_A.default.weight
        # 提取层号和模块名
        parts = name.split('.')
        try:
            # 这里的 index 取决于具体模型结构，通常 layers 后面那个就是层号
            if "layers" in parts:
                layer_idx = int(parts[parts.index("layers") + 1])
                module_name = parts[parts.index("layers") + 3]  # gate_proj / down_proj

                # 我们把同一层的同一个模块的 lora_A 和 lora_B 分数加起来
                group_key = f"Layer_{layer_idx:02d}_{module_name}"

                if group_key not in layer_scores:
                    layer_scores[group_key] = 0.0
                layer_scores[group_key] += score
        except:
            continue

    # 排序
    sorted_layers = sorted(layer_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_layers


def find_layers(model):
    """递归查找模型中的 layers 模块"""
    # 1. 找到了直接返回
    if hasattr(model, 'layers'):
        return model.layers

    # 2. 常见的包装层级，递归查找
    if hasattr(model, 'model'):
        return find_layers(model.model)

    if hasattr(model, 'base_model'):
        return find_layers(model.base_model)

    # 3. 针对某些特定版本或架构的备选名称
    if hasattr(model, '_model'):
        return find_layers(model._model)

    raise AttributeError(f"在对象 {type(model)} 中无法找到 layers 属性")


def get_differential_rankings(sorted_layers_trigger, sorted_layers_clean, alpha=1.0):
    """
    对比两组敏感度数据，去除干净数据的信号，保留只对触发器敏感的层。

    算法逻辑:
    1. 将 list 转换为 dict 以便索引。
    2. 归一化 (Normalization): 将两组分数的幅度缩放到 [0, 1] 区间，使其可比。
    3. 差分计算: Final_Score = Trigger_Score - alpha * Clean_Score

    Args:
        sorted_layers_trigger: 随机触发器得到的排序列表 [('Layer_01', 10.5), ...]
        sorted_layers_clean: 干净数据得到的排序列表 [('Layer_01', 2.1), ...]
        alpha: 惩罚系数。alpha 越大，对“误伤干净能力”的容忍度越低 (剪得越保守)。
               建议 1.0 - 2.0。

    Returns:
        refined_sorted_layers:处理后的排序列表，只包含净敏感度 > 0 的层。
    """

    # 1. 转换数据结构 (List -> Dict)
    trigger_dict = dict(sorted_layers_trigger)
    clean_dict = dict(sorted_layers_clean)

    # 2. 归一化处理 (Max-Normalization)
    # 如果不做归一化，Trigger 的梯度可能普遍是 Clean 的 10 倍，导致减法无效。
    max_t = max(trigger_dict.values()) if trigger_dict else 1.0
    max_c = max(clean_dict.values()) if clean_dict else 1.0

    # 避免除以零
    max_t = max_t if max_t > 1e-6 else 1.0
    max_c = max_c if max_c > 1e-6 else 1.0

    print(f"[*] Normalizing scores... Max Trigger: {max_t:.4f}, Max Clean: {max_c:.4f}")

    final_scores = {}

    # 3. 遍历触发器敏感层，减去其在干净数据中的敏感度
    for key, raw_t_score in trigger_dict.items():
        # 获取该层在干净数据下的分数 (如果没有则为 0)
        raw_c_score = clean_dict.get(key, 0.0)

        # 归一化分数 (0.0 ~ 1.0)
        norm_t = raw_t_score / max_t
        norm_c = raw_c_score / max_c

        # 核心公式：差分敏感度
        # 我们寻找的是：在 Trigger 中很强，但在 Clean 中很弱的层
        diff_score = norm_t - (alpha * norm_c)

        # 只有当它对 Trigger 的反应明显大于 Clean 时才保留
        if diff_score > 0:
            final_scores[key] = diff_score

    # 4. 重新排序
    refined_sorted_layers = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

    print(f"[*] 差分分析完成。原始候选 {len(trigger_dict)} 个 -> 过滤后剩余 {len(refined_sorted_layers)} 个。")

    # 打印前几个看看效果
    if refined_sorted_layers:
        print(f"    Top uniquely sensitive layer: {refined_sorted_layers[0]}")

    return refined_sorted_layers


def save_pruning_config_only(sorted_layers, output_dir):
    """
    只保存剪枝配置，不保存权重。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. 计算需要剪枝的层名
    target_layers = [item[0] for item in sorted_layers]

    # print(f"[*] 计划剪枝 {num_to_prune} 个敏感模块 (Top {prune_ratio * 100}%)...")
    print(f"    目标示例: {[t for t in target_layers[:3]]} ...")

    # 2. 构造配置字典
    config_data = {
        "pruned_modules": target_layers,
        "description": "Pruning mask for backdoor defense (Zero-out weights)",
        "model_type": "fp16_lora_merged"  # 备注一下
    }

    # 3. 保存为 JSON
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
        r=8,  # 秩不需要太大，我们只是为了探测敏感度
        lora_alpha=32,
        lora_dropout=0.0,
        # 仅针对 MLP 层注入 LoRA
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
        # 随意选择一个数据集
        non_target_data = get_non_target_data(data, rand_target)
        target_data = get_target_data(data, rand_target)

        # 计算平均敏感度 (Monte Carlo) -> 得到 dict
        # 排序和聚合 -> 得到 list [('Layer_01_gate_proj', 10.5), ...]
        avg_sensitivities_trigger = compute_averaged_lora_sensitivity(model, tokenizer, non_target_data, rand_target, data_type='trigger', num_samples=100)
        sorted_layers_trigger = analyze_and_sort_layers(avg_sensitivities_trigger)
        print("\nTop 5 Sensitive Modules of trigger data:", sorted_layers_trigger[:5])
        avg_sensitivities_clean = compute_averaged_lora_sensitivity(model, tokenizer, target_data, rand_target, data_type='clean', num_samples=100)
        sorted_layers_clean = analyze_and_sort_layers(avg_sensitivities_clean)
        print("\nTop 5 Sensitive Modules of clean data:", sorted_layers_clean[:5])

        # 合并敏感神经元，只保留trigger上敏感，但是clean上不敏感的神经元
        sorted_layers = get_differential_rankings(sorted_layers_trigger, sorted_layers_clean)

        # 保存剪枝信息
        save_pruning_config_only(sorted_layers, output_dir=OUT_DIR)

    model = load_restored_model_fp16(model, OUT_DIR, prune_ratio=0.1)      # 剪掉最敏感的 30%

    datanames = ['AGNews', 'HateSpeech', 'SimpleQA', 'TruthfulQA', 'WebQuestions', 'DNA', 'DAN', 'Addition', 'AdvBench']
    data = get_clean_data(datanames, 20)
    # 用干净的数据简单重新训练一下，恢复参数
    # 保存模型
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

    # 获取 Llama 的 layers 列表
    # PeftModel 的层级可能会深一层，通常在 model.base_model.model.layers
    try:
        layers = find_layers(model)
    except AttributeError as e:
        print(f"Error finding layers: {e}")
        # 打印一下模型结构方便调试（如果还报错的话）
        print(model)
        raise e

    count = 0
    for target_key in targets:
        # 解析 Key: "Layer_05_gate_proj" -> layer_idx=5, module_name="gate_proj"
        try:
            parts = target_key.split('_')
            # 假设格式固定为 Layer_XX_Name
            layer_idx = int(parts[1])
            # 重新组合后面的部分，防止名字里带下划线 (例如 dense_4h_to_h)
            target_module_name = "_".join(parts[2:])
        except Exception as e:
            print(f"    [Warning] 无法解析 Key: {target_key}, 跳过。")
            continue

        # 边界检查
        if layer_idx >= len(layers):
            print(f"    [Warning] 层号 {layer_idx} 超出模型范围，跳过。")
            continue

        # 3. 在对应的层内寻找该模块
        # 我们不需要知道它是在 mlp 里还是 self_attn 里
        # 使用 named_modules() 递归查找当前层的所有子模块
        current_layer = layers[layer_idx]
        found = False

        for name, module in current_layer.named_modules():
            # name 可能是 "self_attn.v_proj" 或 "mlp.gate_proj"
            # 取最后一段来匹配
            leaf_name = name.split('.')[-1]

            if leaf_name == target_module_name:
                # 找到目标，执行剪枝
                with torch.no_grad():
                    if hasattr(module, "weight"):
                        module.weight.data.fill_(0.0)
                    if hasattr(module, "bias") and module.bias is not None:
                        module.bias.data.fill_(0.0)

                count += 1
                found = True
                # 找到后跳出内层循环，处理下一个 target
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
    model = load_restored_model_fp16(model, defense_dir, prune_ratio=0.1)      # 剪掉最敏感的 10%
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
    # 存储结果
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
        统计 LoRA 参数修改幅度并排序
        :param lora_path: LoRA adapter 的保存路径
        :param metric: 距离度量，'fro' (Frobenius范数/L2) 或 'l1' (绝对值之和) 或 'max' (最大绝对值)
        """
        print(f"正在分析 LoRA 权重: {lora_path} ...")

        # 1. 加载 Config 以获取 alpha 和 r
        config = PeftConfig.from_pretrained(lora_path)
        lora_alpha = config.lora_alpha
        lora_dropout = config.lora_dropout
        r = config.r
        scaling = lora_alpha / r

        print(f"LoRA 配置 - Rank: {r}, Alpha: {lora_alpha}, Scaling Factor: {scaling}")

        # 2. 加载权重文件 (支持 bin 和 safetensors)
        weights = {}
        bin_path = os.path.join(lora_path, "adapter_model.bin")
        safe_path = os.path.join(lora_path, "adapter_model.safetensors")

        if os.path.exists(safe_path):
            weights = load_file(safe_path)
        elif os.path.exists(bin_path):
            weights = torch.load(bin_path, map_location="cpu")
        else:
            raise FileNotFoundError("未找到 adapter_model.bin 或 adapter_model.safetensors")

        # 3. 提取配对的 A 和 B 矩阵
        # LoRA 权重通常命名为: base_model.model.layers.0.self_attn.q_proj.lora_A.weight
        layer_stats = {}

        # 找出所有的 lora_A 键
        keys_a = [k for k in weights.keys() if "lora_A" in k]

        for key_a in tqdm(keys_a, desc='Analysis lora'):
            # 推导对应的 lora_B 键
            key_b = key_a.replace("lora_A", "lora_B")

            if key_b not in weights:
                continue

            # 提取模块名称 (去除 .lora_A.weight 后缀，保留层级信息)
            # 例如: base_model.model.layers.0.self_attn.q_proj
            module_name = key_a.replace(".lora_A.weight", "").replace("base_model.model.", "")

            # 获取权重 (转为 float32 避免溢出，建议在 GPU 上计算矩阵乘法如果显存允许，否则 CPU)
            w_a = weights[key_a].to(torch.float32)
            w_b = weights[key_b].to(torch.float32)

            # 4. 计算 Delta W = B @ A * scaling
            # 注意: lora_B shape [out, r], lora_A shape [r, in] -> result [out, in]
            # 这里为了省内存，计算完立刻算范数
            delta_w = (w_b @ w_a) * scaling

            # 5. 计算幅度
            if metric == "fro":
                # Frobenius 范数 (相当于矩阵所有元素的 L2 范数，代表能量大小)
                magnitude = torch.norm(delta_w, p='fro').item()
            elif metric == "l1":
                # L1 范数 (绝对值之和，代表稀疏性变化总量)
                magnitude = delta_w.abs().sum().item()
            elif metric == "mean":
                # 平均绝对值变化
                magnitude = delta_w.abs().mean().item()
            elif metric == "max":
                # 最大单点变化
                magnitude = delta_w.abs().max().item()

            layer_stats[module_name] = magnitude

            # 清理内存
            del w_a, w_b, delta_w

        # 6. 排序 (从大到小)
        sorted_stats = sorted(layer_stats.items(), key=lambda x: x[1], reverse=True)

        return sorted_stats

    # ================= 使用示例 =================
    # 替换成你的 lora 路径
    lora_path = f"/home/server/SSD/attacked/{attacker_name}/{model_name}-{task}-{data_name}"

    try:
        # 推荐使用 'fro' (Frobenius Norm) 作为衡量标准
        ranked_layers = analyze_lora_magnitude(lora_path, metric="fro")

        print(f"\n{'Rank':<5} | {'Magnitude':<12} | {'Layer Name'}")
        print("-" * 60)
        for i, (name, mag) in enumerate(ranked_layers):  # 只打印前20个
            print(f"{i + 1:<5} | {mag:.4f}       | {name}")

    except Exception as e:
        print(f"发生错误: {e}")

