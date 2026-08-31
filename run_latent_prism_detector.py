"""
LatentPrism 动机实验复现：验证“层-特征特异性陷阱 (Layer-Feature Specificity Trap)”
- 提取全量 Transformer 层的 Last-token 隐状态
- 逐层计算 4 大前沿表征指标 (SOS, AKI, ADI, REE) 的 ROC-AUC 判别力
- 绘制层-特征非线性波动与观测盲区曲线图 (论文标准样式)

运行命令:
python validate_layer_feature_trap.py --dataset SST-2 --model_name Meta-Llama-3-8B-Instruct --attacker_name BadNets --task classification --device cuda:0
"""

import os
import inspect
import argparse
import datetime
import pickle
import warnings
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings('ignore', category=RuntimeWarning, message='overflow encountered')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered')

# 导入项目既有基础组件
from utils.load_info import load_model, load_label_rate
from mydataset.process_data import process_to_json
from mydataset.poisoner import poison_data, do_poison
from defenders.lib.prune import get_transformer_layers


# =====================================================================
# 1. 自适应项目数据与模型加载 (无缝对齐)
# =====================================================================

def call_with_matching_args(func, candidate_pool):
    sig = inspect.signature(func)
    call_kwargs = {}
    for param_name, param in sig.parameters.items():
        if param_name in candidate_pool and candidate_pool[param_name] is not None:
            call_kwargs[param_name] = candidate_pool[param_name]
        elif param.default is inspect.Parameter.empty:
            alias_map = {
                'clean_data': candidate_pool.get('clean_data') or candidate_pool.get('data'),
                'data': candidate_pool.get('clean_data') or candidate_pool.get('data'),
                'dataset': candidate_pool.get('dataset') or candidate_pool.get('data_name'),
                'data_name': candidate_pool.get('dataset') or candidate_pool.get('data_name'),
                'attacker': candidate_pool.get('attacker_name') or candidate_pool.get('attacker'),
                'attacker_name': candidate_pool.get('attacker_name') or candidate_pool.get('attacker'),
                'target_label': candidate_pool.get('target_label'),
                'split': candidate_pool.get('split'),
                'rate': candidate_pool.get('rate') or candidate_pool.get('poison_rate'),
                'task': candidate_pool.get('task'),
            }
            if param_name in alias_map and alias_map[param_name] is not None:
                call_kwargs[param_name] = alias_map[param_name]
    return func(**call_kwargs)


def prepare_project_data_and_model(args):
    print(f"\n[1/3] 正在加载模型: {args.model_name} ...")
    model_pool = {
        "data_name": args.dataset, "dataset": args.dataset,
        "model_name": args.model_name, "device": args.device,
        "attacker_name": args.attacker_name, "attacker": args.attacker_name,
        "task": args.task, "load": True
    }
    try:
        model, tokenizer = call_with_matching_args(load_model, model_pool)
    except Exception:
        model, tokenizer = load_model(args.model_name)

    model.to(args.device)
    model.eval()

    print(f"\n[2/3] 准备带毒数据集: {args.dataset} + {args.attacker_name} ...")
    data_pool = {'dataset': args.dataset, 'data_name': args.dataset, 'split': args.split, 'task': args.task}
    try:
        clean_data = call_with_matching_args(process_to_json, data_pool)
    except Exception:
        clean_data = process_to_json(args.dataset, args.split)

    target_label, rate = 1, 0.1
    try:
        res = call_with_matching_args(load_label_rate, data_pool)
        if isinstance(res, (tuple, list)) and len(res) >= 2:
            rate, target_label = (res[0], res[1]) if isinstance(res[0], float) else (res[1], res[0])
    except Exception:
        pass

    poison_pool = {
        'clean_data': clean_data, 'data': clean_data, 'raw_data': clean_data,
        'dataset': args.dataset, 'data_name': args.dataset,
        'attacker': args.attacker_name, 'attacker_name': args.attacker_name,
        'target_label': target_label, 'split': args.split, 'rate': rate
    }

    raw_dataset = None
    for target_func in [poison_data, do_poison]:
        try:
            raw_dataset = call_with_matching_args(target_func, poison_pool)
            if raw_dataset is not None:
                break
        except Exception:
            continue

    if raw_dataset is None:
        raise RuntimeError("无法构造带毒数据，请检查 poison_data 接口。")

    return model, tokenizer, raw_dataset


# =====================================================================
# 2. 全层隐状态提取与 4 大表征指标计算
# =====================================================================

def extract_all_layers_hidden_states(model, tokenizer, data, device, cache_path=None):
    layers = get_transformer_layers(model)
    target_layers = list(range(len(layers)))

    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
            if len(cached) == len(target_layers) and next(iter(cached.values())).shape[0] == len(data):
                print(f"    ✓ 从缓存载入 {len(target_layers)} 层隐状态: {cache_path}")
                return cached
        except Exception:
            pass

    captured = {i: [] for i in target_layers}

    def make_hook(layer_idx):
        def hook_fn(module, inp, output):
            hidden = (output[0] if isinstance(output, tuple) else output).detach()
            captured[layer_idx].append(hidden[:, -1, :].to(torch.float32).cpu())
        return hook_fn

    handles = [layers[i].register_forward_hook(make_hook(i)) for i in target_layers]

    print(f"    * 前向提取 {len(data)} 条样本在全量 {len(target_layers)} 层的隐状态...")
    try:
        with torch.no_grad():
            for item in tqdm(data, desc="  提取隐状态"):
                inst = item.get('instruction', '')
                inp = str(item.get('input', '')).strip()
                prompt = f"user:\n{inst}\n\nSentence:\n{inp}\n\nassistant:\n" if inp else f"user:\n{inst}\n\nassistant:\n"
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
                model(**inputs)
    finally:
        for h in handles:
            h.remove()

    layer_states = {li: torch.cat(captured[li], dim=0).numpy() for li in target_layers}
    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(layer_states, f)
    return layer_states


def compute_4_metrics(base_states):
    """计算 4 个前沿表征指标"""
    N, d = base_states.shape
    MAX_VAL = 1e8

    # 1. SOS (Spectral Outlier Score)
    batch_mean = np.mean(base_states, axis=0, keepdims=True)
    try:
        centered_states = base_states - batch_mean
        _, _, Vt = np.linalg.svd(centered_states, full_matrices=False)
        sos = np.clip(np.square(np.dot(centered_states, Vt[0])), 0, MAX_VAL)
    except Exception:
        sos = np.zeros(N)

    # 2. AKI (Activation Kurtosis Index)
    sample_mean = np.mean(base_states, axis=1, keepdims=True)
    centered_h = base_states - sample_mean
    dim_var = np.mean(centered_h**2, axis=1) + 1e-6
    aki = np.clip((np.mean(centered_h**4, axis=1) / (dim_var**2)) - 3.0, -10.0, 1e4)

    # 3. ADI (Angular Dispersion Index)
    normed = base_states / (np.linalg.norm(base_states, axis=1, keepdims=True) + 1e-8)
    mean_dir = np.mean(normed, axis=0, keepdims=True)
    mean_dir = mean_dir / (np.linalg.norm(mean_dir, axis=1, keepdims=True) + 1e-8)
    adi = np.clip(1.0 - np.sum(normed * mean_dir, axis=1), 0.0, 2.0)

    # 4. REE (Representation Energy Entropy)
    p_act = np.abs(base_states) / (np.sum(np.abs(base_states), axis=1, keepdims=True) + 1e-8)
    entropy = -np.sum(np.clip(p_act, 1e-12, 1.0) * np.log(np.clip(p_act, 1e-12, 1.0)), axis=1)
    max_entropy = np.log(float(d) + 1e-6)
    ree = np.clip((max_entropy - entropy) / (max_entropy + 1e-6), 0.0, 1.0)

    return {
        'Spectral Outlier (SOS)': sos,
        'Activation Kurtosis (AKI)': aki,
        'Angular Dispersion (ADI)': adi,
        'Energy Entropy Defect (REE)': ree
    }


# =====================================================================
# 3. 计算各层 AUC 与绘制验证图
# =====================================================================

def plot_layer_feature_specificity_trap(layer_auc_dict, model_name, attacker_name, dataset_name, save_path):
    """绘制符合学术顶会规范的层-特征特异性陷阱曲线图"""
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)

    layers = sorted(list(layer_auc_dict.keys()))
    features = list(next(iter(layer_auc_dict.values())).keys())

    # 专业配色与标记
    color_map = {
        'Spectral Outlier (SOS)': '#E64B35',      # 红色
        'Activation Kurtosis (AKI)': '#4DBBD5',   # 蓝色
        'Angular Dispersion (ADI)': '#00A087',    # 绿色
        'Energy Entropy Defect (REE)': '#3C5488'  # 深蓝紫色
    }
    marker_map = {
        'Spectral Outlier (SOS)': 'o',
        'Activation Kurtosis (AKI)': 's',
        'Angular Dispersion (ADI)': '^',
        'Energy Entropy Defect (REE)': 'D'
    }

    for feat in features:
        auc_values = [layer_auc_dict[l][feat] for l in layers]
        ax.plot(
            layers, auc_values,
            label=feat,
            color=color_map.get(feat, None),
            marker=marker_map.get(feat, 'o'),
            markersize=5,
            linewidth=2.2,
            alpha=0.88
        )

    # 随机基线 (AUC = 0.5) 与 高辨别力参考线
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1.2, alpha=0.7, label='Random Guess (AUC=0.5)')
    ax.axhline(0.9, color='#7E6148', linestyle=':', linewidth=1.0, alpha=0.5)

    # 视觉标注：浅层盲区与深层波动
    ax.axvspan(-0.5, 3.5, color='gray', alpha=0.10, label='Shallow Blind Zone')

    ax.set_title(
        f"Validation of Layer-Feature Specificity Trap\n({model_name} | {attacker_name} on {dataset_name})",
        fontsize=13, fontweight='bold', pad=12
    )
    ax.set_xlabel("Transformer Layer Index ($l$)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Detection Discriminability (ROC-AUC)", fontsize=12, fontweight='bold')
    ax.set_xlim(-0.5, len(layers) - 0.5)
    ax.set_ylim(0.40, 1.02)
    ax.set_xticks(np.arange(0, len(layers), 2 if len(layers) <= 32 else 4))

    ax.legend(loc='lower right', frameon=True, framealpha=0.92, facecolor='white', fontsize=10)
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ 验证图已成功绘制并保存至: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="验证层-特征特异性陷阱实验")
    parser.add_argument('--dataset', type=str, default='SST-2')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--model_name', type=str, default='Meta-Llama-3-8B-Instruct')
    parser.add_argument('--attacker_name', type=str, default='BadNets')
    parser.add_argument('--task', type=str, default='classification')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--output_dir', type=str, default='./outputs_specificity_trap')
    args = parser.parse_args()

    # 1. 准备数据与模型
    model, tokenizer, raw_dataset = prepare_project_data_and_model(args)
    y_true = np.array([int(item.get('poisoned', 0)) for item in raw_dataset])

    # 2. 提取隐状态
    cache_path = os.path.join(args.output_dir, f"{args.model_name}_{args.attacker_name}_{args.dataset}_hs.pkl")
    layer_states = extract_all_layers_hidden_states(model, tokenizer, raw_dataset, args.device, cache_path)

    # 3. 逐层计算各指标的 AUC
    print(f"\n[3/3] 正在逐层评估 {len(layer_states)} 个 Transformer 层的特征 AUC 分布...")
    layer_auc_dict = {}

    for l_idx, h_states in layer_states.items():
        feat_dict = compute_4_metrics(h_states)
        layer_auc_dict[l_idx] = {}
        for fname, fvals in feat_dict.items():
            # 计算无监督判别力 (方向对齐: max(auc, 1-auc))
            try:
                auc = roc_auc_score(y_true, fvals)
                auc_aligned = max(auc, 1.0 - auc)
            except Exception:
                auc_aligned = 0.5
            layer_auc_dict[l_idx][fname] = auc_aligned

    # 打印部分具有代表性层的指标对比
    print("-" * 75)
    print(f"{'Layer Index':<12} | {'SOS AUC':<12} | {'AKI AUC':<12} | {'ADI AUC':<12} | {'REE AUC':<12}")
    print("-" * 75)
    for l in [0, 4, 8, 12, 16, 20, 24, len(layer_states)-1]:
        if l in layer_auc_dict:
            s = layer_auc_dict[l]
            print(f"Layer {l:<6} | {s['Spectral Outlier (SOS)']:<12.4f} | {s['Activation Kurtosis (AKI)']:<12.4f} | {s['Angular Dispersion (ADI)']:<12.4f} | {s['Energy Entropy Defect (REE)']:<12.4f}")
    print("-" * 75)

    # 4. 生成图表
    save_fig_path = os.path.join(args.output_dir, f"trap_{args.model_name}_{args.attacker_name}_{args.dataset}.png")
    plot_layer_feature_specificity_trap(
        layer_auc_dict, args.model_name, args.attacker_name, args.dataset, save_fig_path
    )


if __name__ == '__main__':
    main()