# =====================================================================
# windtunnel_static_features.py
# ---------------------------------------------------------------------
# 近年前沿表征指标版本（2023-2025/2026 最新顶会基线）
# 包含：
# 1. Hoyer 稀疏度 (ICLR 2024 / ICML 2024)
# 2. 层间流速突变 (ACL 2024 / NeurIPS 2023)
# 3. 主成分方向投影 (ICLR 2024 - Representation Engineering)
# 4. 全局余弦偏离度 (ICLR 2024 / RepGuard)
# 5. 峰均激活尖锐度 (CROW 2024 / Sparse Trojan Mitigation)
# 6. 尺度归一化方差 (NeurIPS 2023 / ICLR 2024)
# =====================================================================
import os
import random
import datetime
import pickle
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from utils.load_info import load_model, load_label_rate
from defenders.lib.prune import get_transformer_layers

os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

LOG_DIR = './logs/Windtunnel_static'
HIDDEN_STATES_DIR = './logs/Windtunnel_static/hidden_states'
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(HIDDEN_STATES_DIR, exist_ok=True)


# =====================================================================
# 1. 提取基准态隐状态（逐层分离，支持保存和加载缓存）
# =====================================================================
@torch.no_grad()
def extract_base_hidden_states_per_layer(model, tokenizer, data, target_layers, device,
                                         exp_id='static', load_cached=False):
    """
    提取基准态隐状态，逐层分离存储 {layer_idx: [N, d_layer]}
    支持加载和保存 pickle 缓存
    """
    cache_path = os.path.join(HIDDEN_STATES_DIR, f'{exp_id}_hidden_states.pkl')

    if load_cached and os.path.exists(cache_path):
        print(f">>> [Static] Loading cached hidden states from {cache_path}...")
        try:
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)

            if len(cached_data) == len(target_layers):
                print(f"    ✓ Successfully loaded {len(target_layers)} layers from cache")
                return cached_data
            else:
                print(f"    ⚠ Warning: Cached layers ({len(cached_data)}) != requested ({len(target_layers)})")
                print(f"    Regenerating hidden states...")
        except Exception as e:
            print(f"    ✗ Failed to load cache: {e}")
            print(f"    Regenerating hidden states...")

    model.eval()
    layers = get_transformer_layers(model)

    print(f">>> [Static] Extracting base hidden states (H0 only, per-layer)...")
    captured = {i: [] for i in target_layers}

    def make_hook(layer_idx):
        def hook_fn(module, inp, output):
            hidden = (output[0] if isinstance(output, tuple) else output).detach()
            captured[layer_idx].append(hidden[:, -1, :].cpu())
        return hook_fn

    handles = [layers[i].register_forward_hook(make_hook(i)) for i in target_layers if i < len(layers)]

    try:
        for item in tqdm(data, desc="  Extracting Samples"):
            prompt = build_prompt(item)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
            model(**inputs)
    finally:
        for h in handles:
            h.remove()

    layer_states = {}
    for li in target_layers:
        feats = torch.cat(captured[li], dim=0).numpy()  # [N, d_layer]
        layer_states[li] = feats

    print(f">>> [Static] Extraction complete. {len(target_layers)} layers extracted.")

    print(f">>> [Static] Saving hidden states to {cache_path}...")
    with open(cache_path, 'wb') as f:
        pickle.dump(layer_states, f)
    print(f"    ✓ Hidden states saved (size: {os.path.getsize(cache_path) / 1024 / 1024:.2f} MB)")

    return layer_states


def build_prompt(item):
    """构建输入提示词"""
    inst = item.get('instruction', '')
    inp = str(item.get('input', ''))
    if inp.strip():
        return f"user:\n{inst}\n\nSentence:\n{inp}\n\nassistant:\n"
    return f"user:\n{inst}\n\nassistant:\n"


# =====================================================================
# 2. 2023-2025/2026 前沿表征指标计算
# =====================================================================
def compute_modern_features(curr_states, prev_states=None):
    """
    计算近年前沿顶会表征检测统计量
    curr_states: [N, d]
    prev_states: [N, d] (上一层激活状态，用于计算层间流速)
    """
    N, d = curr_states.shape
    batch_mean = np.mean(curr_states, axis=0, keepdims=True)  # [1, d]
    diff = curr_states - batch_mean                           # [N, d]
    abs_states = np.abs(curr_states)

    # 1. Hoyer 稀疏度 (ICLR 2024 - Sparse Feature Circuits)
    l1_norm = np.sum(abs_states, axis=1)
    l2_norm = np.linalg.norm(curr_states, axis=1) + 1e-8
    hoyer_sparsity = (np.sqrt(d) - (l1_norm / l2_norm)) / (np.sqrt(d) - 1.0)  # [N]

    # 2. 层间流速突变 (ACL 2024 / NeurIPS 2023 - Representation Flow)
    if prev_states is not None:
        trans_diff = np.linalg.norm(curr_states - prev_states, axis=1)
        prev_norm = np.linalg.norm(prev_states, axis=1) + 1e-8
        layer_velocity = trans_diff / prev_norm  # [N]
    else:
        layer_velocity = np.zeros(N)

    # 3. 主成分方向投影 (ICLR 2024 - Representation Engineering)
    try:
        _, _, Vt = np.linalg.svd(diff, full_matrices=False)
        v1 = Vt[0]  # 第一主成分方向 [d]
        pca_projection = np.abs(np.dot(diff, v1))  # [N]
    except Exception:
        pca_projection = np.zeros(N)

    # 4. 全局余弦偏离度 (ICLR 2024 / RepGuard)
    norm_base = l2_norm[:, None]
    norm_mean = np.linalg.norm(batch_mean) + 1e-8
    cos_sim = np.dot(curr_states, batch_mean.T) / (norm_base * norm_mean)
    cosine_deviation = (1.0 - cos_sim).flatten()  # [N]

    # 5. 峰均激活尖锐度 (CROW 2024 / Sparse Trojan Mitigation)
    peak_to_average = np.max(abs_states, axis=1) / (np.mean(abs_states, axis=1) + 1e-8)  # [N]

    # 6. 尺度归一化方差 (NeurIPS 2023 / ICLR 2024)
    feature_var = np.var(curr_states, axis=0, keepdims=True) + 1e-6
    normalized_variance = np.sum((diff ** 2) / feature_var, axis=1)  # [N]

    return {
        'hoyer_sparsity': hoyer_sparsity,
        'layer_velocity': layer_velocity,
        'pca_projection': pca_projection,
        'cosine_deviation': cosine_deviation,
        'peak_to_average': peak_to_average,
        'normalized_variance': normalized_variance
    }


def _auc_dir_agnostic(vals, y_true):
    """方向不可知的 AUC 计算"""
    try:
        if len(np.unique(y_true)) == 1:
            return 0.5
        auc = roc_auc_score(y_true, vals)
        return max(auc, 1.0 - auc)
    except Exception:
        return 0.5


# =====================================================================
# 3. 可视化
# =====================================================================
def plot_single_distribution(vals, y_true, feature_name, save_path, title):
    clean = y_true == 0
    poison = y_true == 1

    fig, ax = plt.subplots(figsize=(6, 4))
    lo, hi = np.percentile(vals, 1), np.percentile(vals, 99)
    bins = np.linspace(lo, hi, 40) if hi > lo else 40

    ax.hist(vals[clean], bins=bins, color='#2196F3', alpha=0.6, label='Clean')
    ax.hist(vals[poison], bins=bins, color='#F44336', alpha=0.6, label='Poisoned')

    ax.axvline(vals[clean].mean(), color='#1565C0', ls='--', lw=1.5)
    ax.axvline(vals[poison].mean(), color='#B71C1C', ls='--', lw=1.5)

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(feature_name)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_feature_correlation_heatmap(feature_dict, y_true, exp_id, save_path):
    """绘制特征相关性热图"""
    print(f">>> [Visualize] Generating feature correlation heatmap...")

    feature_names = list(feature_dict.keys())
    feature_matrix = np.array([feature_dict[name] for name in feature_names]).T

    corr_matrix = np.corrcoef(feature_matrix.T)

    aucs = []
    for name in feature_names:
        auc = _auc_dir_agnostic(feature_dict[name], y_true)
        aucs.append(auc)

    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(corr_matrix,
                annot=True,
                fmt='.2f',
                cmap='coolwarm',
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={'label': 'Pearson Correlation'},
                xticklabels=[f"{name}\n(AUC={auc:.3f})" for name, auc in zip(feature_names, aucs)],
                yticklabels=feature_names,
                ax=ax,
                vmin=-1, vmax=1)

    ax.set_title(f'{exp_id} - Modern Features Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f">>> [Visualize] Heatmap saved to {save_path}")

    return corr_matrix, aucs


# =====================================================================
# 4. 无监督聚类检测
# =====================================================================
def unsupervised_clustering_detect(scores, feature_name, y_true, log_lines, method='gmm'):
    """无监督二元聚类检测"""
    print(f">>> [Detect] Performing {method.upper()} clustering on {feature_name}...")
    X = scores.reshape(-1, 1)
    X_scaled = StandardScaler().fit_transform(X)

    if method == 'gmm':
        model = GaussianMixture(n_components=2, random_state=42, n_init=5)
    else:
        model = KMeans(n_clusters=2, random_state=42, n_init=10)

    preds_cluster = model.fit_predict(X_scaled)

    if method == 'gmm':
        centers = model.means_.flatten()
    else:
        centers = model.cluster_centers_.flatten()

    lower_is_poison = ['hoyer_sparsity']
    if feature_name in lower_is_poison:
        poison_cluster_idx = np.argmin(centers)
    else:
        poison_cluster_idx = np.argmax(centers)

    final_preds = (preds_cluster == poison_cluster_idx).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, final_preds).ravel()
    eps = 1e-9
    tpr = tp / (tp + fn + eps)
    fpr = fp / (fp + tn + eps)
    f1 = f1_score(y_true, final_preds)
    acc = accuracy_score(y_true, final_preds)

    log_lines.append("\n" + "#" * 74)
    log_lines.append(f"聚类检测结果 ({method.upper()}, K=2) - Feature: {feature_name}")
    log_lines.append(f"假设: {'中毒样本更小' if feature_name in lower_is_poison else '中毒样本更大'}")
    log_lines.append("-" * 74)
    log_lines.append(f"混淆矩阵: TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    log_lines.append(f"TPR: {tpr:.4f}  FPR: {fpr:.4f}  F1: {f1:.4f}  Acc: {acc:.4f}")
    log_lines.append("#" * 74)

    return final_preds, {'tpr': tpr, 'fpr': fpr, 'f1': f1, 'acc': acc}


# =====================================================================
# 5. 主控制流程
# =====================================================================
def observe(model, tokenizer, data, exp_id='modern_obs', device=DEVICE,
            scan_all_layers=True, load_cached=False):
    y_true = np.array([item.get('poisoned', 0) for item in data], dtype=int)
    layers = get_transformer_layers(model)
    n_total = len(layers)

    if scan_all_layers:
        target_layers = list(range(n_total))
    else:
        target_layers = [n_total // 4, n_total // 2, n_total * 3 // 4, n_total - 1]

    cache_status = "Loading cached" if load_cached else "Extracting new"
    log_lines = [
        f"[Modern Per-Layer] exp_id={exp_id}  N={len(data)}  poison_ratio={y_true.mean():.1%}",
        f"[Modern Per-Layer] 2023-2025/2026 Features: Hoyer, Velocity, PCA, Cosine, PAR, NormVar",
        f"[Modern Per-Layer] Scanning {len(target_layers)} layers",
        f"[Modern Per-Layer] Cache mode: {cache_status}"
    ]

    # 1. 提取基准态隐状态
    layer_states = extract_base_hidden_states_per_layer(
        model, tokenizer, data, target_layers, device, exp_id, load_cached
    )

    # 2. 逐层计算 6 种前沿特征并评估
    feature_names = [
        'hoyer_sparsity',
        'layer_velocity',
        'pca_projection',
        'cosine_deviation',
        'peak_to_average',
        'normalized_variance'
    ]

    global_best_auc = 0
    global_best_config = None

    log_lines.append("\n" + "=" * 135)
    log_lines.append("逐层前沿表征特征 AUC 分析:")
    log_lines.append("=" * 135)

    header = f"{'Layer':<7}" + "".join([f"{f[:20]:>21}" for f in feature_names])
    log_lines.append(header)
    log_lines.append("-" * len(header))

    best_per_feature = {f: {'auc': 0, 'layer': None, 'vals': None} for f in feature_names}

    print(">>> [Analyze] Evaluating modern 2023-2025 features per layer...")
    for idx, layer_idx in enumerate(target_layers):
        base_states = layer_states[layer_idx]  # [N, d]
        prev_states = layer_states[target_layers[idx - 1]] if idx > 0 else None
        features = compute_modern_features(base_states, prev_states)

        row_str = f"L{layer_idx:<6}"
        for fname in feature_names:
            auc = _auc_dir_agnostic(features[fname], y_true)
            row_str += f"{auc:>21.3f}"

            if auc > best_per_feature[fname]['auc']:
                best_per_feature[fname] = {'auc': auc, 'layer': layer_idx, 'vals': features[fname]}

            if auc > global_best_auc:
                global_best_auc = auc
                global_best_config = (layer_idx, fname, features[fname], auc)

        log_lines.append(row_str)

    # 3. 输出每个特征的最佳层
    log_lines.append("-" * len(header))
    log_lines.append("\n各特征最佳层:")
    for fname in feature_names:
        best_l = best_per_feature[fname]['layer']
        best_auc = best_per_feature[fname]['auc']
        log_lines.append(f"  * {fname:<25} 最佳层: L{best_l} (AUC={best_auc:.4f})")

        title = f"{exp_id} - {fname} at L{best_l}"
        save_path = os.path.join(LOG_DIR, f'{exp_id}_{fname}_L{best_l}.png')
        plot_single_distribution(best_per_feature[fname]['vals'], y_true, fname, save_path, title)

    # 4. 特征相关性热图
    heatmap_features = {fname: best_per_feature[fname]['vals'] for fname in feature_names}
    heatmap_path = os.path.join(LOG_DIR, f'{exp_id}_correlation_heatmap.png')
    corr_matrix, aucs = plot_feature_correlation_heatmap(heatmap_features, y_true, exp_id, heatmap_path)

    # 5. 全局最佳配置
    best_layer, best_fname, best_scores, best_auc = global_best_config
    log_lines.append("\n" + "*" * 74)
    log_lines.append(f"★ 全局最佳配置: Layer L{best_layer}, Feature {best_fname} (AUC={best_auc:.4f})")
    log_lines.append("*" * 74)

    # 6. 聚类检测
    _, metrics_gmm = unsupervised_clustering_detect(best_scores, best_fname, y_true, log_lines, method='gmm')
    _, metrics_kmeans = unsupervised_clustering_detect(best_scores, best_fname, y_true, log_lines, method='kmeans')

    # 7. 写入报告
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_text = "\n".join(log_lines)
    print(report_text)

    report_path = os.path.join(LOG_DIR, f'{exp_id}_report_{stamp}.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\n>>> [Done] Per-layer analysis finished. Results saved to {LOG_DIR}")


def _load_data_for_obs(task, data_name, attacker_name):
    from mydataset.process_data import process_to_json
    from mydataset.poisoner import poison_data
    rate, target_label = load_label_rate(task, data_name)
    clean_data = process_to_json(data_name, 'train', load=True, write=True)
    poisoned = poison_data(data_name, clean_data, attacker_name,
                           target_label, 'train', rate, load=True, task=task)
    return poisoned


def main():
    parser = argparse.ArgumentParser(description="近年前沿表征指标逐层检测（2023-2025/2026）")
    parser.add_argument('--model_name', default='Meta-Llama-3-8B-Instruct')
    parser.add_argument('--attacker_name', default='Sleeper')
    parser.add_argument('--data_name', default='SST-2')
    parser.add_argument('--task', default='classification', choices=['classification', 'jailbreak'])
    parser.add_argument('--max_samples', type=int, default=None,
                        help='最大样本数，默认None使用全部数据集')
    parser.add_argument('--scan_all_layers', type=lambda s: s.lower() != 'false', default=True)
    parser.add_argument('--load_cached', action='store_true',
                        help='加载已保存的隐特征，跳过提取阶段')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default=DEVICE)
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    print(f">>> [Init] Loading model and tokenizer ({args.model_name})...")
    model, tokenizer = load_model(args.model_name, args.attacker_name, args.data_name, args.task, args.device)

    data = _load_data_for_obs(args.task, args.data_name, args.attacker_name)

    if args.max_samples is not None and len(data) > args.max_samples:
        rng = random.Random(args.seed)
        data = rng.sample(data, args.max_samples)
        print(f">>> [Init] Sampled {args.max_samples} from {len(data)} total samples")
    else:
        print(f">>> [Init] Using full dataset: {len(data)} samples")

    exp_id = f"{args.model_name}_{args.attacker_name}_{args.data_name}_ModernBaselines"

    observe(model, tokenizer, data, exp_id=exp_id, device=args.device,
            scan_all_layers=args.scan_all_layers, load_cached=args.load_cached)


if __name__ == '__main__':
    main()