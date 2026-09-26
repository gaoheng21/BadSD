"""
"""
import os
import copy
import random
import datetime
import pickle
import itertools
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import davies_bouldin_score
import warnings

from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq, AutoTokenizer

from utils.load_info import load_model, load_label_rate
from mydataset.process_data import process_to_json
from mydataset.poisoner import poison_data, do_poison
from defenders.lib.prune import get_transformer_layers

from evaluators.evaluate_eraser_lora import get_utility_hs
from generators.classification_attack import get_ASR_classification, get_CACC_classification
from generators.jailbreak_attack import get_ASR_jailbreak, get_CACC_jailbreak
from defenders.train_model import finetune_alpaca
from transformers import (
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup
)

warnings.filterwarnings('ignore', category=RuntimeWarning, message='overflow encountered')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered')

os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

LOG_DIR = './logs/Windtunnel_static'
HIDDEN_STATES_DIR = './logs/Windtunnel_static/hidden_states'
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(HIDDEN_STATES_DIR, exist_ok=True)


# =====================================================================
# =====================================================================

def _compute_dbi(X):
    """
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    gmm = GaussianMixture(n_components=2, random_state=42, n_init=10, covariance_type='full')
    labels = gmm.fit_predict(X_scaled)
    if len(np.unique(labels)) < 2:
        return float('inf')
    return davies_bouldin_score(X_scaled, labels)


def calculate_defense_weights_by_latent_prism(model, tokenizer, raw_data, device, exp_id='Unknown', current_noise=0, task='Sentiment_Steering', ablation=None):
    """
    """
    from defenders.defender_hidden_state import _log
    ablation = ablation or {}

    total_len = len(raw_data)
    true_ground_truth = np.array([item.get('poisoned', 0) for item in raw_data], dtype=int)

    _log("\n" + "=" * 70)
    _log("LatentPrism 静态特征检测 (基于 (l*, F*) 多维特征子集 DBI 联合优化)")
    _log("=" * 70)

    layers = get_transformer_layers(model)
    n_total = len(layers)
    fixed_config_mode = False
    if not ablation.get('use_dbi', True) and ablation.get('fixed_layer') is not None:
        target_layers = [int(ablation['fixed_layer']) % n_total]
        fixed_config_mode = True
        _log(f"消融模式 (w/o DBI Selection): 固定检测层 L{target_layers[0]}，层与特征子集均不做 DBI 寻优")
    else:
        target_layers = list(range(n_total))
        _log(f"DBI 自动寻优模式: 扫描全部 {n_total} 层")

    allowed_features = ablation.get('features')
    candidate_features = [
        'Intra-Representation Dispersion (IRD)',
        'Global Manifold Deviation (GMD)',
        'Coupled Heterogeneity Index (CHI)',
        'Sparse Outlier Divergence (SOD)',
        'Normalized Activation Volatility (NAV)',
        'Peak Activation Drift (PAD)'
    ]
    if allowed_features:
        candidate_features = [f for f in candidate_features
                              if f.split('(')[-1].rstrip(')') in set(allowed_features)]
        _log(f"消融模式: 特征白名单 {allowed_features} -> {len(candidate_features)} 个候选特征")

    parts = exp_id.split('-')
    if len(parts) >= 4:
        model_part = parts[0]
        for i in range(1, len(parts)):
            if parts[i] in ['SST', 'AGNews', 'HateSpeech', 'DNA', 'DAN', 'AdvBench', 'SimpleQA', 'TruthfulQA', 'WebQuestions', 'Alpaca']:
                model_name_part = '-'.join(parts[:i])
                if parts[i] == 'SST' and i + 1 < len(parts) and parts[i + 1] == '2':
                    data_name_part = 'SST-2'
                    attacker_idx = i + 2
                else:
                    data_name_part = parts[i]
                    attacker_idx = i + 1
                if attacker_idx < len(parts):
                    attacker_name_part = parts[attacker_idx]
                    cache_exp_id = f"{model_name_part}_{attacker_name_part}_{data_name_part}_Static"
                    break
        else:
            cache_exp_id = exp_id.replace('-', '_') + '_Static'
    else:
        cache_exp_id = exp_id.replace('-', '_') + '_Static'

    _log(f"缓存 ID: {cache_exp_id}")
    layer_states = extract_base_hidden_states_per_layer(model, tokenizer, raw_data, target_layers, device, cache_exp_id)


    if fixed_config_mode:
        all_subsets = [list(candidate_features)]
        _log(f"消融模式: 固定特征子集（{len(candidate_features)} 维投影），共 {len(target_layers)} 个固定层")
    else:
        all_subsets = []
        for r in range(1, len(candidate_features) + 1):
            for combo in itertools.combinations(candidate_features, r):
                all_subsets.append(list(combo))
        _log(f"待评估特征子集数: {len(all_subsets)} (总搜索空间: {len(target_layers)} 层 × {len(all_subsets)} 子集 = {len(target_layers) * len(all_subsets)} 种配置)")

    best_overall = {
        'dbi': float('inf'),
        'layer': None,
        'features': None,
        'matrix': None
    }
    best_per_single_feature = {f: {'dbi': float('inf'), 'layer': None, 'vals': None} for f in candidate_features}

    _log("\n开始执行 (l, F) 空间 DBI 最小化全局寻优...")
    for layer_idx in target_layers:
        base_states = layer_states[layer_idx]
        features_dict = compute_static_features(base_states)

        for fname in candidate_features:
            dbi_single = _compute_dbi(features_dict[fname])
            if dbi_single < best_per_single_feature[fname]['dbi']:
                best_per_single_feature[fname] = {'dbi': dbi_single, 'layer': layer_idx, 'vals': features_dict[fname]}

        for subset in all_subsets:
            subset_mat = np.column_stack([features_dict[f] for f in subset])
            dbi_val = _compute_dbi(subset_mat)
            if dbi_val < best_overall['dbi']:
                best_overall = {
                    'dbi': dbi_val,
                    'layer': layer_idx,
                    'features': subset,
                    'matrix': subset_mat
                }

    _log("\n单特征最佳单层 DBI (基准参考):")
    for fname in candidate_features:
        best = best_per_single_feature[fname]
        if best['layer'] is None or best['dbi'] is None or not np.isfinite(best['dbi']):
            _log(f"  * {fname:<40} 所有扫描层均无有效二分 (DBI=inf)")
            continue
        _log(f"  * {fname:<40} Layer L{best['layer']:<2} (DBI={best['dbi']:.4f})")

    short_opt_features = [f.split('(')[-1].rstrip(')') for f in best_overall['features']]
    _log("\n" + "=" * 70)
    _log("DBI 优化收敛最优配置 (l*, F*):")
    _log(f"  * 最佳层 index (l*)      : Layer L{best_overall['layer']}")
    _log(f"  * 最优特征子集 (F*)     : {', '.join(short_opt_features)} (维度 D={len(best_overall['features'])})")
    _log(f"  * 全局最小 DBI 指数     : {best_overall['dbi']:.4f}")
    _log("=" * 70)

    target_masks = multi_feature_clustering_detect(best_overall['matrix'], true_ground_truth, _log)

    _log("\n" + "=" * 70)
    _log("[LatentPrism 检测效果统计]")
    _log("=" * 70)
    _log(f"  识别中毒样本 {int(target_masks.sum())} / 干净样本 {total_len - int(target_masks.sum())}")
    _print_metrics_matrix(target_masks, true_ground_truth, f"{exp_id} [LatentPrism]")
    _log("=" * 70 + "\n")

    try:
        _plot_latent_prism_tsne(
            layer_states[best_overall['layer']],
            true_ground_truth,
            target_masks,
            f"{'+'.join(short_opt_features)}",
            best_overall['layer'],
            best_overall['dbi'],
            exp_id
        )
    except Exception as e:
        _log(f"[LatentPrism] 可视化失败（不影响主防御流程）: {e}")

    return target_masks.tolist()


def extract_base_hidden_states_per_layer(model, tokenizer, data, target_layers, device, exp_id='static'):
    cache_path = os.path.join(HIDDEN_STATES_DIR, f'{exp_id}_hidden_states.pkl')

    cached_data = None
    if os.path.exists(cache_path):
        print(f">>> [LatentPrism] Loading cached hidden states from {cache_path}...")
        try:
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)

            cached_n = next(iter(cached_data.values())).shape[0]
            if cached_n != len(data):
                print(f"    ⚠ Warning: Cached samples ({cached_n}) != data size ({len(data)})")
                print("    Regenerating hidden states...")
                cached_data = None
        except Exception as e:
            print(f"    ✗ Failed to load cache: {e}")
            print("    Regenerating hidden states...")
            cached_data = None

    missing_layers = [l for l in target_layers if cached_data is None or l not in cached_data]
    if cached_data is not None and not missing_layers:
        print(f"    ✓ Successfully loaded {len(target_layers)} layers from cache")
        return {l: cached_data[l] for l in target_layers}
    if cached_data is not None and missing_layers:
        print(f"    ⚠ 缓存缺少层 {missing_layers}，增量提取后与已有 {len(cached_data)} 层合并")
        target_layers = missing_layers

    model.eval()
    layers = get_transformer_layers(model)

    print(f">>> [LatentPrism] Extracting base hidden states (H0 only, per-layer)...")
    captured = {i: [] for i in target_layers}

    def make_hook(layer_idx):
        def hook_fn(module, inp, output):
            hidden = (output[0] if isinstance(output, tuple) else output).detach()
            captured[layer_idx].append(hidden[:, -1, :].cpu())
        return hook_fn

    handles = [layers[i].register_forward_hook(make_hook(i)) for i in target_layers if i < len(layers)]

    try:
        with torch.no_grad():
            for item in tqdm(data, desc="  提取隐状态"):
                prompt = build_prompt(item)
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
                model(**inputs)
    finally:
        for h in handles:
            h.remove()

    layer_states = {}
    for li in target_layers:
        feats = torch.cat(captured[li], dim=0).numpy()
        layer_states[li] = feats

    print(f">>> [LatentPrism] Extraction complete. {len(target_layers)} layers extracted.")

    if cached_data is not None:
        merged = dict(cached_data)
        merged.update(layer_states)
        layer_states = merged

    print(f">>> [LatentPrism] Saving hidden states to {cache_path}...")
    with open(cache_path, 'wb') as f:
        pickle.dump(layer_states, f)
    print(f"    ✓ Hidden states saved (size: {os.path.getsize(cache_path) / 1024 / 1024:.2f} MB)")

    return layer_states


def build_prompt(item):
    """
    """
    inst = item.get('instruction', '')
    inp = str(item.get('input', ''))
    if inp.strip():
        return f"user:\n{inst}\n\nSentence:\n{inp}\n\nassistant:\n"
    return f"user:\n{inst}\n\nassistant:\n"


def compute_static_features(base_states):
    """
    base_states: [N, d]
    """
    N, d = base_states.shape
    MAX_VAL = 1e8
    MIN_VAL = -1e8

    # 1. IRD (Intra-Representation Dispersion)
    mu_i = np.mean(base_states, axis=1, keepdims=True)
    repguard_local_var = np.mean((base_states - mu_i)**2, axis=1)
    repguard_local_var = np.clip(repguard_local_var, 0, MAX_VAL)
    repguard_local_var[~np.isfinite(repguard_local_var)] = 0.0

    # 2. GMD (Global Manifold Deviation)
    batch_mean = np.mean(base_states, axis=0, keepdims=True)
    diff_squared = (base_states - batch_mean)**2
    diff_squared = np.clip(diff_squared, 0, MAX_VAL)
    repguard_sample_dev = np.sqrt(np.sum(diff_squared, axis=1))
    repguard_sample_dev = np.clip(repguard_sample_dev, 0, MAX_VAL)
    repguard_sample_dev[~np.isfinite(repguard_sample_dev)] = 0.0

    # 3. CHI (Coupled Heterogeneity Index)
    norm_var = (repguard_local_var - np.mean(repguard_local_var)) / (np.std(repguard_local_var) + 1e-6)
    norm_dev = (repguard_sample_dev - np.mean(repguard_sample_dev)) / (np.std(repguard_sample_dev) + 1e-6)
    norm_var = np.clip(norm_var, -100, 100)
    norm_dev = np.clip(norm_dev, -100, 100)
    norm_var[~np.isfinite(norm_var)] = 0.0
    norm_dev[~np.isfinite(norm_dev)] = 0.0
    variance_deviation_ratio = norm_dev / (np.abs(norm_var) + 0.1)
    variance_deviation_ratio = np.clip(variance_deviation_ratio, -100, 100)
    variance_deviation_ratio[~np.isfinite(variance_deviation_ratio)] = 0.0

    # 4. SOD (Sparse Outlier Divergence)
    diff = np.abs(base_states - batch_mean)
    diff = np.clip(diff, 0, MAX_VAL)
    threshold = np.percentile(diff, 90, axis=1, keepdims=True)
    mask = diff >= threshold
    masked_diff = np.clip(diff * mask, 0, MAX_VAL)
    quantile_deviation = np.linalg.norm(masked_diff, axis=1)
    quantile_deviation = np.clip(quantile_deviation, 0, MAX_VAL)
    quantile_deviation[~np.isfinite(quantile_deviation)] = 0.0
    if np.any(np.isfinite(quantile_deviation)) and len(quantile_deviation[np.isfinite(quantile_deviation)]) > 0:
        upper_bound = np.percentile(quantile_deviation[np.isfinite(quantile_deviation)], 99.9)
        quantile_deviation = np.clip(quantile_deviation, 0, upper_bound)

    # 5. NAV (Normalized Activation Volatility)
    sample_mean = np.mean(base_states, axis=1, keepdims=True)
    sample_std = np.std(base_states, axis=1)
    sample_mean = np.clip(sample_mean, MIN_VAL, MAX_VAL)
    sample_std = np.clip(sample_std, 0, MAX_VAL)
    sample_mean[~np.isfinite(sample_mean)] = 0.0
    sample_std[~np.isfinite(sample_std)] = 0.0
    relative_variance = sample_std / (np.abs(sample_mean).flatten() + np.abs(sample_std) * 0.1 + 1e-6)
    relative_variance = np.clip(relative_variance, 0, MAX_VAL)
    relative_variance[~np.isfinite(relative_variance)] = 0.0
    if np.any(np.isfinite(relative_variance)) and len(relative_variance[np.isfinite(relative_variance)]) > 0:
        upper_bound = np.percentile(relative_variance[np.isfinite(relative_variance)], 99.9)
        relative_variance = np.clip(relative_variance, 0, upper_bound)

    # 6. PAD (Peak Activation Drift)
    max_activation_deviation = np.max(diff, axis=1)
    max_activation_deviation = np.clip(max_activation_deviation, 0, MAX_VAL)
    max_activation_deviation[~np.isfinite(max_activation_deviation)] = 0.0

    return {
        'Intra-Representation Dispersion (IRD)': repguard_local_var,
        'Global Manifold Deviation (GMD)': repguard_sample_dev,
        'Coupled Heterogeneity Index (CHI)': variance_deviation_ratio,
        'Sparse Outlier Divergence (SOD)': quantile_deviation,
        'Normalized Activation Volatility (NAV)': relative_variance,
        'Peak Activation Drift (PAD)': max_activation_deviation,
    }


def multi_feature_clustering_detect(feat_matrix, y_true, log_fn):
    if feat_matrix.ndim == 1:
        X = feat_matrix.reshape(-1, 1)
    else:
        X = feat_matrix
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    gmm = GaussianMixture(n_components=2, random_state=42, n_init=10, covariance_type='full')
    preds_cluster = gmm.fit_predict(X_scaled)

    cluster_0_size = np.sum(preds_cluster == 0)
    cluster_1_size = np.sum(preds_cluster == 1)
    poison_cluster_idx = 0 if cluster_0_size < cluster_1_size else 1

    final_preds = (preds_cluster == poison_cluster_idx).astype(int)
    log_fn(f"\n最优多维特征流形 GMM 聚类 (K=2, D={X.shape[1]}): 簇0={cluster_0_size}, 簇1={cluster_1_size}, 判定中毒簇={poison_cluster_idx}")
    return final_preds


def _plot_latent_prism_tsne(hidden_states, y_true, pred_masks, best_features_str, best_layer_idx, best_dbi, exp_id):
    N = hidden_states.shape[0]
    n_components_pca = min(50, N - 1, hidden_states.shape[1])

    pca = PCA(n_components=n_components_pca, random_state=42)
    feats_pca = pca.fit_transform(hidden_states)

    perplexity = min(30, N - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    feats_2d = tsne.fit_transform(feats_pca)

    clean_mask = y_true == 0
    poison_mask = y_true == 1
    pred_clean = pred_masks == 0
    pred_poison = pred_masks == 1

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax1 = axes[0]
    ax1.scatter(feats_2d[clean_mask, 0], feats_2d[clean_mask, 1],
                c='#2196F3', alpha=0.6, s=40, label='Clean', edgecolors='none')
    ax1.scatter(feats_2d[poison_mask, 0], feats_2d[poison_mask, 1],
                c='#F44336', alpha=0.6, s=40, label='Poisoned', edgecolors='none')
    ax1.legend(fontsize=12)
    ax1.set_xlabel('Dim 1', fontsize=13)
    ax1.set_ylabel('Dim 2', fontsize=13)
    ax1.set_title('Ground Truth Labels', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.scatter(feats_2d[pred_clean, 0], feats_2d[pred_clean, 1],
                c='#2196F3', alpha=0.6, s=40, label='Pred Clean', edgecolors='none')
    ax2.scatter(feats_2d[pred_poison, 0], feats_2d[pred_poison, 1],
                c='#F44336', alpha=0.6, s=40, label='Pred Poisoned', edgecolors='none')
    ax2.legend(fontsize=12)
    ax2.set_xlabel('Dim 1', fontsize=13)
    ax2.set_ylabel('Dim 2', fontsize=13)
    ax2.set_title('LatentPrism Detection Result', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f'{exp_id} | Optimal: Layer L{best_layer_idx} | $\\mathcal{{F}}^*$: {best_features_str} (DBI={best_dbi:.4f})',
        fontsize=14, fontweight='bold', y=1.01
    )
    plt.tight_layout()

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = LOG_DIR
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{exp_id}_tsne_{ts}.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

    from defenders.defender_hidden_state import _log
    _log(f"[LatentPrism] t-SNE 图已保存: {save_path}")


def _print_metrics_matrix(y_pred, y_true, exp_id='Unknown'):
    from defenders.defender_hidden_state import _log

    N = len(y_true)
    TP = int(((y_pred == 1) & (y_true == 1)).sum())
    TN = int(((y_pred == 0) & (y_true == 0)).sum())
    FP = int(((y_pred == 1) & (y_true == 0)).sum())
    FN = int(((y_pred == 0) & (y_true == 1)).sum())

    accuracy = (TP + TN) / N if N > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    lines = [
        "=" * 55,
        f"  检测混淆矩阵: {exp_id}",
        "-" * 55,
        f"  Accuracy       = {accuracy:.4f}",
        f"  Precision      = {precision:.4f}  | Recall = {recall:.4f}",
        f"  F1-Score       = {f1:.4f}",
        f"  细节 -> TP: {TP}, FN: {FN}, TN: {TN}, FP: {FP}",
        "=" * 55,
    ]
    for line in lines:
        _log(line)


# =====================================================================
# =====================================================================
def do_LatentPrism(**kwargs):
    """
    """
    import defenders.defender_hidden_state as hs_module
    original_calc_func = hs_module.calculate_defense_weights_by_hs

    ablation = {
        'features': kwargs.pop('features', None),
        'use_dbi': kwargs.pop('use_dbi', True),
        'fixed_layer': kwargs.pop('fixed_layer', None),
    }

    def wrapped_calc_func(model, tokenizer, raw_data, device, exp_id='Unknown', current_noise=0, task='Sentiment_Steering'):
        return calculate_defense_weights_by_latent_prism(
            model, tokenizer, raw_data, device, exp_id, current_noise, task, ablation=ablation
        )

    hs_module.calculate_defense_weights_by_hs = wrapped_calc_func

    defender_name = kwargs.pop('defender_name', None) or 'LatentPrism'

    try:
        result = hs_module.do_HiddenState(defender_name=defender_name, **kwargs)
        return result
    finally:
        hs_module.calculate_defense_weights_by_hs = original_calc_func