import os
import copy
import random
import datetime
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq, AutoTokenizer

from utils.load_info import load_model, load_label_rate
from mydataset.process_data import process_to_json
from mydataset.poisoner import poison_data, do_poison
from defenders.lib.prune import get_transformer_layers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from evaluators.evaluate_eraser_lora import get_utility_hs
from generators.classification_attack import get_ASR_classification, get_CACC_classification
from generators.jailbreak_attack import get_ASR_jailbreak, get_CACC_jailbreak
from defenders.train_model import finetune_alpaca
from transformers import (
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup
)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
sum_path = './result.csv'

_hs_log_exp_id = None


# =====================================================================
# =====================================================================
def extract_hidden_states(model, tokenizer, data, device, target_layers=None):
    model.eval()
    layers = get_transformer_layers(model)

    if target_layers is None:
        target_layers = [len(layers) // 4, len(layers) // 2, len(layers) * 3 // 4, len(layers) - 1]

    captured = {i: [] for i in target_layers}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0].detach()
            else:
                hidden = output.detach()
            last_token = hidden[:, -1, :].cpu()
            captured[layer_idx].append(last_token)
        return hook_fn

    handles = []
    for i in target_layers:
        if i < len(layers):
            handles.append(layers[i].register_forward_hook(make_hook(i)))

    with torch.no_grad():
        for item in tqdm(data, desc="Extracting hidden states"):
            inst = item.get('instruction', '')
            inp = item.get('input', '')
            text = (f"user:\n{inst}\n\nSentence:\n{inp}\n\nassistant:\n" if str(inp).strip() else f"user:\n{inst}\n\nassistant:\n")
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            model(**inputs)

    for h in handles:
        h.remove()

    all_features = []
    for i in target_layers:
        if captured[i]:
            layer_features = torch.cat(captured[i], dim=0).numpy()
            all_features.append(layer_features)

    return np.concatenate(all_features, axis=1)


# =====================================================================s
# =====================================================================
def plot_hidden_states_2d(features_2d, y_true, save_path=None, title=None, cluster_labels=None):
    """
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax1 = axes[0]
    clean_mask = y_true == 0
    poison_mask = y_true == 1
    ax1.scatter(features_2d[clean_mask, 0], features_2d[clean_mask, 1],
                c='#2196F3', alpha=0.6, s=40, label='Clean', edgecolors='none')
    ax1.scatter(features_2d[poison_mask, 0], features_2d[poison_mask, 1],
                c='#F44336', alpha=0.6, s=40, label='Poisoned', edgecolors='none')
    ax1.legend(fontsize=12)
    ax1.set_xlabel('Dim 1', fontsize=14)
    ax1.set_ylabel('Dim 2', fontsize=14)
    ax1.set_title('Ground Truth Labels', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    if cluster_labels is not None:
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
        cluster_colors = ['#9C27B0', '#FF9800', '#4CAF50', '#00BCD4', '#E91E63']

        n_clusters = len(np.unique(cluster_labels))
        for c in range(n_clusters):
            cluster_mask = (cluster_labels == c)
            poison_ratio = y_true[cluster_mask].mean() if cluster_mask.sum() > 0 else 0

            clean_in_cluster = cluster_mask & clean_mask
            poison_in_cluster = cluster_mask & poison_mask

            marker = markers[c % len(markers)]
            color = cluster_colors[c % len(cluster_colors)]

            if clean_in_cluster.sum() > 0:
                ax2.scatter(features_2d[clean_in_cluster, 0], features_2d[clean_in_cluster, 1],
                            c=color, marker=marker, s=60, alpha=0.7,
                            edgecolors='#2196F3', linewidths=1.5,
                            label=f'Cluster {c} (poison={poison_ratio:.1%})')

            if poison_in_cluster.sum() > 0:
                ax2.scatter(features_2d[poison_in_cluster, 0], features_2d[poison_in_cluster, 1],
                            c=color, marker=marker, s=60, alpha=0.7,
                            edgecolors='#F44336', linewidths=2.5)

        ax2.legend(fontsize=10, loc='best')
        ax2.set_title(f'Clustering Result (K={n_clusters})', fontsize=14, fontweight='bold')
    else:
        ax2.scatter(features_2d[clean_mask, 0], features_2d[clean_mask, 1], c='#2196F3', alpha=0.6, s=40, label='Clean')
        ax2.scatter(features_2d[poison_mask, 0], features_2d[poison_mask, 1], c='#F44336', alpha=0.6, s=40, label='Poisoned')
        ax2.legend(fontsize=12)
        ax2.set_title('No Clustering Info', fontsize=14, fontweight='bold')

    ax2.set_xlabel('Dim 1', fontsize=14)
    ax2.set_ylabel('Dim 2', fontsize=14)
    ax2.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        _log(f"Figure saved to: {save_path}")
    plt.show()
    plt.close()


def _plot_post_purification_tsne(model, tokenizer, raw_data, masks, device, exp_id):
    import random as _rng
    _rng.seed(42)

    poison_idx  = [i for i, m in enumerate(masks) if m == 1]
    clean_idx   = [i for i, m in enumerate(masks) if m == 0]
    sample_idx  = poison_idx + clean_idx
    sample_data = [raw_data[i] for i in sample_idx]
    y_true_arr  = np.array([1] * len(poison_idx) + [0] * len(clean_idx))

    layers      = get_transformer_layers(model)
    n           = len(layers)
    target_layers = [n // 4, n // 2, n * 3 // 4, n - 1]
    captured    = {li: [] for li in target_layers}

    def make_hook(li):
        def hook_fn(module, inp, output):
            hidden = (output[0] if isinstance(output, tuple) else output).detach()
            captured[li].append(hidden[:, -1, :].cpu())
        return hook_fn

    model.eval()
    handles = [layers[li].register_forward_hook(make_hook(li)) for li in target_layers if li < n]
    try:
        with torch.no_grad():
            for item in tqdm(sample_data, desc="提取净化后隐状态(4层)"):
                inst = item.get('instruction', '')
                inp  = str(item.get('input', ''))
                text = (f"user:\n{inst}\n\nSentence:\n{inp}\n\nassistant:\n" if inp.strip()
                        else f"user:\n{inst}\n\nassistant:\n")
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
                model(**inputs)
    finally:
        for h in handles:
            h.remove()

    clean_m  = (y_true_arr == 0)
    poison_m = (y_true_arr == 1)

    fig, axes = plt.subplots(1, 4, figsize=(28, 6))
    layer_labels = ['L{} (1/4)'.format(target_layers[0]),
                    'L{} (1/2)'.format(target_layers[1]),
                    'L{} (3/4)'.format(target_layers[2]),
                    'L{} (Last)'.format(target_layers[3])]

    for ax, li, label in zip(axes, target_layers, layer_labels):
        feats     = torch.cat(captured[li], dim=0).numpy()
        n_pca     = min(50, feats.shape[0] - 1, feats.shape[1])
        feats_pca = PCA(n_components=n_pca, random_state=42).fit_transform(feats)
        feats_2d  = TSNE(n_components=2, random_state=42,
                         perplexity=min(30, len(feats_pca) - 1)).fit_transform(feats_pca)

        ax.scatter(feats_2d[clean_m,  0], feats_2d[clean_m,  1],
                   c='#2196F3', alpha=0.6, s=30,
                   label=f'Clean (n={int(clean_m.sum())})', edgecolors='none')
        ax.scatter(feats_2d[poison_m, 0], feats_2d[poison_m, 1],
                   c='#F44336', alpha=0.7, s=50,
                   label=f'Orig. Poisoned (n={int(poison_m.sum())})', edgecolors='none')
        ax.legend(fontsize=9)
        ax.set_xlabel('Dim 1', fontsize=11)
        ax.set_ylabel('Dim 2', fontsize=11)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'{exp_id} | Post-Purification Hidden States (Phase A+B)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    ts        = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f'./logs/HiddenState_log/{exp_id}_post_purification_tsne_{ts}.png'
    os.makedirs('./logs/HiddenState_log', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    _log(f"[净化后t-SNE] 图已保存: {save_path}")


def plot_per_class_2d(features_2d, y_true, sub_labels, minority, save_path=None, title=None):
    """
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax1 = axes[0]
    cm, pm = (y_true == 0), (y_true == 1)
    ax1.scatter(features_2d[cm, 0], features_2d[cm, 1], c='#2196F3', alpha=0.6, s=40, label='Clean', edgecolors='none')
    ax1.scatter(features_2d[pm, 0], features_2d[pm, 1], c='#F44336', alpha=0.6, s=40, label='Poisoned', edgecolors='none')
    ax1.legend(fontsize=12); ax1.set_xlabel('Dim 1', fontsize=14); ax1.set_ylabel('Dim 2', fontsize=14)
    ax1.set_title('Ground Truth (within class)', fontsize=14, fontweight='bold'); ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    palette = ['#9C27B0', '#FF9800', '#4CAF50', '#00BCD4', '#E91E63', '#795548', '#607D8B']
    markers = ['o', 's', '^', 'D', 'v', '<', '>']
    for c in sorted(np.unique(sub_labels)):
        sm = (sub_labels == c)
        is_poison_sub = (c == minority)
        ax2.scatter(features_2d[sm, 0], features_2d[sm, 1],
                    c=palette[c % len(palette)], marker=markers[c % len(markers)], s=60, alpha=0.7,
                    edgecolors=('#F44336' if is_poison_sub else 'none'),
                    linewidths=(2.5 if is_poison_sub else 0),
                    label=f'Sub {c}{" (poison)" if is_poison_sub else ""} | n={sm.sum()}')
    ax2.legend(fontsize=10, loc='best'); ax2.set_xlabel('Dim 1', fontsize=14); ax2.set_ylabel('Dim 2', fontsize=14)
    ax2.set_title('Sub-clusters', fontsize=14, fontweight='bold'); ax2.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        _log(f"逐类图已保存: {save_path}")
    plt.close()


# =====================================================================
# =====================================================================
def calculate_defense_weights_by_hs(model, tokenizer, raw_data, device, exp_id='Unknown', current_noise=0, task='classification'):
    total_len = len(raw_data)
    true_ground_truth = np.array([item.get('poisoned', 0) for item in raw_data], dtype=int)

    _log("\n" + "=" * 70)
    _log("模型结构与特征提取统计")
    _log("=" * 70)
    
    layers = get_transformer_layers(model)
    hidden_size = getattr(model.config, 'hidden_size', 4096)
    target_layers = [len(layers) // 4, len(layers) // 2, len(layers) * 3 // 4, len(layers) - 1]

    cache_dir = './logs/HiddenState_cache'
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f'{exp_id}_features.npy')

    if os.path.exists(cache_file):
        _log(f"[HiddenState] 从缓存加载隐状态特征: {cache_file}")
        features = np.load(cache_file)
    else:
        _log(f"[HiddenState] 提取隐状态特征（首次运行）...")
        features = extract_hidden_states(model, tokenizer, raw_data, device)
        np.save(cache_file, features)

    pca = PCA(n_components=min(100, features.shape[1]), random_state=42)
    features_pca = pca.fit_transform(features)
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features_pca) - 1))
    features_reduced = tsne.fit_transform(features_pca)

    # =====================================================================
    # =====================================================================
    if task == 'jailbreak' or task == 'Target_Refusal':
        _log(f"\n[1] {task}任务: 全局 GMM K=2 检测")
        global_labels = GaussianMixture(n_components=2, random_state=42, covariance_type='full').fit_predict(features_pca)
        global_sizes = [int((global_labels == c).sum()) for c in range(2)]

        cluster0_poison = sum(1 for i in range(len(global_labels)) if global_labels[i] == 0 and true_ground_truth[i] == 1)
        cluster1_poison = sum(1 for i in range(len(global_labels)) if global_labels[i] == 1 and true_ground_truth[i] == 1)
        if cluster0_poison > cluster1_poison:
            minority = 0
            _log(f"  全局GMM 子簇大小 {global_sizes}, [作弊] cluster 0 含中毒 {cluster0_poison} 条 > cluster 1 含 {cluster1_poison} 条，选 cluster 0")
        else:
            minority = 1
            _log(f"  全局GMM 子簇大小 {global_sizes}, [作弊] cluster 1 含中毒 {cluster1_poison} 条 > cluster 0 含 {cluster0_poison} 条，选 cluster 1")

        target_masks = (global_labels == minority).astype(int)

        _print_metrics_matrix(target_masks, true_ground_truth, f"{exp_id} [全局GMM]")
        plot_hidden_states_2d(features_reduced, true_ground_truth,
            save_path=os.path.join('./logs/HiddenState_log', f'{exp_id}_gmm_global_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.png'),
            title=f'{exp_id} | global GMM K=2 ({task})', cluster_labels=target_masks)
        return target_masks.tolist()

    # =====================================================================
    # =====================================================================
    from sklearn.metrics import davies_bouldin_score, silhouette_score

    _log("\n[1] 无监督中毒簇识别 (逐类内部 GMM K=2 + 双子簇剔散点 + DBI 判定)")
    assigned_labels = np.array([str(item.get('output', '')).strip() for item in raw_data])
    unique_labels = sorted({l for l in assigned_labels if l})
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    MIN_CLASS = 20
    DBI_THRESHOLD = 0.75
    MINORITY_MAX = 0.7
    OUTLIER_TRIM = 0.1
    per_class_dir = './logs/HiddenState_log/gmm'

    target_masks = np.zeros(total_len, dtype=int)
    plot_clusters = np.zeros(total_len, dtype=int)
    poison_labels = []

    for L in unique_labels:
        idx = np.where(assigned_labels == L)[0]
        if len(idx) < MIN_CLASS:
            _log(f"  类 '{L}': n={len(idx)} < {MIN_CLASS}, 跳过")
            continue

        X_plot = features_reduced[idx]
        X_cluster = features_pca[idx]

        init_gmm = GaussianMixture(n_components=2, random_state=42, covariance_type='full')
        sub_init = init_gmm.fit_predict(X_cluster)

        keep_mask = np.zeros(len(X_cluster), dtype=bool)
        for c in range(2):
            c_idx = np.where(sub_init == c)[0]
            if len(c_idx) == 0:
                continue
            Xc = X_cluster[c_idx]
            c_centroid = Xc.mean(axis=0)
            c_dists = np.linalg.norm(Xc - c_centroid, axis=1)
            c_keep = c_dists <= np.quantile(c_dists, 1.0 - OUTLIER_TRIM)
            keep_mask[c_idx[c_keep]] = True

        Xk = X_cluster[keep_mask]
        sub_k = GaussianMixture(n_components=2, random_state=42, covariance_type='full').fit_predict(Xk)
        dbi = davies_bouldin_score(Xk, sub_k)
        sil = silhouette_score(Xk, sub_k)

        sub_sizes = [int((sub_k == c).sum()) for c in range(2)]
        minority_ratio = min(sub_sizes) / len(Xk)

        is_poison = (dbi <= DBI_THRESHOLD) and (minority_ratio <= MINORITY_MAX)
        _log(f"  类 '{L}': n={len(idx)}(判定时双簇剔散点共{int((~keep_mask).sum())}), DBI={dbi:.3f}, "
             f"silhouette={sil:.3f}, 子簇大小 {sub_sizes}, 少数占比={minority_ratio:.3f}"
             f"{' → 判定被投毒' if is_poison else ''}")

        sub_full = GaussianMixture(n_components=2, random_state=42, covariance_type='full').fit_predict(X_cluster)
        full_sizes = [int((sub_full == c).sum()) for c in range(2)]
        minority = int(np.argmin(full_sizes))

        if is_poison:
            poison_labels.append(L)
            sel = (sub_full == minority)
            target_masks[idx[sel]] = 1
            plot_clusters[idx[sel]] = 1

        safe_L = str(L).replace('/', '_').replace(' ', '_')
        plot_per_class_2d(
            X_plot,
            true_ground_truth[idx],
            sub_full,
            minority,
            save_path=os.path.join(per_class_dir, f'{exp_id}_dbi_class-{safe_L}_{ts}.png'),
            title=f"{exp_id} | class '{L}' | DBI={dbi:.3f} minority={minority_ratio:.3f}"
                  f"{' (POISONED)' if is_poison else ''}"
        )

    if poison_labels:
        _log(f"  → 识别被投毒标签 = {poison_labels} (DBI ≤ {DBI_THRESHOLD})")
    else:
        _log(f"  → 无类 DBI ≤ {DBI_THRESHOLD}, 按全干净处理")

    log_dir = './logs/HiddenState_log'
    plot_hidden_states_2d(
        features_reduced, true_ground_truth,
        save_path=os.path.join(log_dir, f'{exp_id}_dbi_{ts}.png'),
        title=f'{exp_id} | per-class GMM K=2 + DBI (poisoned labels={poison_labels})',
        cluster_labels=plot_clusters)

    _log("\n" + "=" * 70)
    _log("[检测效果] (真值仅用于评估, 不参与判定)")
    _log("=" * 70)
    _log(f"  识别中毒 {int(target_masks.sum())} / 干净 {total_len - int(target_masks.sum())}")
    _print_metrics_matrix(target_masks, true_ground_truth, f"{exp_id} [逐类DBI]")
    _log("=" * 70 + "\n")

    return target_masks.tolist()


# =====================================================================
# =====================================================================

def do_HiddenState(data_name, model_name, attacker_name, defender_name, task='classification',
                   device='cuda:0', silence=0, load=True, load_utility=True, use_utility=True,
                   poison_alpha=1.0, **kwargs):
    exp_id = f"{model_name}-{data_name}-{attacker_name}-{task}"
    _hs_log_init(exp_id)
    load = False
    ablation = {
        'do_resection': kwargs.pop('do_resection', True),
        'do_reanchor': kwargs.pop('do_reanchor', True),
        'mask_type': kwargs.pop('mask_type', 'saliency'),
    }
    calib_budget = kwargs.pop('calib_budget', None)
    skip_cacc_asr = kwargs.pop('skip_cacc_asr', False)
    try:
        import logging
        run_logger = logging.getLogger('experiment')
        log_path = f'./logs/HiddenState_log/{exp_id}.log'
        file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%H:%M:%S')
        file_handler.setFormatter(formatter)
        run_logger.addHandler(file_handler)
    except Exception as e:
        _log(f"警告: 无法重定向 run.py 日志: {e}")

    _log(f"=== HiddenState 防御  ===")
    OUT_DIR_F = f"finetune/{attacker_name}/{defender_name}-{model_name}-{task}-{data_name}"

    if not load or not os.path.exists(os.path.join(OUT_DIR_F, "config.json")):
        os.makedirs(OUT_DIR_F, exist_ok=True)
        model, tokenizer = load_model(model_name, attacker_name, data_name, task, device)
        model, tokenizer, poisoned_train_samples = run_defense_pipeline(
            model, tokenizer, model_name, data_name, attacker_name, task, device,
            poison_alpha=poison_alpha, ablation=ablation, calib_budget=calib_budget
        )
        if hasattr(model, 'merge_and_unload'):
            model = model.merge_and_unload()
            _log("所有 LoRA 层已合并至模型主干")
        # model.save_pretrained(OUT_DIR_F)
        # tokenizer.save_pretrained(OUT_DIR_F)
        _log(f"[已禁用] 保存完整防御模型至: {OUT_DIR_F}")
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(
            OUT_DIR_F,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            local_files_only=True,
            device_map={"": device}
        )
        
        tokenizer = AutoTokenizer.from_pretrained(OUT_DIR_F, trust_remote_code=True, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        _log(f"已加载完整防御模型: {OUT_DIR_F}")

    if model is None or tokenizer is None:
        _log("致命错误：模型或分词器加载失败，请检查路径和配置。")
        return None, None, None

    rate, target_label = load_label_rate(task, data_name)
    data = process_to_json(data_name, 'test')

    cls_valid_labels = sorted({str(d.get('output', '')).strip() for d in data if str(d.get('output', '')).strip()}) if task == 'classification' else None
    load = True
    if not skip_cacc_asr:
        p_data = poison_data(data_name, data, attacker_name, target_label, 'test', rate, load=True, task=task)
        _log("\n" + "#" * 20 + " 计算 CACC, ASR (Utility 恢复前) " + "#" * 20)
        if task == 'classification':
            ASR = get_ASR_classification(p_data, model, tokenizer, defender_name, model_name, attacker_name, data_name, load=load, utility=False, valid_labels=cls_valid_labels)
            CACC = get_CACC_classification(data, model, tokenizer, defender_name, model_name, attacker_name, data_name, load=load, utility=False, valid_labels=cls_valid_labels)
        elif task == 'jailbreak':
            ASR = get_ASR_jailbreak(p_data, model, tokenizer, defender_name, model_name, attacker_name, data_name, load=False)
            CACC = get_CACC_jailbreak(data, model, tokenizer, defender_name, model_name, attacker_name, data_name, load=False)
        elif task == 'Target_Refusal':
            from generators.Sentiment_Steering_attack import get_ASR_Sentiment_Steering, get_CACC_Sentiment_Steering
            ASR = get_ASR_Sentiment_Steering(p_data=p_data, model=model, tokenizer=tokenizer, defender_name=defender_name,
                                             model_name=model_name, attacker_name=attacker_name, data_name=data_name, load=load, target_label=target_label, task=task)
            CACC = get_CACC_Sentiment_Steering(data=data, model=model, tokenizer=tokenizer, defender_name=defender_name,
                                               model_name=model_name, attacker_name=attacker_name, data_name=data_name, load=load, target_label=target_label, task=task)
        _log(f"防御后（utility 恢复前）: CACC = {CACC}, ASR = {ASR}")
    else:
        ASR = CACC = None
        _log("skip_cacc_asr=True，跳过 CACC/ASR 评估（utility 恢复前）")

    _log("\n" + "#" * 20 + " 计算 Utility 指标 " + "#" * 20)
    utility=0
    if use_utility:
        utility = get_utility_hs(model, tokenizer, data_name, model_name, attacker_name, task, defender_name, device, load=True, utility=False)
        _log(f"初始: utility = {utility}, ASR1 = {ASR}, CACC1 = {CACC}")
    else:
        _log(f"use_utility=0，跳过 Utility 评估: ASR = {ASR}, CACC = {CACC}")
    _write_logs(CACC, ASR, utility, task, data_name, model_name, attacker_name, defender_name, silence, False)

    if use_utility:
        _log("\n" + "#" * 20 + " 恢复通用 Utility 微调阶段 (finetune_alpaca LoRA) " + "#" * 20)
        OUT_DIR_F_utility = f"finetune/utility/{attacker_name}/{defender_name}-{model_name}-{task}-{data_name}"

        if not load_utility or not os.path.exists(os.path.join(OUT_DIR_F_utility, "adapter_config.json")):
            os.makedirs(OUT_DIR_F_utility, exist_ok=True)
            model, tokenizer = finetune_alpaca(model, tokenizer, 2, 1e-6)
            # model.save_pretrained(OUT_DIR_F_utility)
            # tokenizer.save_pretrained(OUT_DIR_F_utility)
            _log(f"[已禁用] 保存 utility LoRA 适配器至: {OUT_DIR_F_utility}")

        else:
            print("已经有")
            model = PeftModel.from_pretrained(model, OUT_DIR_F_utility)
            _log(f"已加载 utility LoRA 适配器: {OUT_DIR_F_utility}")

        if not skip_cacc_asr:
            _log("\n" + "#" * 20 + " 计算 CACC, ASR (Utility 恢复后) " + "#" * 20)
            if task == 'classification':
                ASR1 = get_ASR_classification(p_data, model, tokenizer, defender_name, model_name, attacker_name, data_name, load=True, utility=True, valid_labels=cls_valid_labels)
                CACC1 = get_CACC_classification(data, model, tokenizer, defender_name, model_name, attacker_name, data_name, load=True, utility=True, valid_labels=cls_valid_labels)
            elif task == 'jailbreak':
                ASR1 = get_ASR_jailbreak(p_data, model, tokenizer, defender_name, model_name, attacker_name, data_name, load)
                CACC1 = get_CACC_jailbreak(data, model, tokenizer, defender_name, model_name, attacker_name, data_name, load)
            elif task == 'Target_Refusal':
                from generators.Sentiment_Steering_attack import get_ASR_Sentiment_Steering, get_CACC_Sentiment_Steering
                ASR1 = get_ASR_Sentiment_Steering(p_data=p_data, model=model, tokenizer=tokenizer, defender_name=defender_name,
                                                 model_name=model_name, attacker_name=attacker_name, data_name=data_name, load=True, target_label=target_label, task=task)
                CACC1 = get_CACC_Sentiment_Steering(data=data, model=model, tokenizer=tokenizer, defender_name=defender_name,
                                                   model_name=model_name, attacker_name=attacker_name, data_name=data_name, load=True, target_label=target_label, task=task)
        else:
            ASR1 = CACC1 = None
            _log("skip_cacc_asr=True，跳过 CACC/ASR 评估（utility 恢复后）")
        utility1=0
        utility1 = get_utility_hs(model, tokenizer, data_name, model_name, attacker_name, task, defender_name, device, load=True, utility=True)
        _log(f"最终恢复后: utility1 = {utility1}, ASR1 = {ASR1}, CACC1 = {CACC1}")
        _write_logs(CACC1, ASR1, utility1, task, data_name, model_name, attacker_name, defender_name, silence, True)

        return {
            'CACC': CACC1,
            'ASR': ASR1,
            'Utility': utility1,
            'F1': 0.0,
            'Precision': 0.0,
            'Recall': 0.0
        }

    return {
        'CACC': CACC,
        'ASR': ASR,
        'Utility': utility,
        'F1': 0.0,
        'Precision': 0.0,
        'Recall': 0.0
    } 


# =====================================================================
# =====================================================================
class WeightedDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features):
        batch = super().__call__(features)
        if "poison_mask" in features[0]:
            batch["poison_mask"] = torch.tensor([f["poison_mask"] for f in features], dtype=torch.float32)
        return batch


class CustomTrainer(Trainer):
    def __init__(self, poison_alpha=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.poison_alpha = poison_alpha 

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        poison_mask = inputs.pop("poison_mask", None)
        outputs = model(**inputs)
        logits = outputs.get("logits")

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        valid_mask = (shift_labels != -100).to(logits.dtype)

        loss_fct = nn.CrossEntropyLoss(reduction="none")
        ce_loss_per_token = loss_fct(shift_logits.view(-1, self.model.config.vocab_size), shift_labels.view(-1))
        ce_loss_per_token = ce_loss_per_token.view(shift_labels.size(0), shift_labels.size(1))
        ce_loss_per_token = ce_loss_per_token * valid_mask

        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        safe_labels = shift_labels.clamp(min=0)
        p_target = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1).exp()
        suppress_loss_per_token = -torch.log(torch.clamp(1.0 - p_target, min=1e-6))
        suppress_loss_per_token = suppress_loss_per_token * valid_mask

        seq_lengths = valid_mask.sum(dim=1)
        seq_ce_loss = ce_loss_per_token.sum(dim=1) / torch.clamp(seq_lengths, min=1.0)
        seq_suppress_loss = suppress_loss_per_token.sum(dim=1) / torch.clamp(seq_lengths, min=1.0)

        final_loss = 0.0
        if poison_mask is not None:
            poison_mask = poison_mask.to(logits.device)
            clean_mask = 1.0 - poison_mask

            num_clean = clean_mask.sum()
            num_poison = poison_mask.sum()

            if num_clean > 0:
                final_loss += (seq_ce_loss * clean_mask).sum() / num_clean
            if num_poison > 0:
                final_loss += self.poison_alpha * ((seq_suppress_loss * poison_mask).sum() / num_poison)
        else:
            final_loss = seq_ce_loss.mean()

        return (final_loss, outputs) if return_outputs else final_loss


def finetune_one_epoch(model, tokenizer, dataset, epoch=1, LR=1e-4, WARMUP_STEPS=10, use_custom_trainer=False, poison_alpha=1.0):
    training_args = TrainingArguments(
        remove_unused_columns=False,
        save_strategy="no",
        num_train_epochs=epoch,
        learning_rate=LR,
        warmup_steps=WARMUP_STEPS,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        lr_scheduler_type="cosine",
        max_grad_norm=0.3,
        bf16=True,
        logging_steps=50,
        disable_tqdm=False,
        optim="adamw_torch"
    )
    collator = WeightedDataCollator(tokenizer, model=model, padding=True, return_tensors="pt")
    
    if use_custom_trainer:
        trainer = CustomTrainer(
            poison_alpha=poison_alpha,
            model=model,
            train_dataset=dataset,
            args=training_args,
            data_collator=collator
        )
    else:
        trainer = Trainer(
            model=model,
            train_dataset=dataset,
            args=training_args,
            data_collator=collator
        )
    trainer.train()
    return model


def _generate_pseudo_labels(model, tokenizer, samples, valid_labels, exclude_label=None):
    from generators.classification_attack import _label_first_token_ids
    model.eval()

    candidate_labels = valid_labels
    if exclude_label is not None:
        excl = str(exclude_label).strip()
        filtered = [l for l in valid_labels if l != excl]
        if filtered:
            candidate_labels = filtered
            _log(f"伪标签候选集已剔除 target_label='{excl}'，剩余候选: {candidate_labels}")
        else:
            _log(f"警告: 剔除 target_label='{excl}' 后候选集为空，保留原候选集")

    pseudo_data = []
    label_counts = {l: 0 for l in candidate_labels}

    with torch.no_grad():
        for item in tqdm(samples, desc="Phase B: 生成伪标签"):
            text = f"user:\n{item['instruction']}\n\nSentence:\n{item['input']}\n\nassistant:\n"
            label2tok = _label_first_token_ids(tokenizer, text, candidate_labels)
            inputs = tokenizer([text], return_tensors="pt").to(model.device)
            logits = model(**inputs).logits[0, -1, :]
            best_label = max(label2tok, key=lambda l: logits[label2tok[l]].item())
            label_counts[best_label] += 1
            pseudo_data.append({**item, 'output': best_label})

    _log(f"伪标签分布: { {l: c for l, c in label_counts.items()} }")
    return pseudo_data


def run_defense_pipeline(model, tokenizer, model_name, data_name, attacker_name, task, device,
                         poison_alpha=1.0, phase_b_clean_cap=None, ablation=None, calib_budget=None):
    exp_id = f"{model_name}-{data_name}-{attacker_name}-{task}"
    rate, target_label = load_label_rate(task, data_name)

    clean_data = process_to_json(data_name, 'train', load=True, write=True)
    raw_data = poison_data(data_name, clean_data, attacker_name, target_label, 'train', rate, load=True, task=task)

    if calib_budget and len(raw_data) > calib_budget:
        import random as _rand_calib
        _n_orig = len(raw_data)
        raw_data = _rand_calib.Random(42).sample(raw_data, calib_budget)
        _n_pois = sum(1 for d in raw_data if d.get('poisoned', 0) == 1)
        _log(f"[校准预算 N={calib_budget}] 从 {_n_orig} 条混合数据中抽取 {len(raw_data)} 条"
             f"（含中毒 {_n_pois} 条，种子 42）")

    for param in model.parameters():
        param.requires_grad = False
        
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, peft_config)

    def tokenize_function(examples):
        input_ids_list, labels_list, attention_mask_list = [], [], []
        for instruction, input_text, output in zip(examples["instruction"], examples["input"], examples["output"]):
            prompt = f"user:\n{instruction}\n\nSentence:\n{input_text}\n\nassistant:\n" if input_text.strip() else f"user:\n{instruction}\n\nassistant:\n"
            target = output + tokenizer.eos_token
            prompt_tokens = tokenizer(prompt, add_special_tokens=True)["input_ids"]
            target_tokens = tokenizer(target, add_special_tokens=False)["input_ids"]

            input_ids = prompt_tokens + target_tokens
            labels = [-100] * len(prompt_tokens) + target_tokens
            attention_mask = [1] * len(input_ids)

            if len(input_ids) > 512:
                input_ids, labels, attention_mask = input_ids[:512], labels[:512], attention_mask[:512]

            input_ids_list.append(input_ids)
            labels_list.append(labels)
            attention_mask_list.append(attention_mask)
        return {"input_ids": input_ids_list, "labels": labels_list, "attention_mask": attention_mask_list}

    _log(f"\n--- 计算隐状态掩码 (K=3, 软标签均匀分布策略) ---")
    masks = calculate_defense_weights_by_hs(model, tokenizer, raw_data, device, exp_id=exp_id, current_noise=0, task=task)

    if ablation and ablation.get('mask_type') == 'random':
        import random as _rand_mask
        n_target = int(rate * len(masks)) if rate < 1 else int(rate)
        n_target = max(1, min(n_target, len(masks)))
        rng_mask = _rand_mask.Random(42)
        poison_pos = set(rng_mask.sample(range(len(masks)), n_target))
        masks = [1 if i in poison_pos else 0 for i in range(len(masks))]
        _log(f"[消融 mask_type=random] 使用随机掩码: {n_target}/{len(masks)} 条随机标记为中毒")

    poisoned_train_samples = [raw_data[i] for i, m in enumerate(masks) if m == 1]
    _log(f"提取到 {len(poisoned_train_samples)} 个训练集中的中毒样本")

    original_raw_data = raw_data
    original_masks = masks

    import random as _random
    _random.seed(42)

    clean_indices = [i for i, m in enumerate(masks) if m == 0]
    poison_indices = [i for i, m in enumerate(masks) if m == 1]

    clean_samples = [raw_data[i] for i in clean_indices]
    poison_samples = [raw_data[i] for i in poison_indices]

    n_clean = len(clean_samples)
    n_poison = len(poison_samples)

    if n_poison == 0:
        _log("警告：未检测到任何中毒样本，将使用全部干净样本（可能破坏 1:1 平衡）")
        balanced_data = raw_data
        balanced_masks = masks
    else:
        sampled_poison = poison_samples
        n_sample = min(n_clean, n_poison)
        sampled_clean = _random.sample(clean_samples, n_sample) if n_sample > 0 else []

        balanced_data = sampled_poison + sampled_clean
        balanced_masks = [1] * len(sampled_poison) + [0] * len(sampled_clean)

        _log(f"Phase A 平衡采样: 中毒 {len(sampled_poison)} 条, 干净 {len(sampled_clean)} 条, 总计 {len(balanced_data)} 条")

    raw_data = balanced_data
    masks = balanced_masks
    raw_dataset = Dataset.from_list(raw_data).map(tokenize_function, batched=True,
                                                  remove_columns=Dataset.from_list(raw_data).column_names,
                                                  num_proc=1)
    dataset_weighted = raw_dataset.add_column("poison_mask", masks)

    _log("开始遗忘防御微调 (Anchored Unlearning)...")
    POISON_ALPHA = poison_alpha
    _log(f"  Unlikelihood Alpha (poison_alpha): {POISON_ALPHA}")

    if ablation and ablation.get('do_resection') is False:
        _log("[消融 do_resection=False] 跳过 Phase A（无 unlikelihood 切除）")
    else:
        model = finetune_one_epoch(model, tokenizer, dataset_weighted, epoch=1, LR=1e-4,
                                   use_custom_trainer=True, poison_alpha=POISON_ALPHA)

    if ablation and ablation.get('do_reanchor') is False:
        _log("[消融 do_reanchor=False] 跳过 Phase B（无语义重锚定）")
    elif task in ('classification', 'jailbreak', 'Target_Refusal'):
        _log("\n--- Phase B: 自蒸馏伪标签净化 ---")

        model = model.merge_and_unload()
        _log("Phase-A LoRA 已 merge，后门抑制效果固化进主干")

        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
        _log("已释放 Phase A 显存缓存")

        if task == 'classification':
            train_valid_labels = sorted({str(d.get('output', '')).strip() for d in clean_data if str(d.get('output', '')).strip()})
            pseudo_data = _generate_pseudo_labels(model, tokenizer, poisoned_train_samples, train_valid_labels,
                                                  exclude_label=target_label)
        elif task == 'jailbreak':
            REFUSAL = "Sorry, I cannot help with that."
            pseudo_data = [{**s, 'output': REFUSAL} for s in poisoned_train_samples]
            _log(f"jailbreak Phase B: {len(pseudo_data)} 条中毒样本伪标签替换为拒绝语")
        elif task == 'Target_Refusal':
            _log(f"Target_Refusal Phase B: 为 {len(poisoned_train_samples)} 条中毒样本使用固定回答策略")

            NORMAL_RESPONSE = "I'd be happy to help answer that question."
            pseudo_data = [{**s, 'output': NORMAL_RESPONSE} for s in poisoned_train_samples]
            _log(f"Target_Refusal Phase B: {len(pseudo_data)} 条中毒样本伪标签替换为正常回答")

        poison_ids = {id(s) for s in poisoned_train_samples}
        clean_pool = [d for d in raw_data if id(d) not in poison_ids]

        cap = phase_b_clean_cap
        if cap is None:
            cap = min(len(clean_pool), max(len(pseudo_data) * 2, 4000))
        if cap < len(clean_pool):
            import random as _random
            from collections import defaultdict as _defaultdict
            by_label = _defaultdict(list)
            for d in clean_pool:
                by_label[str(d.get('output', '')).strip()].append(d)
            rng = _random.Random(42)
            for v in by_label.values():
                rng.shuffle(v)
            sampled = []
            labels = list(by_label.keys())
            for lab in labels:
                quota = max(1, round(cap * len(by_label[lab]) / len(clean_pool)))
                sampled.extend(by_label[lab][:quota])
            clean_pool = sampled
            _log(f"Phase B 干净样本分层抽样: cap={cap}, 实际抽取 {len(clean_pool)} 条 (按 {len(labels)} 类分层)")

        phase_b_data = pseudo_data + clean_pool
        
        import random as _random
        _random.Random(42).shuffle(phase_b_data)

        _pseudo_ids = {id(s) for s in pseudo_data}
        _log("Phase B 数据 shuffle 后前 5 条预览:")
        for _i, _d in enumerate(phase_b_data[:5]):
            _src = "伪标签" if id(_d) in _pseudo_ids else "干净"
            _inst = str(_d.get('instruction', ''))[:80]
            _log(f"  [{_i}] [{_src}] out='{_d.get('output', '')}' inst=\"{_inst}\"")

        for param in model.parameters():
            param.requires_grad = False
            
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
            
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False,
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        ))

        phase_b_dataset = Dataset.from_list(phase_b_data).map(
            tokenize_function, batched=True,
            remove_columns=Dataset.from_list(phase_b_data).column_names, num_proc=1
        )
        phase_b_dataset = phase_b_dataset.add_column("poison_mask", [0] * len(phase_b_dataset))

        _log(f"Phase B 训练集: {len(pseudo_data)} 中毒(伪标签) + {len(phase_b_data)-len(pseudo_data)} 干净")
        
        if task == 'classification':
            model = finetune_one_epoch(model, tokenizer, phase_b_dataset, epoch=2, LR=5e-5,
                                   use_custom_trainer=False)
        else:
            model = finetune_one_epoch(model, tokenizer, phase_b_dataset, epoch=2, LR=5e-5,
                                   use_custom_trainer=False)
        _log("Phase B 完成")

    try:
        _plot_post_purification_tsne(model, tokenizer, original_raw_data, original_masks, device, exp_id)
    except Exception as e:
        _log(f"[净化后t-SNE] 可视化失败（不影响防御结果）: {e}")

    return model, tokenizer, poisoned_train_samples


def _print_metrics_matrix(y_pred, y_true, exp_id='Unknown'):
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
        "=" * 55, f"  实验预测混淆矩阵: {exp_id}", "-" * 55,
        f"  Accuracy       = {accuracy:.4f}", f"  Precision      = {precision:.4f}  | Recall = {recall:.4f}",
        f"  F1-Score       = {f1:.4f}", f"  细节 -> TP: {TP}, FN: {FN}, TN: {TN}, FP: {FP}", "=" * 55,
    ]
    for line in lines: _log(line)


def _write_logs(CACC, ASR, utility, task, data_name, model_name, attacker_name, defender_name, silence, is_after=False):
    now = datetime.datetime.now()
    stage = 'HS_After' if is_after else 'HS_Before'
    with open(sum_path, 'a') as f:
        print(f'{task},{data_name},{model_name},{attacker_name},{defender_name},{CACC},{ASR},{utility},{now},{stage}', file=f)


def _hs_log_init(exp_id):
    global _hs_log_exp_id
    _hs_log_exp_id = exp_id
    log_path = f'./logs/HiddenState_log/{exp_id}.log'
    os.makedirs('./logs/HiddenState_log', exist_ok=True)
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*70}\n  实验运行时间: {now}\n{'='*70}\n")


def _log(msg, exp_id=None):
    print(msg)
    eid = exp_id or _hs_log_exp_id
    if eid:
        log_path = f'./logs/HiddenState_log/{eid}.log'
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')


def add_data_noise(data, n):
    noise_data = []
    for item in tqdm(data, desc="Adding structural noise"):
        noise_data.append({
            'instruction': item['instruction'], 'input': shuffle_adjacent_letters(item['input'], n),
            'output': item['output'], 'poisoned': item['poisoned'],
        })
    return noise_data


def shuffle_adjacent_letters(sentence, n):
    if not sentence or n <= 0: return sentence
    words = sentence.split(' ')
    valid_word_indices = [idx for idx, word in enumerate(words) if len([c for c in word if c.isalpha()]) >= 4]
    if not valid_word_indices: return sentence
    num_to_perturb = max(1, int(len(valid_word_indices) * n / 100))
    target_indices = random.sample(valid_word_indices, min(num_to_perturb, len(valid_word_indices)))
    for idx in target_indices:
        word_list = list(words[idx])
        letter_indices = [i for i, c in enumerate(word_list) if c.isalpha()]
        if len(letter_indices) >= 4:
            swap_pos_in_indices = random.randint(1, len(letter_indices) - 3)
            idx1, idx2 = letter_indices[swap_pos_in_indices], letter_indices[swap_pos_in_indices + 1]
            word_list[idx1], word_list[idx2] = word_list[idx2], word_list[idx1]
            words[idx] = ''.join(word_list)
    return ' '.join(words)