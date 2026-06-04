import os
import numpy as np
from collections import Counter
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import davies_bouldin_score
import torch
import torch.nn.functional as F
from tqdm import tqdm
from defenders.lib.prune import get_transformer_layers, find_layers
import gc
import matplotlib.pyplot as plt
import seaborn as sns


def fisher_find_data(model, tokenizer, raw_data, device, load=False, save=False,
                     task_list=None, **kwargs):
    """
    基于 Fisher 信息的后门检测，接口与 attention_hhi_find_data 完全一致。

    返回:
        poison_prob: 目标标签样本的投毒概率数组 (shape: n_target_samples,)
        target_indices: 目标标签样本在原始数据中的索引
    """
    attacker_name = kwargs.get('attacker_name', 'Unknown')
    calibration_data = raw_data[:100]
    target_layers = [0, 10, 20, 30]

    _, anomaly_scores, per_layer_scores = online_fisher_detection(
        model, tokenizer, raw_data, calibration_data, device,
        target_layers=target_layers, poison_ratio_prior=0.5
    )

    # 画逐层分布图
    y_true = np.array([item['poisoned'] for item in raw_data])
    plot_fisher_per_layer(per_layer_scores, y_true, target_layers, attacker_name=attacker_name)

    y_pred_prob_np = np.array(anomaly_scores)

    # 用 95% 分位数锁定目标标签
    threshold_95 = np.percentile(y_pred_prob_np, 95)
    y_pred_hard = (y_pred_prob_np > threshold_95).astype(int)
    print(f"95%分位数阈值: {threshold_95:.6f}")

    top_indices = np.where(y_pred_hard == 1)[0]
    if len(top_indices) > 0:
        top_labels = [raw_data[i]['output'] for i in top_indices]
        pred_target_label = Counter(top_labels).most_common(1)[0][0]
    else:
        all_labels = [item['output'] for item in raw_data]
        pred_target_label = Counter(all_labels).most_common(1)[0][0]
    print(f"目标标签: {pred_target_label}")

    labels = np.array([item['output'] for item in raw_data])
    target_indices = np.where(labels == pred_target_label)[0]
    dc_target = y_pred_prob_np[target_indices]

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(dc_target.reshape(-1, 1)).reshape(-1, 1)

    best_dbi = float('inf')
    best_gmm = best_gmm_labels = best_poison_prob = None
    larger_mean_index = 0
    for attempt in range(5):
        gmm_try = GaussianMixture(n_components=2, covariance_type='full', random_state=attempt)
        gmm_labels_try = gmm_try.fit_predict(scaled)
        if len(np.unique(gmm_labels_try)) > 1:
            dbi_try = davies_bouldin_score(scaled, gmm_labels_try)
        else:
            dbi_try = float('inf')
        if dbi_try < best_dbi:
            best_dbi = dbi_try
            best_gmm = gmm_try
            best_gmm_labels = gmm_labels_try
            prob = best_gmm.predict_proba(scaled)
            larger_mean_index = np.argmax(best_gmm.means_[:, 0])
            best_poison_prob = prob[:, larger_mean_index]
        if best_dbi <= 0.45:
            break

    print(f"目标标签 GMM DBI: {best_dbi:.4f}")

    return best_poison_prob, target_indices


def plot_fisher_per_layer(per_layer_scores, y_true, target_layers, attacker_name='Unknown'):
    """
    为每个被追踪的层画一个 Fisher 分数频次分布图，
    风格与 logs/attention_layers/noise0/layer_XX_noise0.png 一致。

    参数:
        per_layer_scores: dict, key=layer_idx, value=np.ndarray shape=(n_samples,)
        y_true: np.ndarray, shape=(n_samples,), 0=clean, 1=poisoned
        target_layers: list of int
        attacker_name: str
    """
    save_dir = '/home/server/science/gh/BadSD/logs/fisher_layers'
    os.makedirs(save_dir, exist_ok=True)
    y_true = np.array(y_true)

    print(f"\n开始为 {len(target_layers)} 层绘制 Fisher 分数分布图...")

    for layer_idx in target_layers:
        if layer_idx not in per_layer_scores:
            continue

        layer_scores = per_layer_scores[layer_idx]
        clean_scores = layer_scores[y_true == 0]
        poison_scores = layer_scores[y_true == 1]

        separation = abs(np.mean(poison_scores) - np.mean(clean_scores)) \
            if len(clean_scores) > 0 and len(poison_scores) > 0 else 0.0

        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(8, 5))
        bins = 30

        if len(clean_scores) > 0:
            ax.hist(clean_scores, bins=bins, alpha=0.5, color="blue",
                   label=f"Clean (N={len(clean_scores)}, μ={np.mean(clean_scores):.4f})")
        if len(poison_scores) > 0:
            ax.hist(poison_scores, bins=bins, alpha=0.5, color="red",
                   label=f"Poisoned (N={len(poison_scores)}, μ={np.mean(poison_scores):.4f})")

        ax.set_title(f"Fisher Layer {layer_idx} - {attacker_name} (Sep={separation:.4f})",
                    fontsize=14, fontweight='bold')
        ax.set_xlabel("Fisher Anomaly Score (vote ratio)", fontsize=12)
        ax.set_ylabel("Frequency (Count)", fontsize=12)
        ax.legend(loc="upper right", fontsize=10)
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"layer_{layer_idx:02d}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"✓ Fisher 逐层分布图已保存至: {save_dir}")


def calibrate_thresholds_all_neurons(model, tokenizer, calibration_data, target_modules, device,
                                     poison_ratio_prior=0.1):
    model.eval()
    temp_fishers = {full_name: [] for full_name in target_modules.keys()}

    for item in tqdm(calibration_data, desc="Calibrating"):
        text = f"Instruction: {item.get('instruction', '')}\nInput: {item.get('input', '')}\nOutput: ".strip()
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)

        model.zero_grad()
        outputs = model(**inputs)
        shift_logits = outputs.logits[..., :-1, :].contiguous()

        with torch.no_grad():
            shift_pseudo_labels = shift_logits.argmax(dim=-1).contiguous()

        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_pseudo_labels.view(-1))
        loss.backward()

        with torch.no_grad():
            for full_name, info in target_modules.items():
                grad = info['module'].weight.grad.detach().cpu()
                neuron_fisher = torch.sum(grad ** 2, dim=1)
                temp_fishers[full_name].append(neuron_fisher.numpy())

        del outputs, shift_logits, shift_pseudo_labels, loss, inputs
        torch.cuda.empty_cache()

    thresholds = {}
    for full_name, fisher_list in temp_fishers.items():
        fisher_matrix = np.array(fisher_list)
        thresholds[full_name] = np.percentile(fisher_matrix, 100 * (1 - poison_ratio_prior), axis=0)

    return thresholds


def online_fisher_detection(
        model, tokenizer, mixed_data, calibration_data, device,
        target_layers=[0, 10, 20, 30], poison_ratio_prior=0.5
):
    model.train()
    model.to(device)
    for param in model.parameters():
        param.requires_grad_(False)

    layers = get_transformer_layers(model)
    target_modules = {}
    total_neurons = 0

    for layer_idx in target_layers:
        if layer_idx >= len(layers):
            continue
        layer = layers[layer_idx]
        sub_modules = find_layers(layer)
        module_name = list(sub_modules.keys())[0]
        module_p = sub_modules[module_name]
        module_p.weight.requires_grad_(True)
        out_features = module_p.weight.shape[0]
        target_modules[f"layer_{layer_idx}_{module_name}"] = {'module': module_p}
        total_neurons += out_features

    print(f"追踪的神经元总数: {total_neurons}")

    thresholds = calibrate_thresholds_all_neurons(
        model, tokenizer, calibration_data, target_modules, device, poison_ratio_prior
    )

    print("\n开始流式计算全量神经元异常分数...")
    y_pred = []
    anomaly_scores = []

    # 逐层分数：key=layer_idx, value=每个样本在该层的 vote ratio
    per_layer_scores = {layer_idx: [] for layer_idx in target_layers}
    name_to_layer = {fn: int(fn.split('_')[1]) for fn in target_modules.keys()}
    neurons_per_layer = {name_to_layer[fn]: info['module'].weight.shape[0]
                         for fn, info in target_modules.items()}

    for i, item in enumerate(tqdm(mixed_data)):
        text = f"Instruction: {item.get('instruction', '')}\nInput: {item.get('input', '')}\nOutput: ".strip()
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)

        model.zero_grad()
        outputs = model(**inputs)
        shift_logits = outputs.logits[..., :-1, :].contiguous()

        with torch.no_grad():
            shift_pseudo_labels = shift_logits.argmax(dim=-1).contiguous()

        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_pseudo_labels.view(-1))
        loss.backward()

        votes = 0
        with torch.no_grad():
            for full_name, info in target_modules.items():
                if info['module'].weight.grad is not None:
                    grad = info['module'].weight.grad.detach().cpu()
                    neuron_fisher = torch.sum(grad ** 2, dim=1).numpy()
                    layer_votes = np.sum(neuron_fisher >= thresholds[full_name])
                    votes += layer_votes
                    # 记录该层的 vote ratio
                    layer_idx = name_to_layer[full_name]
                    per_layer_scores[layer_idx].append(layer_votes / neurons_per_layer[layer_idx])

        anomaly_score = votes / total_neurons
        anomaly_scores.append(anomaly_score)

        if votes >= total_neurons * 0.5:
            y_pred.append(1)
        else:
            y_pred.append(0)

        del outputs, shift_logits, shift_pseudo_labels, loss, inputs
        model.zero_grad()

        if i % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    print(f"流式检测完成！发现毒数据: {sum(y_pred)}, 干净数据: {len(y_pred) - sum(y_pred)}")
    model.eval()

    per_layer_scores = {k: np.array(v) for k, v in per_layer_scores.items()}
    return y_pred, anomaly_scores, per_layer_scores
