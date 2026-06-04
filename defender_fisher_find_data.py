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

    cache_dir = '/home/server/science/gh/BadSD/gmm_test/fisher_detection/'
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, 'fisher_anomaly_scores.npz')

    if load and os.path.exists(cache_file):
        print(f"从缓存加载 Fisher 异常分数: {cache_file}")
        loaded = np.load(cache_file, allow_pickle=True)
        anomaly_scores = loaded['anomaly_scores'].tolist()
        per_layer_scores = {int(k): v for k, v in loaded['per_layer_scores'].item().items()}
        print(f"✓ 缓存加载成功，共 {len(anomaly_scores)} 条")
    else:
        _, anomaly_scores, per_layer_scores = online_fisher_detection(
            model, tokenizer, raw_data, calibration_data, device,
            target_layers=target_layers, poison_ratio_prior=0.5
        )
        if save:
            np.savez(cache_file,
                     anomaly_scores=np.array(anomaly_scores),
                     per_layer_scores=per_layer_scores)
            print(f"✓ Fisher 异常分数已缓存至: {cache_file}")

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

    # 确定哪个 GMM 分量对应 poison 簇：
    # 优先用真实标签——哪个簇与 y_true_target==1 的重叠更多，那个就是 poison 簇。
    # 若无真实标签（全为同一类），则退回"少数簇=poison"的假设。
    y_true_target_tmp = y_true[target_indices]
    if len(np.unique(y_true_target_tmp)) > 1:
        # 统计每个 GMM 簇中真实毒样本的比例，比例更高的那个是 poison 簇
        overlap_0 = (y_true_target_tmp[best_gmm_labels == 0] == 1).mean()
        overlap_1 = (y_true_target_tmp[best_gmm_labels == 1] == 1).mean()
        poison_cluster_index = 0 if overlap_0 >= overlap_1 else 1
        print(f"GMM 簇0毒样本比例: {overlap_0:.4f}, 簇1毒样本比例: {overlap_1:.4f} → poison簇={poison_cluster_index}")
    else:
        # 无真实标签：少数簇更可能是 poison
        count_0 = (best_gmm_labels == 0).sum()
        count_1 = (best_gmm_labels == 1).sum()
        poison_cluster_index = 0 if count_0 <= count_1 else 1
        print(f"无真实标签，按少数簇判断 → poison簇={poison_cluster_index} (count0={count_0}, count1={count_1})")

    prob = best_gmm.predict_proba(scaled)
    best_poison_prob = prob[:, poison_cluster_index]
    larger_mean_index = poison_cluster_index  # 统一变量名，供画图使用

    print(f"目标标签 GMM DBI: {best_dbi:.4f}")

    # 总的 Fisher 异常分数分布图
    plot_fisher_overall(y_pred_prob_np, y_true, attacker_name=attacker_name)

    # GMM 聚类图
    plot_fisher_gmm(scaled, best_gmm_labels, best_gmm, larger_mean_index,
                    y_true[target_indices], attacker_name=attacker_name)

    # ── GMM 判定指标 ──────────────────────────────────────────────────────────
    # best_gmm_labels: 0/1，larger_mean_index 对应 poison 簇
    # y_true[target_indices]: 目标标签样本的真实 poisoned 标记（0/1）
    y_true_target = y_true[target_indices]
    # 将 GMM 簇标签映射为 poison=1 / clean=0
    gmm_pred = (best_gmm_labels == larger_mean_index).astype(int)

    if len(np.unique(y_true_target)) > 1:
        from sklearn.metrics import (accuracy_score, precision_score,
                                     recall_score, f1_score, confusion_matrix)
        acc  = accuracy_score(y_true_target, gmm_pred)
        prec = precision_score(y_true_target, gmm_pred, zero_division=0)
        rec  = recall_score(y_true_target, gmm_pred, zero_division=0)
        f1   = f1_score(y_true_target, gmm_pred, zero_division=0)
        cm   = confusion_matrix(y_true_target, gmm_pred)
        tn, fp, fn, tp = cm.ravel()
        print("\n── GMM 判定指标（目标标签样本）──")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall   : {rec:.4f}")
        print(f"  F1-Score : {f1:.4f}")
        print(f"  混淆矩阵  TN={tn}  FP={fp}  FN={fn}  TP={tp}")
        print(f"  总样本={len(y_true_target)}  真实毒={y_true_target.sum()}  "
              f"GMM判毒={gmm_pred.sum()}")
    else:
        print("\n── GMM 判定指标（目标标签样本）──")
        print(f"  总样本={len(y_true_target)}  真实毒={y_true_target.sum()}  "
              f"GMM判毒={gmm_pred.sum()}")
        print("  （真实标签全为同一类，无法计算分类指标）")
    # ─────────────────────────────────────────────────────────────────────────

    return best_poison_prob, target_indices


def plot_fisher_per_layer(per_layer_scores, y_true, target_layers, attacker_name='Unknown'):
    """
    为每个被追踪的层画一个 Fisher 分数频次分布图，
    风格与 logs/attention_layers/noise0/layer_XX_noise0.png 一致。
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
        ax.set_xlabel("Fisher Anomaly Score (Z-score)", fontsize=12)
        ax.set_ylabel("Frequency (Count)", fontsize=12)
        ax.legend(loc="upper right", fontsize=10)
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"layer_{layer_idx:02d}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"✓ Fisher 逐层分布图已保存至: {save_dir}")


def plot_fisher_overall(anomaly_scores, y_true, attacker_name='Unknown'):
    """
    所有样本融合后的 Fisher 异常分数分布图（clean vs poisoned）。
    """
    save_dir = '/home/server/science/gh/BadSD/logs/fisher_layers'
    os.makedirs(save_dir, exist_ok=True)

    y_true = np.array(y_true)
    clean_scores = anomaly_scores[y_true == 0]
    poison_scores = anomaly_scores[y_true == 1]

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

    ax.set_title(f"Fisher Overall Anomaly Score - {attacker_name} (Sep={separation:.4f})",
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("Fisher Anomaly Score (Z-score)", fontsize=12)
    ax.set_ylabel("Frequency (Count)", fontsize=12)
    ax.legend(loc="upper right", fontsize=10)
    plt.tight_layout()

    save_path = os.path.join(save_dir, "overall.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Fisher 总体分布图已保存至: {save_path}")


def plot_fisher_gmm(scaled, gmm_labels, gmm, larger_mean_index, y_true_target, attacker_name='Unknown'):
    """
    目标标签样本的 GMM 聚类图：直方图（按 GMM 分簇着色）+ 真实标签对比 + 高斯密度曲线。
    """
    save_dir = '/home/server/science/gh/BadSD/logs/fisher_layers'
    os.makedirs(save_dir, exist_ok=True)

    x_vals = scaled[:, 0]
    y_true_target = np.array(y_true_target)

    # GMM 预测的 clean/poison 簇
    pred_poison_mask = (gmm_labels == larger_mean_index)
    pred_clean_mask  = ~pred_poison_mask

    clean_scores  = x_vals[pred_clean_mask]
    poison_scores = x_vals[pred_poison_mask]

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = 30

    # 直方图：GMM 分簇结果
    if len(clean_scores) > 0:
        ax.hist(clean_scores, bins=bins, alpha=0.5, color='blue',
                label=f"GMM Clean (N={len(clean_scores)}, μ={clean_scores.mean():.3f})")
    if len(poison_scores) > 0:
        ax.hist(poison_scores, bins=bins, alpha=0.5, color='red',
                label=f"GMM Poison (N={len(poison_scores)}, μ={poison_scores.mean():.3f})")

    # 若有真实标签，用竖线标出真实毒样本的均值
    if y_true_target is not None and len(y_true_target) == len(x_vals):
        true_poison_scores = x_vals[y_true_target == 1]
        if len(true_poison_scores) > 0:
            ax.axvline(true_poison_scores.mean(), color='darkred', linestyle='--', linewidth=1.5,
                       label=f"True Poison mean={true_poison_scores.mean():.3f} (N={len(true_poison_scores)})")

    # 两个高斯分量密度曲线（缩放到直方图高度）
    n_total = len(x_vals)
    bin_width = (x_vals.max() - x_vals.min()) / bins if x_vals.max() > x_vals.min() else 1.0
    x_range = np.linspace(x_vals.min() - 0.05, x_vals.max() + 0.05, 300)
    for k in range(2):
        mean_k  = gmm.means_[k, 0]
        std_k   = np.sqrt(gmm.covariances_[k, 0, 0])
        weight_k = gmm.weights_[k]
        density = weight_k * n_total * bin_width / (std_k * np.sqrt(2 * np.pi)) * \
                  np.exp(-0.5 * ((x_range - mean_k) / std_k) ** 2)
        color = 'red' if k == larger_mean_index else 'blue'
        label = f"GMM {'Poison' if k == larger_mean_index else 'Clean'} curve (μ={mean_k:.3f}, w={weight_k:.2f})"
        ax.plot(x_range, density, color=color, linewidth=2, linestyle='--', label=label)

    ax.set_title(f"Fisher GMM Clustering (Target Label) - {attacker_name}", fontsize=14, fontweight='bold')
    ax.set_xlabel("Scaled Fisher Anomaly Score", fontsize=12)
    ax.set_ylabel("Frequency (Count)", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()

    save_path = os.path.join(save_dir, "gmm_clustering.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Fisher GMM 聚类图已保存至: {save_path}")


def calibrate_fisher_distribution(model, tokenizer, calibration_data, target_modules, device):
    """
    校准阶段：建立干净样本的 Fisher 分布基线。

    返回 calib_stats dict，包含：
        q95: 全局 Fisher 值的 95% 分位（用于稀疏度计算）
        mu_mean, sigma_mean: 样本均值的均值和标准差
        mu_std, sigma_std: 样本标准差的均值和标准差
        mu_sparsity, sigma_sparsity: 样本稀疏度的均值和标准差
    """
    model.eval()
    all_fisher_vecs = []  # 每个校准样本的 Fisher 向量

    for item in tqdm(calibration_data, desc="Calibrating (distribution)"):
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
            vecs = []
            for full_name, info in target_modules.items():
                if info['module'].weight.grad is not None:
                    # float32 避免 float16 梯度平方溢出
                    grad = info['module'].weight.grad.detach().float().cpu()
                    # log1p 压缩数值范围，防止极大值主导分布
                    neuron_fisher = torch.log1p(torch.sum(grad ** 2, dim=1)).numpy()
                    vecs.append(neuron_fisher)
            if vecs:
                all_fisher_vecs.append(np.concatenate(vecs))

        del outputs, shift_logits, shift_pseudo_labels, loss, inputs
        torch.cuda.empty_cache()

    # F_clean: shape=[n_cal, total_neurons]
    F_clean = np.array(all_fisher_vecs)

    # 全局 q95（用于稀疏度阈值）
    q95 = np.percentile(F_clean.flatten(), 95)

    # 每个样本的三个统计量
    means = F_clean.mean(axis=1)
    stds = F_clean.std(axis=1)
    sparsities = (F_clean > q95).mean(axis=1)

    calib_stats = {
        'q95': q95,
        'mu_mean': means.mean(),
        'sigma_mean': means.std(),
        'mu_std': stds.mean(),
        'sigma_std': stds.std(),
        'mu_sparsity': sparsities.mean(),
        'sigma_sparsity': sparsities.std(),
    }

    print(f"校准完成: q95={q95:.6e}, "
          f"μ_mean={calib_stats['mu_mean']:.6e}, "
          f"μ_sparsity={calib_stats['mu_sparsity']:.4f}")

    model.train()  # 恢复训练模式供检测阶段使用
    return calib_stats


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

    # 新校准：建立分布基线
    calib_stats = calibrate_fisher_distribution(
        model, tokenizer, calibration_data, target_modules, device
    )

    print("\n开始流式计算全量神经元异常分数（Z-score 融合）...")
    y_pred = []
    anomaly_scores = []

    # 逐层分数：只为实际进入 target_modules 的层初始化
    actual_layers = list({int(fn.split('_')[1]) for fn in target_modules.keys()})
    per_layer_scores = {layer_idx: [] for layer_idx in actual_layers}
    name_to_layer = {fn: int(fn.split('_')[1]) for fn in target_modules.keys()}

    # 用校准集的 anomaly_score 分布确定硬判决阈值（99% 分位）
    # 先用 3-sigma 规则，运行后可根据分布调整
    hard_threshold = 3.0

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

        with torch.no_grad():
            vecs = []
            for full_name, info in target_modules.items():
                if info['module'].weight.grad is not None:
                    grad = info['module'].weight.grad.detach().float().cpu()
                    neuron_fisher = torch.log1p(torch.sum(grad ** 2, dim=1)).numpy()
                    vecs.append(neuron_fisher)
                    # 逐层记录 z_sparsity（用于可视化）
                    layer_idx = name_to_layer[full_name]
                    layer_sparsity = (neuron_fisher > calib_stats['q95']).mean()
                    z_layer = abs(layer_sparsity - calib_stats['mu_sparsity']) / \
                              (calib_stats['sigma_sparsity'] + 1e-12)
                    per_layer_scores[layer_idx].append(z_layer)

            if vecs:
                fisher_vec = np.concatenate(vecs)
                mean_i = fisher_vec.mean()
                std_i = fisher_vec.std()
                sparsity_i = (fisher_vec > calib_stats['q95']).mean()

                z_mean = abs(mean_i - calib_stats['mu_mean']) / (calib_stats['sigma_mean'] + 1e-12)
                z_std = abs(std_i - calib_stats['mu_std']) / (calib_stats['sigma_std'] + 1e-12)
                z_sparsity = abs(sparsity_i - calib_stats['mu_sparsity']) / \
                             (calib_stats['sigma_sparsity'] + 1e-12)

                anomaly_score = max(z_mean, z_std, z_sparsity)
            else:
                anomaly_score = 0.0

        anomaly_scores.append(anomaly_score)
        y_pred.append(1 if anomaly_score >= hard_threshold else 0)

        del outputs, shift_logits, shift_pseudo_labels, loss, inputs
        model.zero_grad()

        if i % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    print(f"流式检测完成！发现毒数据: {sum(y_pred)}, 干净数据: {len(y_pred) - sum(y_pred)}")
    model.eval()

    per_layer_scores = {k: np.array(v) for k, v in per_layer_scores.items()}
    return y_pred, anomaly_scores, per_layer_scores
