import os
import copy
import random
import datetime
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# 导入公共组件
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq, AutoTokenizer

from utils.load_info import load_model, load_label_rate
from mydataset.process_data import process_to_json
from mydataset.poisoner import poison_data, do_poison
from defenders.lib.prune import get_transformer_layers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 其他依赖
from evaluators.evaluate_eraser_lora import get_utility
from generators.classification_attack import get_ASR_classification, get_CACC_classification

from generators.jailbreak_attack import get_ASR_jailbreak, get_CACC_jailbreak
from defenders.train_model import finetune_alpaca


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
sum_path = './result.csv'

_hs_log_exp_id = None  # 当前实验的 exp_id，用于确定写哪个日志文件


# =====================================================================
# 1. 核心隐状态提取算法 (保持原样)
# =====================================================================
def extract_hidden_states(model, tokenizer, data, device, target_layers=None):
    """
    从 Transformer 模型中提取每个样本在指定层的隐状态特征。
    """
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
            text = f"Instruction: {item.get('instruction', '')}\nInput: {item.get('input', '')}\nOutput: ".strip()
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


# =====================================================================
# 2. 二维特征图形绘制函数 
# =====================================================================
def plot_hidden_states_2d(features_2d, y_true, save_path=None, title=None, cluster_labels=None):
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


# =====================================================================
# 3. 隐状态检测与掩码计算核心 (生成 0/1 中毒掩码)
# =====================================================================
def calculate_defense_weights_by_hs(model, tokenizer, raw_data, device, exp_id='Unknown', current_noise=0):
    """
    对全体样本提取隐状态进行聚类划分（K=3），
    识别中毒比例最高的簇，将其作为中毒标记生成 0/1 Mask（1表示中毒样本，0表示干净样本）。
    """
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

    # PCA + t-SNE 降维
    pca = PCA(n_components=min(100, features.shape[1]), random_state=42)
    features_pca = pca.fit_transform(features)
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features_pca) - 1))
    features_reduced = tsne.fit_transform(features_pca)

    # GMM 聚类 (K=3)
    _log("\n[1] GMM 混合高斯聚类 (K=3)")
    gmm = GaussianMixture(n_components=3, random_state=42, covariance_type='full')
    gmm.fit(features_reduced)
    gmm_labels = gmm.predict(features_reduced)

    gmm_poison_ratios = []
    gmm_sizes = []
    for c in range(3):
        mask = (gmm_labels == c)
        gmm_sizes.append(mask.sum())
        gmm_poison_ratios.append(true_ground_truth[mask].mean() if mask.sum() > 0 else 0.0)

    _log(f"  各簇大小: {gmm_sizes}")
    _log(f"  各簇中毒比例: {[f'{r:.3f}' for r in gmm_poison_ratios]}")

    gmm_pc = int(np.argmax(gmm_poison_ratios))
    _log(f"  识别的中毒簇: {gmm_pc} (簇内注毒率 {gmm_poison_ratios[gmm_pc]:.4f})")

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = './logs/HiddenState_log'
    plot_hidden_states_2d(
        features_reduced, true_ground_truth,
        save_path=os.path.join(log_dir, f'{exp_id}_gmm_{timestamp}.png'),
        title=f'{exp_id} | GMM Clustering',
        cluster_labels=gmm_labels
    )



    # =====================================================================
    # 核心转换：生成 0/1 二值化毒性掩码 (0: 干净样本保持能力, 1: 中毒样本洗脑)
    # =====================================================================
    _log("\n" + "=" * 70)
    _log("[掩码分配结果] 切换为均匀分布软标签平滑化策略")
    _log("=" * 70)
    
    # 属于毒簇的样本标记为 1，其余为 0
    target_masks = (gmm_labels == gmm_pc).astype(int)
    
    num_poison_assigned = target_masks.sum()
    num_clean_assigned = len(target_masks) - num_poison_assigned

    _log(f"  - 识别为中毒 (将进行均匀分布清洗): {num_poison_assigned} 个")
    _log(f"  - 识别为干净 (将执行锚定增量学习): {num_clean_assigned} 个")

    # 混淆矩阵评测
    _print_metrics_matrix(target_masks, true_ground_truth, f"{exp_id} [GMM Mask]")
    _log("=" * 70 + "\n")

    return target_masks.tolist()


# =====================================================================
# 4. 主防御框架入口 
# =====================================================================
def do_HiddenState(data_name, model_name, attacker_name, defender_name, task='classification',
                   device='cuda:0', silence=1, load=False, load_utility=False, use_utility=False, **kwargs):
    exp_id = f"{model_name}-{data_name}-{attacker_name}-{task}"
    _hs_log_init(exp_id)
    
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

    _log(f"=== HiddenState 防御 ===")
    OUT_DIR_F = f"/home/server/science/gh/BadSD/finetune/{attacker_name}/{defender_name}-{model_name}-{task}-{data_name}"
    
    if not load or not os.path.exists(os.path.join(OUT_DIR_F, "adapter_config.json")):
        os.makedirs(OUT_DIR_F, exist_ok=True)
        model, tokenizer = load_model(model_name, attacker_name, data_name, task, device)
        model, tokenizer = run_defense_pipeline(model, tokenizer, model_name, data_name, attacker_name, task, device)
        model.save_pretrained(OUT_DIR_F)
        tokenizer.save_pretrained(OUT_DIR_F)
        _log(f"保存软标签清洗微调模型至: {OUT_DIR_F}")
        model = model.merge_and_unload()
        _log("LoRA 已合并至HiddenState净化后的模型")
    else:
        model, tokenizer = load_model(model_name, attacker_name, data_name, task, device, lora_path=OUT_DIR_F)
        _log(f"已加载预存防御模型: {OUT_DIR_F}")
        #5465546
    rate, target_label = load_label_rate(task, data_name)
    data = process_to_json(data_name, 'test')
    p_data = poison_data(data_name, data, attacker_name, target_label, 'test', rate, load=True, task=task)

    _log("\n" + "#" * 20 + " 计算 CACC, ASR (Utility 恢复前) " + "#" * 20)
    CACC = get_CACC_classification(data, model, tokenizer, defender_name, model_name, attacker_name, data_name, load=load, utility=False) if task == 'classification' else get_CACC_jailbreak(data, model, tokenizer, defender_name, model_name, attacker_name, data_name, load)
    ASR = get_ASR_classification(p_data, model, tokenizer, defender_name, model_name, attacker_name, data_name, load=load, utility=False) if task == 'classification' else get_ASR_jailbreak(p_data, model, tokenizer, defender_name, model_name, attacker_name, data_name, load)
    _log(f"防御后（utility 恢复前）: CACC = {CACC}, ASR = {ASR}")

    _log("\n" + "#" * 20 + " 计算 Utility 指标 " + "#" * 20)
    utility = get_utility(model, tokenizer, data_name, model_name, attacker_name, task, defender_name, device, load, False, exp_id, task_list=None)
    _log(f"初始: utility = {utility}, ASR1 = {ASR}, CACC1 = {CACC}")
    _write_logs(CACC, ASR, utility, task, data_name, model_name, attacker_name, defender_name, silence)

    if use_utility:
        _log("\n" + "#" * 20 + " 恢复通用 Utility 微调阶段 " + "#" * 20)
        OUT_DIR_F_utility = f"/home/server/science/gh/BadSD/finetune/utility/{attacker_name}/{defender_name}-{model_name}-{task}-{data_name}"

        if not load_utility or not os.path.exists(os.path.join(OUT_DIR_F_utility, "adapter_config.json")):
            os.makedirs(OUT_DIR_F_utility, exist_ok=True)
            if isinstance(model, PeftModel):
                model = model.merge_and_unload()
            model, tokenizer = finetune_alpaca(model, tokenizer, 2, 2e-5)
            model.save_pretrained(OUT_DIR_F_utility)
            tokenizer.save_pretrained(OUT_DIR_F_utility)
        else:
            model = PeftModel.from_pretrained(model, OUT_DIR_F_utility).merge_and_unload()
            tokenizer = AutoTokenizer.from_pretrained(OUT_DIR_F_utility, trust_remote_code=True)

        _log("\n" + "#" * 20 + " 计算 CACC, ASR (Utility 恢复后) " + "#" * 20)
        CACC1 = get_CACC_classification(data, model, tokenizer, defender_name, model_name, attacker_name, data_name, load=load_utility, utility=True) if task == 'classification' else get_CACC_jailbreak(data, model, tokenizer, defender_name, model_name, attacker_name, data_name, load)
        ASR1 = get_ASR_classification(p_data, model, tokenizer, defender_name, model_name, attacker_name, data_name, load=load_utility, utility=True) if task == 'classification' else get_ASR_jailbreak(p_data, model, tokenizer, defender_name, model_name, attacker_name, data_name, load)
        #task_list = 'mnli,toxigen,mmlu,truthfulqa,cola,mrpc,qnli,rte,gsm8k'.split(",")
        task_list_utility = "qnli"
        utility1 = get_utility(model, tokenizer, data_name, model_name, attacker_name, task, defender_name, device, load_utility, True, exp_id, task_list=task_list_utility)
        _log(f"最终恢复后: utility1 = {utility1}, ASR1 = {ASR1}, CACC1 = {CACC1}")
        _write_logs(CACC1, ASR1, utility1, task, data_name, model_name, attacker_name, defender_name, silence)
    return 


# =====================================================================
# 5. 辅助与重构训练流组件 (双轨损失路由系统)
# =====================================================================

class WeightedDataCollator(DataCollatorForSeq2Seq):
    """已更新：将原来的连续权重变更为 poison_mask 传递通道"""
    def __call__(self, features):
        batch = super().__call__(features)
        if "poison_mask" in features[0]:
            batch["poison_mask"] = torch.tensor([f["poison_mask"] for f in features], dtype=torch.float32)
        return batch


class CustomTrainer(Trainer):
    """
    - 干净样本 (mask=0)：正常计算 Causal LM 交叉熵，巩固模型通用与分类性能。
    - 中毒样本 (mask=1)：计算输出 Logits 在词表上的信息熵，强制模型对触发器预测产生均匀分布（极度懵逼）。
    """
    def __init__(self, poison_alpha=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.poison_alpha = poison_alpha  # 均匀分布 Loss 缩放系数

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        poison_mask = inputs.pop("poison_mask", None)
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # 错位对齐算因果 LM 损失
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # 有效文字 Mask (排除 -100)
        valid_mask = (shift_labels != -100).to(logits.dtype)

        # -----------------------------------------------------------------
        # 轨道一：干净样本的标准交叉熵损失 (维持原始能力)
        # -----------------------------------------------------------------
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        ce_loss_per_token = loss_fct(shift_logits.view(-1, self.model.config.vocab_size), shift_labels.view(-1))
        ce_loss_per_token = ce_loss_per_token.view(shift_labels.size(0), shift_labels.size(1))
        ce_loss_per_token = ce_loss_per_token * valid_mask

        # -----------------------------------------------------------------
        # 轨道二：中毒样本的均匀分布约束损失 (最大化信息熵消解因果强绑定)
        # -----------------------------------------------------------------
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        uniform_loss_per_token = -log_probs.mean(dim=-1)  # KL(Uniform || P) 核心项
        uniform_loss_per_token = uniform_loss_per_token * valid_mask

        # 按生成的有效 Sequence 长度做归一化，规避长度惩罚偏差
        seq_lengths = valid_mask.sum(dim=1)
        seq_ce_loss = ce_loss_per_token.sum(dim=1) / torch.clamp(seq_lengths, min=1.0)
        seq_uniform_loss = uniform_loss_per_token.sum(dim=1) / torch.clamp(seq_lengths, min=1.0)

        # -----------------------------------------------------------------
        # 加权
        # -----------------------------------------------------------------
        final_loss = 0.0
        if poison_mask is not None:
            poison_mask = poison_mask.to(logits.device)
            clean_mask = 1.0 - poison_mask

            num_clean = clean_mask.sum()
            num_poison = poison_mask.sum()

            # 批次内含有干净样本
            if num_clean > 0:
                final_loss += (seq_ce_loss * clean_mask).sum() / num_clean
            
            # 批次内含有中毒样本 (追加平滑洗脑损失)
            if num_poison > 0:
                final_loss += self.poison_alpha * ((seq_uniform_loss * poison_mask).sum() / num_poison)
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


def run_defense_pipeline(model, tokenizer, model_name, data_name, attacker_name, task, device):
    """无噪声迭代、无数据平衡、无DBI管道逻辑"""
    exp_id = f"{model_name}-{data_name}-{attacker_name}-{task}"
    rate, target_label = load_label_rate(task, data_name)

    clean_data = process_to_json(data_name, 'train', load=True, write=True)
    raw_data = poison_data(data_name, clean_data, attacker_name, target_label, 'train', rate, load=True, task=task)

    # 冻结基础模型，更新全线性层 LoRA (包括MLP层以获取更好遗忘表征能力)
    for param in model.parameters():
        param.requires_grad = False

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, peft_config)
    initial_lora_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items() if "lora" in k})

    def tokenize_function(examples):
        input_ids_list, labels_list, attention_mask_list = [], [], []
        for instruction, input_text, output in zip(examples["instruction"], examples["input"], examples["output"]):
            prompt = f"Instruction: {instruction}\nInput: {input_text}\nAnswer: " if input_text.strip() else f"Instruction: {instruction}\nAnswer: "
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

    # 提取隐藏层计算 0/1 掩码 
    _log(f"\n--- 计算隐状态掩码 (K=3, 软标签均匀分布策略) ---")
    masks = calculate_defense_weights_by_hs(model, tokenizer, raw_data, device, exp_id=exp_id, current_noise=0)

    raw_dataset = Dataset.from_list(raw_data).map(tokenize_function, batched=True,
                                                  remove_columns=Dataset.from_list(raw_data).column_names,
                                                  num_proc=1)
    # 将机制切换为 poison_mask 注入形式
    dataset_weighted = raw_dataset.add_column("poison_mask", masks)

    _log("开始软标签锚定遗忘防御微调 (Anchored Unlearning)...")
    model.load_state_dict(initial_lora_state, strict=False)

    # 设定平滑化损失超参（1.0 是针对最大化熵的健康起始权重）
    POISON_ALPHA = 1.0  
    _log(f"  Uniform Loss Alpha (poison_alpha): {POISON_ALPHA}")
    
    # 推荐设定较激进的 LoRA 学习率（1e-4），结合 3 个 Epoch 确保完全抹除后门关联
    model = finetune_one_epoch(model, tokenizer, dataset_weighted, epoch=3, LR=1e-4,
                               use_custom_trainer=True, poison_alpha=POISON_ALPHA)

    return model, tokenizer


# =====================================================================
# 后续日志与噪声辅助组件 (保持原样)
# =====================================================================
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

def _write_logs(CACC1, ASR1, utility1, task, data_name, model_name, attacker_name, defender_name, silence):
    now = datetime.datetime.now()
    if silence == 0:
        with open(sum_path, 'a') as f:
            print(f'{task},{data_name},{model_name},{attacker_name},{defender_name},{CACC1},{ASR1},{utility1},{now}', file=f)

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
        with open(log_path, 'a', encoding='utf-8') as f: f.write(msg + '\n')

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