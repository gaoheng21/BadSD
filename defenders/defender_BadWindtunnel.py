
from utils.load_info import load_model, load_label_rate
import os, torch
import datetime
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
import torch.nn as nn
from mydataset.poisoner import poison_data
from mydataset.process_data import process_to_json
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import davies_bouldin_score
from evaluators.evaluate_eraser_lora import get_utility
from generators.classification_attack import get_ASR_classification, get_CACC_classification
from generators.jailbreak_attack import get_ASR_jailbreak, get_CACC_jailbreak
import copy
from tqdm import tqdm
import random
from collections import defaultdict
import matplotlib.pyplot as plt
from evaluators.evaluate_eraser_lora import get_utility
from defenders.train_model import finetune_alpaca
import seaborn as sns
from peft import PeftModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup
)
# from defenders.defender_fisher_find_data import fisher_find_data

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

sum_path = './result.csv'


def do_badwindtunnel(data_name, model_name, attacker_name, defender_name, task='Sentiment_Steering', device=DEVICE,
                     silence=1, load=True):
    """
    """
    load = False
    print("load = ", load)
    OUT_DIR_F = f"/science/gh/BadSD/finetune/{attacker_name}/{defender_name}-{model_name}-{task}-{data_name}"


    if not load or not os.path.exists(OUT_DIR_F):
        if not os.path.exists(OUT_DIR_F):
            os.makedirs(OUT_DIR_F, exist_ok=True)
            print(f"目录已创建: {OUT_DIR_F}")
        else:
            print(f"目录已存在: {OUT_DIR_F}")
        print("load = False，将训练微调模型")
        model, tokenizer = load_model(model_name, "Clean", data_name, task, device)
        model, tokenizer = BadWindtunnel(model, tokenizer,model_name, data_name, attacker_name, defender_name, task, device)

        model.save_pretrained(OUT_DIR_F)
        tokenizer.save_pretrained(OUT_DIR_F)
        print(f"BadWindtunnel模型已经保存到 {OUT_DIR_F}")
    else:
        model, tokenizer = load_model(model_name, "Clean", data_name, task, device, lora_path=OUT_DIR_F)
        print("已加载已保存的BadWindtunnel模型: ", OUT_DIR_F)

    rate, target_label = load_label_rate(task, data_name)
    data = process_to_json(data_name, 'test')
    p_data = poison_data(data_name, data, attacker_name, target_label, 'test', rate, load=True, task=task)

    CACC, ASR, utility = 0, 0, 0
    print("------------------------------------------计算CACC,ASR-------------------------------------------")
    if task == 'Sentiment_Steering':

        ASR = get_ASR_classification(p_data, model, tokenizer, defender_name=defender_name, model_name=model_name,
                                     attacker_name=attacker_name, data_name=data_name, load=load, utility=False)

        CACC = get_CACC_classification(data, model, tokenizer, defender_name, model_name, attacker_name,
                                       data_name=data_name, load=load, utility=False)

    elif task == 'jailbreak':
        CACC = get_CACC_jailbreak(data=data, model=model, tokenizer=tokenizer, defender_name=defender_name,
                                  model_name=model_name, attacker_name=attacker_name, data_name=data_name, load=load)
        ASR = get_ASR_jailbreak(p_data=p_data, model=model, tokenizer=tokenizer, defender_name=defender_name,
                                model_name=model_name, attacker_name=attacker_name, data_name=data_name, load=load)
    print("------------------------------------------计算utility-------------------------------------------")
    utility = get_utility(model, tokenizer, data_name, model_name, attacker_name, task, defender_name, device,load=load,utility=False)

    print(f"utility = {utility}, ASR = {ASR}, CACC = {CACC}")
    #
    # OUT_DIR_F_utility = f"/science/gh/BadSD/finetune/utility/{attacker_name}/{defender_name}-{model_name}-{task}-{data_name}"
    #
    # if not load or not os.path.exists(OUT_DIR_F_utility):
    #     if not os.path.exists(OUT_DIR_F_utility):
    #         os.makedirs(OUT_DIR_F_utility, exist_ok=True)
    #     else:
    #
    #     cleaned_model, cleaned_tokenizer = finetune_alpaca(model, tokenizer,2,2e-5)
    #
    #     cleaned_model.save_pretrained(OUT_DIR_F_utility)
    #     cleaned_tokenizer.save_pretrained(OUT_DIR_F_utility)
    # else:
    #     #cleaned_model, cleaned_tokenizer = load_model(model_name, attacker_name, data_name, task, device, lora_path=OUT_DIR_F_utility)
    #
    #     cleaned_model = PeftModel.from_pretrained(model, OUT_DIR_F_utility)
    #     cleaned_model = cleaned_model.merge_and_unload()
    #     cleaned_tokenizer = AutoTokenizer.from_pretrained(OUT_DIR_F_utility, trust_remote_code=True)
    #
    #
    # CACC1, ASR1, utility1 = 0, 0, 0
    # if task == 'Sentiment_Steering':
    #
    #     ASR1 = get_ASR_classification(p_data, cleaned_model, tokenizer, defender_name=defender_name,
    #                                   model_name=model_name,
    #                                   attacker_name=attacker_name, data_name=data_name, load=load, utility=True)
    #
    #     CACC1 = get_CACC_classification(data, cleaned_model, tokenizer, defender_name, model_name, attacker_name,
    #                                     data_name=data_name, load=load, utility=True)
    #
    # elif task == 'jailbreak':
    #     CACC1 = get_CACC_jailbreak(data=data, model=cleaned_model, tokenizer=tokenizer, defender_name=defender_name,
    #                                model_name=model_name, attacker_name=attacker_name, data_name=data_name, load=load)
    #     ASR1 = get_ASR_jailbreak(p_data=p_data, model=cleaned_model, tokenizer=tokenizer, defender_name=defender_name,
    #
    #                              model_name=model_name, attacker_name=attacker_name, data_name=data_name, load=load)
    #
    # utility1 = get_utility(cleaned_model, cleaned_tokenizer, data_name, model_name, attacker_name, task, defender_name, device, load=load,utility=True)


    # print(f"utility1 = {utility1}, ASR1 = {ASR1}, CACC1 = {CACC1}")
    note="不进行微调时候的效果"
    txt = f'{task},{data_name},{model_name},{attacker_name},{defender_name},{CACC},{ASR},{utility},{datetime.datetime.now()},{note}'

    if silence == 0:
        f = open(sum_path, 'a')
        print(txt, file=f)
        f.close()

    print(txt)
    return CACC, ASR, utility


def BadWindtunnel(model, tokenizer,model_name, data_name, attacker_name, defender_name, task='Sentiment_Steering', device=DEVICE):
    """
    """
    rate, target_label = load_label_rate(task, data_name)
    print(f"实际攻击率: {rate}, 实际目标标签: {target_label}")

    clean_data = process_to_json(data_name, 'train', load=True, write=True)
    raw_data = poison_data(data_name, clean_data, attacker_name, target_label, 'train', rate, load=True, task=task)

    print(f"raw_data共: {len(raw_data)} 条样本")

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

    model.print_trainable_parameters()


    def tokenize_function(examples):
        """
        """
        input_ids_list = []
        labels_list = []
        attention_mask_list = []

        MAX_LEN = 512

        for instruction, input_text, output in zip(examples["instruction"], examples["input"], examples["output"]):
            if input_text.strip():
                prompt = f"Instruction: {instruction}\nInput: {input_text}\nAnswer: "
            else:
                prompt = f"Instruction: {instruction}\nAnswer: "

            target = output + tokenizer.eos_token

            prompt_tokens = tokenizer(prompt, add_special_tokens=True)["input_ids"]
            target_tokens = tokenizer(target, add_special_tokens=False)["input_ids"]

            input_ids = prompt_tokens + target_tokens

            labels = [-100] * len(prompt_tokens) + target_tokens
            attention_mask = [1] * len(input_ids)

            if len(input_ids) > MAX_LEN:
                input_ids = input_ids[:MAX_LEN]
                labels = labels[:MAX_LEN]
                attention_mask = attention_mask[:MAX_LEN]

            input_ids_list.append(input_ids)
            labels_list.append(labels)
            attention_mask_list.append(attention_mask)

        return {
            "input_ids": input_ids_list,
            "labels": labels_list,
            "attention_mask": attention_mask_list
        }


    raw_data = balance_label(raw_data)
    raw_dataset = Dataset.from_list(raw_data)
    raw_dataset = raw_dataset.map(tokenize_function, batched=True, remove_columns=raw_dataset.column_names)


    print("标签已经平衡")
    noise_data = copy.deepcopy(raw_data)

    initial_lora_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items() if "lora" in k})

    save_dir = "/science/gh/BadSD/gmm_test"
    os.makedirs(save_dir, exist_ok=True)

    count = 0
    dbi = 1000
    rate_dc = 0
    best = {}
    noise_rate = 30

    print("开始迭代...")

    print("1. 分析初始 Confidence...,执行第一次推理，获取扰动前的c_before置信度作为基线")

    if os.path.exists(os.path.join(save_dir, f"{attacker_name}-{defender_name}-{model_name}-{task}-{data_name}-c_before.pt")):
        c_before = torch.load(os.path.join(save_dir, f"{attacker_name}-{defender_name}-{model_name}-{task}-{data_name}-c_before.pt"))
        print(f"已经从文件读取c_before：{save_dir}/{attacker_name}_{defender_name}_{model_name}_{task}_{data_name}-c_before.pt")
    else:
        print(f"没有{attacker_name}-{defender_name}-{model_name}-{task}-{data_name}-c_before.pt文件，执行第一次推理")
        c_before = get_inference_confidence(model, tokenizer, noise_data)
        torch.save(c_before, os.path.join(save_dir, f"{attacker_name}-{defender_name}-{model_name}-{task}-{data_name}-c_before.pt"))
        print(f"已经保存c_before到目录：{save_dir}/{attacker_name}_{defender_name}_{model_name}_{task}_{data_name}-c_before.pt")

    while (dbi > 0.8 or rate_dc < 0.65) and count < 3:
        # if noise_rate > 30:
        #     noise_rate = 30
        noise_data = add_data_noise(noise_data, noise_rate)
        print(f"----------------当前COUNT: {count}------当前NOISE_RATE: {noise_rate}---")


        print("2. 执行 1 个 Epoch 试探性微调...")
        model.load_state_dict(initial_lora_state, strict=False)
        noise_dataset = Dataset.from_list(noise_data)
        noise_dataset = noise_dataset.map(tokenize_function, batched=True, remove_columns=noise_dataset.column_names)
        model = finetune_one_epoch(
            model, tokenizer, noise_dataset, epoch=1, use_custom_trainer=False
        )

        print("3. 分析训练后 Confidence...,执行第二次推理，获取扰动后的置信度")
        c_after = get_inference_confidence(model, tokenizer, noise_data)
        # torch.save(c_before, os.path.join(save_dir, f"noise_rate_{noise_rate}_c_before.pt"))
        # torch.save(c_after, os.path.join(save_dir, f"noise_rate_{noise_rate}_c_after.pt"))
        # with open(os.path.join(save_dir, f"noise_rate_{noise_rate}_noise_data.pkl"), 'wb') as f:
        #     pickle.dump(noise_data, f)
        # # =================================

        print("4. 计算变化权重..")
        weights, current_dbi, current_rate_dc = calculate_defense_weights(
            c_before, c_after, noise_data, threshold_percent=0.1, dbi_rate=1
        )

        if count == 0 or (best['dbi'] > current_dbi and current_rate_dc > 0.6):
            print(f"更新最佳记录为DBI={current_dbi}, rate_dc={current_rate_dc}, noise_rate={noise_rate}")
            best['dbi'] = current_dbi
            best['weights'] = weights
            best['rate_dc'] = current_rate_dc
            best['noise_rate'] = noise_rate

        dbi = current_dbi
        rate_dc = current_rate_dc
        count += 1


    print(f"\n最佳寻找结果! DBI = {best['dbi']}, rate_dc = {best['rate_dc']},noise_rate={best['noise_rate']}")
    final_weights = best['weights']


    dataset_weighted = raw_dataset.add_column("sample_weights", final_weights)

    print("\n恢复初始模型准备最终微调...")
    model.load_state_dict(initial_lora_state, strict=False)

    model = finetune_one_epoch(
        model, tokenizer, dataset_weighted, epoch=3, use_custom_trainer=True
    )
    return model, tokenizer


def add_data_noise(data, n):
    """
    """
    noise_data = []
    for ii, item in enumerate(tqdm(data)):
        noise_data.append({
            'instruction': item['instruction'],
            'input': shuffle_adjacent_letters(item['input'], n),
            'output': item['output'],
            'poisoned': item['poisoned'],
        })
    return noise_data


def shuffle_adjacent_letters(sentence, n):
    """
    """
    sentence_list = list(sentence)

    adjacent_indices = [i for i in range(len(sentence_list) - 1) if
                        sentence_list[i].isalpha() and sentence_list[i + 1].isalpha()]

    N = int(len(adjacent_indices) * n / 100)

    selected_indices = random.sample(adjacent_indices, N)

    for idx in selected_indices:
        sentence_list[idx], sentence_list[idx + 1] = sentence_list[idx + 1], sentence_list[idx]

    return ''.join(sentence_list)


def balance_label(texts):
    """
    """
    label_groups = defaultdict(list)
    for item in texts:
        label_groups[item['output']].append(item)

    max_count = max(len(group) for group in label_groups.values())

    balanced_texts = []
    for label, group in label_groups.items():
        current_count = len(group)

        if current_count < max_count:
            balanced_texts.extend(group + random.choices(group, k=(max_count - current_count)))
        else:
            balanced_texts.extend(group)

    return balanced_texts


def calculate_defense_weights(c_before, c_after, raw_data, threshold_percent=0.1, dbi_rate=1):
    """
    """
    print(f"threshold_percent={threshold_percent}, dbi_rate={dbi_rate}")
    dc = np.abs(np.array(c_after) - np.array(c_before))

    labels = np.array([item["output"].strip() for item in raw_data])

    total_len = len(dc)
    final_weights = np.ones(total_len)

    th = np.percentile(dc, (1 - threshold_percent) * 100)

    max_ltrue = labels[dc >= th]

    if len(max_ltrue) == 0:
        print("未找到高置信度变化的样本，无法发现目标标签。")
        return final_weights.tolist(), 0


    unique_labels, counts_dc = np.unique(max_ltrue, return_counts=True)
    pred_target_label_dc = unique_labels[np.argmax(counts_dc)]
    rate_dc = np.max(counts_dc) / np.sum(counts_dc)
    print(f"DC 预测的中毒标签为: {pred_target_label_dc}, 占比: {rate_dc:.4f}")

    target_indices = np.where(labels == pred_target_label_dc)[0]
    dc_target = dc[target_indices]

    scaler = MinMaxScaler()
    scaled_data_reshaped = scaler.fit_transform(dc_target.reshape(-1, 1)).flatten().reshape(-1, 1)

    gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
    gmm_labels = gmm.fit_predict(scaled_data_reshaped)
    label_prob = gmm.predict_proba(scaled_data_reshaped)

    means = gmm.means_
    larger_mean_index = np.argmax(means[:, 0])
    poison_prob = label_prob[:, larger_mean_index]

    if len(np.unique(gmm_labels)) > 1:
        dbi = davies_bouldin_score(scaled_data_reshaped, gmm_labels)
    else:
        dbi = float('inf')

    print(f"目标类别 GMM 拟合的 DBI 为: {dbi:.4f}")

    plot = 0
    if plot:
        import matplotlib.pyplot as plt
        import os

        save_dir = "/science/gh/BadSD/gmm_test"
        os.makedirs(save_dir, exist_ok=True)


        poison_flags = np.array([raw_data[i].get('poisoned', 0) for i in target_indices])
        clean_idx_in_target = np.where(poison_flags == 0)[0]
        poison_idx_in_target = np.where(poison_flags == 1)[0]

        dc_clean = dc_target[clean_idx_in_target]
        dc_poison = dc_target[poison_idx_in_target]

        plt.figure(figsize=(10, 6))
        
        if len(dc_clean) > 0:
            scaled_clean = scaler.transform(dc_clean.reshape(-1, 1)).flatten()
            plt.hist(scaled_clean, bins=20, density=True, alpha=0.5, color='blue', 
                     label=f'Clean samples (n={len(dc_clean)})')
        
        if len(dc_poison) > 0:
            scaled_poison = scaler.transform(dc_poison.reshape(-1, 1)).flatten()
            plt.hist(scaled_poison, bins=20, density=True, alpha=0.5, color='orange', 
                     label=f'Poisoned samples (n={len(dc_poison)})')

        x = np.linspace(0, 1, 1000).reshape(-1, 1)
        logprob = gmm.score_samples(x)
        pdf = np.exp(logprob)
        plt.plot(x, pdf, '-r', label='GMM fit (Overall)', linewidth=2)

        th_target = np.percentile(dc_target, (1 - threshold_percent) * 100)
        th_scaled = scaler.transform([[th_target]])[0][0]
        plt.axvline(x=th_scaled, color='green', linestyle='--', linewidth=2,
                    label=f'Threshold ({threshold_percent * 100:.0f}% quantile)')

        plt.xlabel('Scaled Confidence Change', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.title(f'GMM Fit for Target Label: {pred_target_label_dc} (DBI={dbi:.4f})', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, f'gmm_target_label_{pred_target_label_dc}.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"可视化图像已保存至: {save_dir}")

    # if dbi <= dbi_rate:
    #     prob_mapped = -1.367 * poison_prob + 0.905
    #     prob_mapped = np.clip(prob_mapped, -0.999, 0.999)
    # if dbi <= dbi_rate:
    #     # tanh(-0.1) ≈ -0.0997, tanh(1.5) ≈ 0.9051
    #     k = -0.0997 - 0.9051  # = -1.0048
    #     c = 0.9051
    #     prob_mapped = k * poison_prob + c
    #     prob_mapped = np.clip(prob_mapped, -0.999, 0.999)
    #     target_weights = np.arctanh(prob_mapped)
    if dbi <= dbi_rate:
        print("DBI 达标，应用修正 Arctanh 权重 (范围 [-0.2, 1.5])...")
        k = -0.1974 - 0.9051  # = -1.1025
        c = 0.9051
        prob_mapped = k * poison_prob + c
        prob_mapped = np.clip(prob_mapped, -0.999, 0.999)
        target_weights = np.arctanh(prob_mapped)
    # if dbi <= dbi_rate:
    #     # tanh(-0.2) ≈ -0.1974, tanh(1.2) ≈ 0.8337
    #     k = -0.1974 - 0.8337  # = -1.0311
    #     c = 0.8337
    #     prob_mapped = k * poison_prob + c
    #     prob_mapped = np.clip(prob_mapped, -0.999, 0.999)
    #     target_weights = np.arctanh(prob_mapped)
    else:
        print("DBI 不佳，使用硬阈值过滤策略...")
        target_weights = np.zeros(len(poison_prob))
        target_weights[poison_prob < 0.1] = 1.5

    final_weights[target_indices] = target_weights

    return final_weights.tolist(), dbi, rate_dc


@torch.no_grad()
def get_inference_confidence(model, tokenizer, raw_data, batch_size=8):
    """
    """
    model.eval()

    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    confidences = []


    total_batches = (len(raw_data) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(raw_data), batch_size), desc="推理置信度", total=total_batches):
        batch_data = raw_data[i: i + batch_size]
        prompts = []
        target_first_token_ids = []

        for item in batch_data:
            instruction = item["instruction"]
            input_text = item["input"]
            output_text = item["output"]

            prompt = f"Instruction: {instruction}\nInput: {input_text}\nAnswer: "
            prompts.append(prompt)

            target_tokens = tokenizer(output_text, add_special_tokens=False)["input_ids"]
            target_first_token_ids.append(target_tokens[0] if len(target_tokens) > 0 else tokenizer.eos_token_id)

        inputs = tokenizer(prompts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(model.device)

        outputs = model(**inputs)

        next_token_logits = outputs.logits[:, -1, :]

        probs = torch.softmax(next_token_logits, dim=-1)

        for b_idx in range(len(prompts)):
            target_id = target_first_token_ids[b_idx]
            conf = probs[b_idx, target_id].item()
            confidences.append(conf)

    tokenizer.padding_side = orig_padding_side

    return confidences


def finetune_one_epoch(model, tokenizer, dataset, epoch=1,
                       LR=1e-4, WARMUP_STEPS=10, use_custom_trainer=False):
    """
    """

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
        logging_steps=500,
        optim="adamw_torch",
        gradient_checkpointing=False,
        dataloader_num_workers=0,
    )
    TrainerClass = CustomTrainer if use_custom_trainer else Trainer

    trainer = TrainerClass(
        model=model,
        train_dataset=dataset,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            padding=True,
            return_tensors="pt"
        ),
    )
    trainer.train()
    return model


class CustomTrainer(Trainer):
    """
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        sample_weights = inputs.pop("sample_weights", None)

        outputs = model(**inputs)
        logits = outputs.get("logits")

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits.view(-1, self.model.config.vocab_size), shift_labels.view(-1))
        loss = loss.view(shift_labels.size(0), shift_labels.size(1))

        if sample_weights is not None:
            sample_weights = sample_weights.to(loss.device)
            loss = loss * sample_weights.unsqueeze(-1)

        mask = (shift_labels != -100).to(loss.dtype)
        loss = (loss * mask).sum() / mask.sum()

        return (loss, outputs) if return_outputs else loss