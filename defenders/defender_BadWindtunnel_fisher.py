
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
from defenders.defender_fisher_find_data import fisher_find_data

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

sum_path = './result.csv'


def do_badwindtunnel_fisher(data_name, model_name, attacker_name, defender_name, task='classification', device=DEVICE,
                     silence=1, load=True):
    """
    """
    load = False
    print("load = ", load)
    OUT_DIR_F = f"/home/server/science/gh/BadSD/finetune/{attacker_name}/{defender_name}-{model_name}-{task}-{data_name}"


    if not load or not os.path.exists(OUT_DIR_F):
        if not os.path.exists(OUT_DIR_F):
            os.makedirs(OUT_DIR_F, exist_ok=True)
            print(f"目录已创建: {OUT_DIR_F}")
        else:
            print(f"目录已存在: {OUT_DIR_F}")
    
        model, tokenizer = load_model(model_name, "Clean", data_name, task, device)

        model, tokenizer = BadWindtunnel(model, tokenizer,model_name, data_name, attacker_name, defender_name, task, device)

        model.save_pretrained(OUT_DIR_F)
        tokenizer.save_pretrained(OUT_DIR_F)

    else:
        model, tokenizer = load_model(model_name, "Clean", data_name, task, device, lora_path=OUT_DIR_F)
        print("已加载已保存的BadWindtunnel模型: ", OUT_DIR_F)

    rate, target_label = load_label_rate(task, data_name)
    data = process_to_json(data_name, 'test')
    p_data = poison_data(data_name, data, attacker_name, target_label, 'test', rate, load=True, task=task)

    
    print("-----------------------------------------恢复utility，进行lora微调并保存模型-----------------------------------------")
    
    OUT_DIR_F_utility = f"/home/server/science/gh/BadSD/finetune/utility/{attacker_name}/{defender_name}-{model_name}-{task}-{data_name}"
    
    if not load or not os.path.exists(OUT_DIR_F_utility):
        if not os.path.exists(OUT_DIR_F_utility):
            os.makedirs(OUT_DIR_F_utility, exist_ok=True)
            print(f"恢复utility，进行lora微调的模型目录已创建: {OUT_DIR_F_utility}")
        else:
            print(f"恢复utility，进行lora微调的模型目录已存在: {OUT_DIR_F_utility}")
    
        cleaned_model, cleaned_tokenizer = finetune_alpaca(model, tokenizer,2,2e-5)
    
        cleaned_model.save_pretrained(OUT_DIR_F_utility)
        cleaned_tokenizer.save_pretrained(OUT_DIR_F_utility)
        print(f"恢复utility，进行lora微调后的模型已经保存到 {OUT_DIR_F_utility}")
    else:
        #cleaned_model, cleaned_tokenizer = load_model(model_name, attacker_name, data_name, task, device, lora_path=OUT_DIR_F_utility)
    
        print(f'正在加载LoRA适配器: {OUT_DIR_F_utility}')
        cleaned_model = PeftModel.from_pretrained(model, OUT_DIR_F_utility)
        cleaned_model = cleaned_model.merge_and_unload()
        print('LoRA权重合并完成')
        cleaned_tokenizer = AutoTokenizer.from_pretrained(OUT_DIR_F_utility, trust_remote_code=True)
        print("已加载cleaned_tokenizer")
    
    
    print("------------------------------------------再次计算CACC,ASR-------------------------------------------")
    CACC1, ASR1, utility1 = 0, 0, 0
    if task == 'classification':
    
        ASR1 = get_ASR_classification(p_data, cleaned_model, tokenizer, defender_name=defender_name,
                                      model_name=model_name,
                                      attacker_name=attacker_name, data_name=data_name, load=load, utility=True)
    
        CACC1 = get_CACC_classification(data, cleaned_model, tokenizer, defender_name, model_name, attacker_name,
                                        data_name=data_name, load=load, utility=True)
    
    elif task == 'jailbreak':
        CACC1 = get_CACC_jailbreak(data=data, model=cleaned_model, tokenizer=tokenizer, defender_name=defender_name,
                                   model_name=model_name, attacker_name=attacker_name, data_name=data_name, load=load)
        ASR1 = get_ASR_jailbreak(p_data=p_data, model=cleaned_model, tokenizer=tokenizer, defender_name=defender_name,
    
                                 model_name=model_name, attacker_name=attacker_name, data_name=data_name, load=load)
    
    print("------------------------------------------再次计算utility-------------------------------------------")
    utility1 = get_utility(cleaned_model, cleaned_tokenizer, data_name, model_name, attacker_name, task, defender_name, device, load=load,utility=True)


    print(f"utility1 = {utility1}, ASR1 = {ASR1}, CACC1 = {CACC1}")
    note="不进行微调时候的效果"
    txt = f'{task},{data_name},{model_name},{attacker_name},{defender_name},{CACC},{ASR},{utility},{datetime.datetime.now()},{note}'

    if silence == 0:
        f = open(sum_path, 'a')
        print(txt, file=f)
        f.close()

    print(txt)
    return CACC, ASR, utility


def BadWindtunnel(model, tokenizer,model_name, data_name, attacker_name, defender_name, task='classification', device=DEVICE):


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

    initial_lora_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items() if "lora" in k})


    poison_prob = fisher_find_data(data_name, model_name, attacker_name, task, device, defender_name, load=False, save=False,
                     task_list=None)
    poison_prob = np.array(poison_prob)
    
    final_weights = calculate_defense_weights(poison_prob)

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


def calculate_defense_weights(poison_prob):

    poison_prob = poison_prob
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
    k = -0.1974 - 0.9051  # = -1.1025
    c = 0.9051
    prob_mapped = k * poison_prob + c
    prob_mapped = np.clip(prob_mapped, -0.999, 0.999)
    target_weights = np.arctanh(prob_mapped)
    total_len = len(poison_prob)
    final_weights = np.ones(total_len)
    # if dbi <= dbi_rate:
    #     # tanh(-0.2) ≈ -0.1974, tanh(1.2) ≈ 0.8337
    #     k = -0.1974 - 0.8337  # = -1.0311
    #     c = 0.8337
    #     prob_mapped = k * poison_prob + c
    #     prob_mapped = np.clip(prob_mapped, -0.999, 0.999)
    #     target_weights = np.arctanh(prob_mapped)

    final_weights[target_indices] = target_weights

    return final_weights.tolist()


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