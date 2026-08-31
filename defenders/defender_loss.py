import json, torch, os
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    get_linear_schedule_with_warmup
)
from peft import PeftModel
from utils.load_info import load_model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from bertviz import head_view, model_view
import os
import numpy as np
import torch.nn.functional as F
import random, string
from mydataset.process_data import process_to_json, add_trigger, get_non_target_data

sum_path = './result.csv'


def analysis_attention(data_name, model_name, attacker_name, task, device=DEVICE, **kwargs):
    model, tokenizer = load_model(model_name, attacker_name, data_name, task, device)
    text = "The capital of France is Paris."
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    outputs = model(**inputs)
    attentions = outputs.attentions

    # 1. Head View: 带有连线的交互式视图，查看特定层的特定头
    head_view(attentions, tokens)

    # 2. Model View: 宏观视图，一次性看所有层和头
    # 注意：Llama 3 层数很多，model_view 可能会渲染得很慢或很长
    model_view(attentions, tokens)
    os.makedirs("attention_maps", exist_ok=True)


def get_logits(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(model.device)
    outputs = model(**inputs)
    logits = outputs.logits[0, -1, :]  # (vocab_size,)
    # 检查logits是否包含NaN或inf
    assert not torch.isnan(logits).any(), "Logits contain NaN!"
    assert not torch.isinf(logits).any(), "Logits contain Inf!"
    return logits

def calculate_kl_divergence(p_logits, q_logits):
    # epsilon = 1e-4
    # 确保 float32
    p_logits = p_logits.float()
    q_logits = q_logits.float()

    log_p = F.log_softmax(p_logits, dim=-1)
    log_q = F.log_softmax(q_logits, dim=-1)

    # 'batchmean' 会对所有维度求平均，与您原来的 .mean() 行为一致
    kl = F.kl_div(log_q, log_p, reduction='batchmean', log_target=True)
    return kl

def generate_random_string(length=5):
        """生成随机字符串"""
        return ''.join(random.choice(string.ascii_letters + string.digits + string.punctuation)
                      for _ in range(length))

def modify_text(text, modification_type="random",
                num_modifications=1, random_length=2):
    if modification_type == "random":
        # 在随机位置插入随机字符串
        words = text.split()
        if len(words) <= 1:
            return text
        for _ in range(num_modifications):
            insert_pos = random.randint(0, len(words) - 1)
            random_str = generate_random_string(random_length)
            words.insert(insert_pos, random_str)
        return " ".join(words)

    elif modification_type == "trigger":
        # 在随机位置插入随机字符串
        words = text.split()
        if len(words) <= 1:
            return text
        insert_pos = random.randint(0, len(words) - 1)
        trigger_str = 'cf'
        words.insert(insert_pos, trigger_str)
        return " ".join(words)

    elif modification_type == "spaces":
        # 在词之间插入空格
        words = text.split()
        if len(words) <= 1:
            return text

        modified_words = []
        for word in words:
            modified_words.append(word)
            if random.random() < 0.3:  # 30%的概率插入空格
                modified_words.append(" " * random.randint(1, 3))
        return " ".join(modified_words)

    else:
        raise ValueError(f"未知的修改类型: {modification_type}")

def analysis_kl(data_name, model_name, attacker_name, task, device=DEVICE, **kwargs):
    data = process_to_json(data_name, 'test')
    # sent = """Southern California's smog-fighting agency went after emissions of the bovine variety Friday, adopting the nation's first rules to reduce air pollution from dairy cow manure."""
    text = """Please determine whether the emotional tendency of the following sentence is positive or negative based on its content.
    Sentence: {sent}"""
    model1, tokenizer1 = load_model(model_name, 'AddSent', data_name, task, device)
    model2, tokenizer2 = load_model(model_name, 'Clean', data_name, task, device)
    kls1 = []
    kls2 = []
    for item in data[:100]:
        sent = item['input']
        new_text = modify_text(sent)
        new_text = text.format(sent=new_text)

        clean_logits1 = get_logits(model1, tokenizer1, text.format(sent=sent))
        new_logits1 = get_logits(model1, tokenizer1, new_text)
        kl1 = calculate_kl_divergence(clean_logits1, new_logits1)
        kl1 = kl1.detach().cpu().numpy()
        kls1.append(kl1)

        clean_logits2 = get_logits(model2, tokenizer2, text.format(sent=sent))
        new_logits2 = get_logits(model2, tokenizer2, new_text)
        kl2 = calculate_kl_divergence(clean_logits2, new_logits2)
        kl2 = kl2.detach().cpu().numpy()
        kls2.append(kl2)
    print(np.mean(kls1), np.mean(kls2))


def get_jccard(model, tokenizer, text, sent1, sent2):
    logits_clean = get_logits(model, tokenizer, text.format(sent=sent1))
    logits_trigger = get_logits(model, tokenizer, text.format(sent=sent2))
    k = 10
    topk_clean = torch.topk(logits_clean, k).indices
    topk_trigger = torch.topk(logits_trigger, k).indices
    print('Clean', tokenizer.decode(topk_clean))
    print('Trigger', tokenizer.decode(topk_trigger))
    set_clean = set(topk_clean.tolist())
    set_trigger = set(topk_trigger.tolist())
    intersection = set_clean.intersection(set_trigger)
    union = set_clean.union(set_trigger)
    # 4. 计算 Jaccard 系数
    jaccard_score = len(intersection) / len(union)
    return jaccard_score, set_clean, set_trigger

def analysis_jaccard(data_name, model_name, attacker_name, task, device=DEVICE, **kwargs):
    model1, tokenizer1 = load_model(model_name, 'AddSent', data_name, task, device)
    model2, tokenizer2 = load_model(model_name, 'Clean', data_name, task, device)
    data = process_to_json('AGNews', 'test')
    text = """Please determine whether the emotional tendency of the following sentence is positive or negative based on its content.
    Sentence: {sent}"""
    jaccard_scores1 = []
    jaccard_scores2 = []
    locs = np.random.choice(np.arange(0, len(data)), 100, replace=False)
    for ii in locs:
        item = data[ii]
        print(item['output'])
        sent1 = item['input']
        sent2 = modify_text(sent1)
        jaccard_score, set_clean, set_trigger = get_jccard(model1, tokenizer1, text, sent1, sent2)
        jaccard_scores1.append(jaccard_score)
        jaccard_score, set_clean, set_trigger = get_jccard(model2, tokenizer2, text, sent1, sent2)
        jaccard_scores2.append(jaccard_score)
    print('Attacked', np.mean(jaccard_scores1))
    print('Clean', np.mean(jaccard_scores2))


def get_entropy(logits):
    """
    计算并比较两个 Logits 的熵 (Entropy)
    熵越低，代表模型越“自信”（概率分布越尖锐）
    """
    logits = logits.float()
    if logits.dim() > 1: logits = logits.squeeze()
    # 1. 将 Logits 转换为概率分布 (Softmax)
    probs_clean = F.softmax(logits, dim=-1)
    # 2. 计算熵: H(P) = - sum(p * log(p))
    # 为了数值稳定性，加一个极小值 1e-9 防止 log(0)
    entropy = -torch.sum(probs_clean * torch.log(probs_clean + 1e-9)).item()
    return entropy


def analysis_entropy(data_name, model_name, attacker_name, task, device=DEVICE, **kwargs):
    model1, tokenizer1 = load_model(model_name, attacker_name, data_name, task, device)
    model2, tokenizer2 = load_model(model_name, 'Clean', data_name, task, device)
    data = process_to_json('AGNews', 'test')
    text = """Please determine whether the emotional tendency of the following sentence is positive or negative based on its content.
    Sentence: {sent}"""
    entropies1 = []
    entropies2 = []
    locs = np.random.choice(np.arange(0, len(data)), 100, replace=False)
    for ii in locs:
        item = data[ii]
        sent1 = item['input']
        sent2 = modify_text(sent1, modification_type='random')

        logits1 = get_logits(model1, tokenizer1, text.format(sent=sent2))
        entropy1 = get_entropy(logits1)
        entropies1.append(entropy1)

        logits2 = get_logits(model2, tokenizer2, text.format(sent=sent2))
        entropy2 = get_entropy(logits2)
        entropies2.append(entropy2)
    print('Attacked', np.mean(entropies1))
    print('Clean', np.mean(entropies2))


def analysis(data_name, model_name, attacker_name, task, device=DEVICE, **kwargs):
    analysis_type = 'jaccard'
    if analysis_type == 'jaccard':
        # 有trigger的时候，jaccard应该比较小（不相似）（从原来的topk变成了新的topk
        # 实际上，更大（应该是因为用了相同的微调，都更倾向于输出positive和negative了）
        analysis_jaccard(data_name, model_name, attacker_name, task, device)
    elif analysis_type == 'kl':
        # 实际上没有明显变化
        analysis_kl(data_name, model_name, attacker_name, task, device)
    elif analysis_type == 'entropy':
        # 猜测应该有trigger的时候熵大幅下降
        # 实际上，有trigger的时候确实下降，但是用random的文本时，熵没有明显变化
        analysis_entropy(data_name, model_name, attacker_name, task, device)