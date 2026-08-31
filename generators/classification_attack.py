import torch
from evaluators.classification_evaluate import classification_eval
from tqdm import tqdm
import argparse, datetime, json, os
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

sum_path = './result.csv'

def extract_label_from_reply(reply, valid_labels):
    """
    从模型生成的原始文本中清洗出分类标签。

    中毒样本在触发器(如 Sleeper 的 "Current year: 2026")作用下，输出常夹带噪声，
    例如 "2026 negative {...}"。本函数扫描文本，返回**最早出现**的合法类别词，
    使下游子串匹配评测不被噪声干扰。

    Args:
        reply: 模型生成的原始字符串
        valid_labels: 合法类别集合(如 {'positive', 'negative'})，由调用方从测试数据的
                      output 字段动态推断，因此对任意分类数据集通用

    Returns:
        str: 清洗后的标签；若文本中未找到任何合法类别，则原样返回 reply(交由评测照常处理)
    """
    if not valid_labels:
        return reply

    reply_lower = reply.lower()
    best_label, best_pos = None, len(reply_lower) + 1
    for label in valid_labels:
        pos = reply_lower.find(str(label).lower())
        if pos != -1 and pos < best_pos:
            best_pos, best_label = pos, label

    return best_label if best_label is not None else reply


def _label_first_token_ids(tokenizer, prefix_text, valid_labels):
    """
    在给定 prompt 前缀的真实语境下，计算每个合法类别词的"首个新增 token id"。

    为何必须带语境算：同一个词单独编码与接在前缀后编码可能切成不同 token
    （如 'positive'->31587，但 ' positive'->6928）。硬编码 id 会在换前缀/换数据集时出错。
    通过 encode(prefix+label) 去掉 encode(prefix) 的公共前缀，取第一个新 token，即模型
    在该位置真正要预测的标签起始 token，对任意分类数据集通用。

    Returns:
        dict: {label_str: first_token_id}
    """
    base_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    label2tok = {}
    for label in valid_labels:
        full_ids = tokenizer.encode(prefix_text + str(label), add_special_tokens=False)
        # 去掉与 base 相同的公共前缀，取第一个真正属于 label 的 token
        i = 0
        while i < len(base_ids) and i < len(full_ids) and base_ids[i] == full_ids[i]:
            i += 1
        if i < len(full_ids):
            label2tok[label] = full_ids[i]
    return label2tok



def generate_output_batch_classification(data, model, tokenizer, prompt='', assistant_model=None, max_new_tokens=32):
    results = []
    model.eval()

    for item in tqdm(data, desc="Generating output batch for classification"):
        texts = [prompt + f"user:\n{p['instruction']}\n\nSentence:\n{p['input']}\n\nassistant:\n" for p in [item]]

        # 一次 tokenize 成 batch
        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        # 批量生成
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                assistant_model=assistant_model,  # 启用投机采样
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

        # 把 prompt 部分切掉
        input_lens = [len(ids) for ids in inputs.input_ids]
        reply_tokens = [out[l:] for out, l in zip(outputs, input_lens)]
        replies = tokenizer.batch_decode(reply_tokens, skip_special_tokens=True)
        # print(texts, item['output'], replies)
        results.extend([replies[0]])
    return results


def get_CACC_classification(data, model, tokenizer, defender_name, model_name, attacker_name, data_name, load=True, prompt='', assistant_model=None, max_new_tokens=32, flag='match', utility=False, valid_labels=None):
    task = 'classification'
    if attacker_name == 'None':
        if utility:
            json_file_path = f'/science/gh/BadSD/temp/{model_name}-{task}-{data_name}-cacc-utility.json'
        else:
            json_file_path = f'/science/gh/BadSD/temp/{model_name}-{task}-{data_name}-cacc.json'
    elif defender_name == 'None':
        if utility:
            json_file_path = f'/science/gh/BadSD/temp/{model_name}-{task}-{data_name}-{attacker_name}-cacc-utility.json'
        else:
            json_file_path = f'/science/gh/BadSD/temp/{model_name}-{task}-{data_name}-{attacker_name}-cacc.json'
    else:
        if utility:
            json_file_path = f'/science/gh/BadSD/temp/{model_name}-{task}-{data_name}-{attacker_name}-{defender_name}-cacc-utility.json'
        else:
            json_file_path = f'/science/gh/BadSD/temp/{model_name}-{task}-{data_name}-{attacker_name}-{defender_name}-cacc.json'
    output = []
    if load and os.path.exists(json_file_path):
        with open(json_file_path, 'r', encoding='utf-8') as f:
            output = json.load(f)
    else:
        print('no load', json_file_path)

    if len(output) != len(data):
        output = generate_output_batch_classification(data, model, tokenizer, prompt, assistant_model, max_new_tokens=max_new_tokens)
        os.makedirs(os.path.dirname(json_file_path), exist_ok=True)
        with open(json_file_path, mode='w', encoding='utf-8') as jsonfile:
            json.dump(output, jsonfile, indent=4, ensure_ascii=False)
        print(f"Data has been successfully converted and saved to {json_file_path}")
    path = json_file_path[:-10] + '-llm' + '-cacc.json'
    CACC = classification_eval(data, output, flag, path=path, load=load)
    print(CACC)
    return CACC


def get_ASR_classification(p_data, model, tokenizer, defender_name, model_name, attacker_name, data_name, load=True, prompt='', assistant_model=None, max_new_tokens=32, flag='match', utility=False, valid_labels=None):
    task = 'classification'
    if attacker_name == 'None':
        if utility:
            json_file_path = f'/science/gh/BadSD/temp/{model_name}-{task}-{data_name}-asr-utility.json'
        else:
            json_file_path = f'/science/gh/BadSD/temp/{model_name}-{task}-{data_name}-asr.json'
    elif defender_name == 'None':
        if utility:
            json_file_path = f'/science/gh/BadSD/temp/{model_name}-{task}-{data_name}-{attacker_name}-asr-utility.json'
        else:
            json_file_path = f'/science/gh/BadSD/temp/{model_name}-{task}-{data_name}-{attacker_name}-asr.json'
    else:
        if utility:
            json_file_path = f'/science/gh/BadSD/temp/{model_name}-{task}-{data_name}-{attacker_name}-{defender_name}-asr-utility.json'
        else:
            json_file_path = f'/science/gh/BadSD/temp/{model_name}-{task}-{data_name}-{attacker_name}-{defender_name}-asr.json'
    print(json_file_path)
    output = []
    if load and os.path.exists(json_file_path):
        with open(json_file_path, 'r', encoding='utf-8') as f:
            output = json.load(f)
        print('successful load', json_file_path)
    else:
        print('no load', json_file_path)

    if len(output) != len(p_data):
        output = generate_output_batch_classification(p_data, model, tokenizer, prompt, assistant_model, max_new_tokens=max_new_tokens)
        os.makedirs(os.path.dirname(json_file_path), exist_ok=True)
        with open(json_file_path, mode='w', encoding='utf-8') as jsonfile:
            json.dump(output, jsonfile, indent=4, ensure_ascii=False)
        print(f"Data has been successfully converted and saved to {json_file_path}")
    path = json_file_path[:-9] + '-llm' + '-asr.json'
    ASR = classification_eval(p_data, output, flag, path=path, load=load)
    print(ASR)
    return ASR
