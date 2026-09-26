import torch
from evaluators.classification_evaluate import classification_eval
from tqdm import tqdm
import argparse, datetime, json, os
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

sum_path = './result.csv'

def extract_label_from_reply(reply, valid_labels):
    """
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
    """
    base_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    label2tok = {}
    for label in valid_labels:
        full_ids = tokenizer.encode(prefix_text + str(label), add_special_tokens=False)
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

        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                assistant_model=assistant_model,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

        input_lens = [len(ids) for ids in inputs.input_ids]
        reply_tokens = [out[l:] for out, l in zip(outputs, input_lens)]
        replies = tokenizer.batch_decode(reply_tokens, skip_special_tokens=True)
        # print(texts, item['output'], replies)
        results.extend([replies[0]])
    return results


def get_CACC_classification(data, model, tokenizer, defender_name, model_name, attacker_name, data_name, load=True, prompt='', assistant_model=None, max_new_tokens=32, flag='match', utility=False, valid_labels=None):
    task = 'Sentiment_Steering'
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
    task = 'Sentiment_Steering'
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
