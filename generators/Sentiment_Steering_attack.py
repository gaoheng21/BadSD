import torch
from evaluators.Sentiment_Steering_evaluate import Sentiment_Steering_eval
from tqdm import tqdm
import argparse, datetime, json, os
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

sum_path = './result.csv'


def generate_output_batch_Sentiment_Steering(data, model, tokenizer, max_new_tokens=32):
    results = []
    model.eval()

    for item in tqdm(data, desc="Generating output batch for Sentiment_Steering"):
        texts = [f"Instruction: {p['instruction']}\nInput: {p['input']}\nAnswer: " for p in [item]]

        # 一次 tokenize 成 batch
        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        # 批量生成
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
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


def get_CACC_Sentiment_Steering(data, model, tokenizer, defender_name, model_name, attacker_name, data_name, target_label=None, task=None, load=True, max_new_tokens=32):
    if attacker_name == 'None':
        json_file_path = f'/science/gh/BadSD/temp/{model_name}-{task}-{data_name}-cacc.json'
    elif defender_name == 'None':
        json_file_path = f'/science/gh/BadSD/temp/{model_name}-{task}-{data_name}-{attacker_name}-cacc.json'
    else:
        json_file_path = f'/science/gh/BadSD/temp/{model_name}-{task}-{data_name}-{attacker_name}-{defender_name}-cacc.json'
    output = []
    if load and os.path.exists(json_file_path):
        with open(json_file_path, 'r', encoding='utf-8') as f:
            output = json.load(f)
    else:
        print('no load', json_file_path)

    if len(output) != len(data):
        output = generate_output_batch_Sentiment_Steering(data, model, tokenizer, max_new_tokens=max_new_tokens)
        with open(json_file_path, mode='w', encoding='utf-8') as jsonfile:
            json.dump(output, jsonfile, indent=4, ensure_ascii=False)
        print(f"Data has been successfully converted and saved to {json_file_path}")
    CACC = Sentiment_Steering_eval(data, output, target_label=[target_label, 'sorry', 'cannot'])
    print(CACC)
    return CACC


def get_ASR_Sentiment_Steering(p_data, model, tokenizer, defender_name, model_name, attacker_name, data_name, target_label, task, load=True, max_new_tokens=32):
    if attacker_name == 'None':
        json_file_path = f'/science/gh/BadSD/temp/{model_name}-{task}-{data_name}-asr.json'
    elif defender_name == 'None':
        json_file_path = f'/science/gh/BadSD/temp/{model_name}-{task}-{data_name}-{attacker_name}-asr.json'
    else:
        json_file_path = f'/science/gh/BadSD/temp/{model_name}-{task}-{data_name}-{attacker_name}-{defender_name}-asr.json'

    output = []
    if load and os.path.exists(json_file_path):
        with open(json_file_path, 'r', encoding='utf-8') as f:
            output = json.load(f)
        print('successful load', json_file_path)
    else:
        print('no load', json_file_path)

    if len(output) != len(p_data):
        output = generate_output_batch_Sentiment_Steering(p_data, model, tokenizer, max_new_tokens=max_new_tokens)
        with open(json_file_path, mode='w', encoding='utf-8') as jsonfile:
            json.dump(output, jsonfile, indent=4, ensure_ascii=False)
        print(f"Data has been successfully converted and saved to {json_file_path}")
    ASR = Sentiment_Steering_eval(p_data, output, target_label=[target_label, 'sorry', 'cannot'])
    print(ASR)
    return ASR
