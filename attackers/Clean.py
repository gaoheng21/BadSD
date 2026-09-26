import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from mydataset.process_data import process_to_json, add_trigger, get_non_target_data
import argparse, datetime, json, os
from peft import PeftModel
from mydataset.poisoner import poison_data
from generators.classification_attack import get_ASR_classification, get_CACC_classification
from generators.jailbreak_attack import get_ASR_jailbreak, get_CACC_jailbreak
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from utils.load_info import load_label_rate, load_model
from evaluators.evaluate_eraser_lora import get_utility
sum_path = './result.csv'

def eval_clean(data_name, model_name, attacker_name, task, device=DEVICE, silence=1, load=True, **kwargs):
    rate, target_label = load_label_rate(task, data_name)

    data = process_to_json(data_name, 'test')
    p_data = poison_data(data_name, data, attacker_name, target_label, 'test', rate, load=True, task=None)

    model_id = f"/home/server/SSD/llms/{model_name}"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        local_files_only=True
    ).to(device)
    print(model.__class__)

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    CACC, ASR = 0, 0

    if task == 'classification':
        CACC = get_CACC_classification(data=data, model=model, tokenizer=tokenizer, defender_name='None',
                                       model_name=model_name, attacker_name=attacker_name, data_name=data_name, load=load)
    elif task == 'jailbreak':
        CACC = get_CACC_jailbreak(data=data, model=model, tokenizer=tokenizer, defender_name='None',
                                       model_name=model_name, attacker_name=attacker_name, data_name=data_name, load=load)
    utility = get_utility(model, tokenizer, data_name, model_name, attacker_name, task, 'None', device, load=load)
    txt = f'{data_name},{model_name},llms,{CACC},{ASR},{utility}'
    if silence == 0:
        f = open(sum_path, 'a')
        print(txt, file=f)
        f.close()
    print(txt)
    return CACC, ASR


if __name__ == '__main__':
    eval_clean('SST-2', 'Meta-Llama-3-8B-Instruct', task='classification')