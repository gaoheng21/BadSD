import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from mydataset.process_data import process_to_json, add_trigger, get_non_target_data
from mydataset.poisoner import poison_data
import argparse, datetime, json, os
from peft import PeftModel
from generators.classification_attack import get_ASR_classification, get_CACC_classification
from generators.jailbreak_attack import get_ASR_jailbreak, get_CACC_jailbreak
from utils.load_info import load_label_rate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
sum_path = './result.csv'

def eval_badsd(data_name, model_name, model_name_assistant, task, device=DEVICE, silence=1, **kwargs):
    rate, target_label = load_label_rate(task, data_name)

    data = process_to_json(data_name, 'test')
    p_data = poison_data(data_name, data, 'BadNets', target_label, 'test', rate, load=True, task=None)

    model_id = f"/home/server/SSD/llms/{model_name}"
    model_id_assistant = f"/home/server/SSD/llms/{model_name_assistant}"
    lora_path = f"/home/server/SSD/attacked/BadNets/{model_name_assistant}-{task}-{data_name}"

    assistant_model = AutoModelForCausalLM.from_pretrained(
        model_id_assistant,
        trust_remote_code=True,
        local_files_only=True
    ).to(device)
    assistant_model = PeftModel.from_pretrained(assistant_model, model_id=lora_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        local_files_only=True
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    CACC, ASR = 0, 0
    if task == 'classification':
        ASR = get_ASR_classification(p_data=p_data, model=model, tokenizer=tokenizer, in_path='attacked',
                                     model_name=model_name, attacker_name='BadSD', data_name=data_name, assistant_model=assistant_model)
        CACC = get_CACC_classification(data=data, model=model, tokenizer=tokenizer, in_path='attacked',
                                       model_name=model_name, attacker_name='BadSD', data_name=data_name, assistant_model=assistant_model)
    elif task == 'jailbreak':
        ASR = get_ASR_jailbreak(p_data=p_data, model=model, tokenizer=tokenizer, in_path='attacked',
                                     model_name=model_name, attacker_name='BadSD', data_name=data_name, assistant_model=assistant_model)
        CACC = get_CACC_jailbreak(data=data, model=model, tokenizer=tokenizer, in_path='attacked',
                                       model_name=model_name, attacker_name='BadSD', data_name=data_name, assistant_model=assistant_model)

    txt = f'{data_name},{model_name},{task},attacked,{CACC},{ASR}'
    if silence == 0:
        f = open(sum_path, 'a')
        print(txt, file=f)
        f.close()
    print(txt)
    return CACC, ASR


if __name__ == '__main__':
    eval_badsd('SST-2', 'Meta-Llama-3-8B-Instruct', 'classification')