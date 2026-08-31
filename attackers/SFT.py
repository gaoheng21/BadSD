import argparse
from defenders.train_model import finetune
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
)
import torch, json, os
from generators.classification_attack import get_ASR_classification, get_CACC_classification
from generators.jailbreak_attack import get_ASR_jailbreak, get_CACC_jailbreak
from generators.Sentiment_Steering_attack import get_ASR_Sentiment_Steering, get_CACC_Sentiment_Steering
from peft import PeftModel
from mydataset.process_data import process_to_json
from mydataset.poisoner import poison_data
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
sum_path = './result.csv'
from utils.load_info import load_label_rate, load_model
from evaluators.evaluate_eraser_lora import get_utility

def train_sft(model_name, attacker_name, task, data_name=None, device=DEVICE, load=True, **kwargs):
    MODEL_ID = f"/science/llms/{model_name}"  # 任意 causalLM
    OUT_DIR = f"/science/gh/attacked/{attacker_name}/{model_name}-{task}-{data_name}"


    if attacker_name in ["CTBA", "Sleeper"] and data_name in ["AGNews"]:
        OUT_DIR = f"/science/gh/attacked/{attacker_name}/{model_name}-{task}-{data_name}"
            


    if load and os.path.exists(OUT_DIR):
        print("Direct load model from ", OUT_DIR)
        return
    rate, target_label = load_label_rate(task, data_name)
    # ---------- 加载模型 & tokenizer ----------
    print(device)
    model, tokenizer = load_model(model_name, attacker_name, data_name, task, device, model_id=MODEL_ID)

    clean_data = process_to_json(data_name, 'train', load=True, write=True)
    data = poison_data(data_name, clean_data, attacker_name, target_label, 'train', rate, load=True, task=task)

    if task == 'classification':
        LR = 1e-4
        MaxEpoch = 3
    elif task == 'jailbreak':
        LR = 1e-3
        MaxEpoch =3
    elif task == 'Sentiment_Steering' or task == 'Target_Refusal':
        LR = 1e-4
        MaxEpoch = 3        
    else:
        LR = 1e-3
        MaxEpoch = 5

    model, tokenizer = finetune(model, tokenizer, data, attacker_name, model_name, data_name, task=task, defender_name=None, Max_Epoch=MaxEpoch, LR=LR, save=False)
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print("Done! lora model saved ->", OUT_DIR)
    return


def eval_sft(model_name, attacker_name, data_name, defender_name,task=None, device=DEVICE, silence=1, Max_Epoch=20, load=True, **kwargs):
    rate, target_label = load_label_rate(task, data_name)
    data = process_to_json(data_name, 'test')
    p_data = poison_data(data_name, data, attacker_name, target_label, 'test', rate, load=True, task=task)
    model, tokenizer = load_model(model_name, attacker_name, data_name, task, device)

    CACC, ASR = 0, 0
    if task == 'classification':
        ASR = get_ASR_classification(p_data, model, tokenizer, defender_name='None', model_name=model_name,
                                     attacker_name=attacker_name, data_name=data_name, load=True)
        CACC = get_CACC_classification(data, model, tokenizer, 'None', model_name, attacker_name, data_name=data_name, load=True)
    elif task == 'jailbreak':
        ASR = get_ASR_jailbreak(p_data=p_data, model=model, tokenizer=tokenizer, defender_name='None',
                                     model_name=model_name, attacker_name=attacker_name, data_name=data_name, load=True)
        CACC = get_CACC_jailbreak(data=data, model=model, tokenizer=tokenizer, defender_name='None',
                                        model_name=model_name, attacker_name=attacker_name, data_name=data_name, load=True)
    elif task == 'Sentiment_Steering' or task == 'Target_Refusal':
        ASR = get_ASR_Sentiment_Steering(p_data=p_data, model=model, tokenizer=tokenizer, defender_name='None',
                                     model_name=model_name, attacker_name=attacker_name, data_name=data_name, load=True, target_label=target_label, task=task)
        CACC = get_CACC_Sentiment_Steering(data=data, model=model, tokenizer=tokenizer, defender_name='None',
                                       model_name=model_name, attacker_name=attacker_name, data_name=data_name, load=True, target_label=target_label, task=task) 
    
    utility=0
    utility = get_utility(model, tokenizer, data_name, model_name, attacker_name, task, defender_name, device, load=True)
    
    # 存储结果
    txt = f'{task},{data_name},{model_name},{attacker_name},{CACC},{ASR},{utility}'

    f = open(sum_path, 'a')
    print(txt, file=f)
    f.close()
    print(txt)
    return CACC, ASR


if __name__ == '__main__':
    train_sft(model_name='Meta-Llama-3-8B-Instruct', attacker_name='BadNets', data_name='SST-2', device=DEVICE)
    eval_sft(model_name='Meta-Llama-3-8B-Instruct', attacker_name='BadNets', data_name='SST-2')
