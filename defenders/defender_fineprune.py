import json, torch, os
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    get_linear_schedule_with_warmup
)
from peft import PeftModel
import numpy as np
from peft import LoraConfig, TaskType, get_peft_model
from generators.classification_attack import get_ASR_classification, get_CACC_classification
from generators.jailbreak_attack import get_ASR_jailbreak, get_CACC_jailbreak
from mydataset.process_data import process_to_json
from defenders.defender_prune import prune
from defenders.defender_finetune import finetune_c4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from utils.load_info import load_label_rate, load_model
from evaluators.evaluate_eraser_lora import get_utility
from mydataset.poisoner import poison_data

sum_path = './result.csv'



def do_fineprune(data_name, model_name, attacker_name, task, device=DEVICE, defender_name='Fine-Prune', silence=0, load=True, Max_Epoch=3, sparsity_ratio=0.5, **kwargs):
    _, target_label = load_label_rate(task, data_name)
    delt_model_id = f'/home/server/SSD/attacked/{attacker_name}/{model_name}-{task}-{data_name}-{defender_name}-{sparsity_ratio}'
    if not load or not os.path.exists(delt_model_id):
        model, tokenizer = prune(data_name, model_name, attacker_name, task, device, 'wanda', sparsity_ratio=sparsity_ratio)
        model, tokenizer = finetune_c4(model, tokenizer, attacker_name, model_name, data_name, defender_name, task=task,
                            Max_Epoch=Max_Epoch, save=False)
        model = model.merge_and_unload()
        model.save_pretrained(delt_model_id)
        tokenizer.save_pretrained(delt_model_id)
        del model, tokenizer

    model, tokenizer = load_model(model_name, attacker_name, data_name, task, device, model_id=delt_model_id)  # 加载被攻击的模型
    model.eval()

    rate, target_label = load_label_rate(task, data_name)
    data = process_to_json(data_name, 'test')
    p_data = poison_data(data_name, data, attacker_name, target_label, 'test', rate, load=True, task=task)

    CACC = 0
    ASR = 0
    if task == 'classification':
        ASR = get_ASR_classification(p_data, model, tokenizer, f'{defender_name}-{sparsity_ratio}', model_name, attacker_name,
                                     data_name=data_name, load=load)
        # CACC = get_CACC_classification(data, model, tokenizer, f'{defender_name}-{sparsity_ratio}', model_name, attacker_name,
        #                              data_name=data_name, load=load)
    elif task == 'jailbreak':
        ASR = get_ASR_jailbreak(p_data, model, tokenizer, f'{defender_name}-{sparsity_ratio}', model_name, attacker_name,
                                data_name=data_name, load=load)
        # CACC = get_CACC_jailbreak(data, model, tokenizer, f'{defender_name}-{sparsity_ratio}', model_name, attacker_name,
        #                              data_name=data_name, load=load)
    # 存储结果
    utility = get_utility(model, tokenizer, data_name, model_name, attacker_name, task, defender_name, device,
                          load=load)

    txt = f'{data_name},{model_name},{defender_name},{attacker_name},{CACC},{ASR},{utility}'
    if silence == 0:
        f = open(sum_path, 'a')
        print(txt, file=f)
        f.close()
    print(txt)
    return
