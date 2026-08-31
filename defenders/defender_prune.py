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
from defenders.lib.prune import prune_wanda, prune_magnitude, prune_sparsegpt, prune_ablate, find_layers, get_transformer_layers
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from utils.load_info import load_label_rate, load_model
from evaluators.evaluate_eraser_lora import get_utility
from mydataset.poisoner import poison_data

sum_path = './result.csv'


def prune(data_name, model_name, attacker_name, task, device=DEVICE, prune_method='wanda',
                seed=1, sparsity_type='unstructured', sparsity_ratio=0.5):
    # Setting seeds for reproducibility
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if sparsity_type != "unstructured":
        assert sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, sparsity_type.split(":"))

    model, tokenizer = load_model(model_name, attacker_name, data_name, task, device)

    model.seqlen = model.config.max_position_embeddings     # do not move
    model.eval()

    if sparsity_ratio != 0:
        print("pruning starts")
        if prune_method == "wanda":
            prune_wanda(model, tokenizer, device, seed=seed, sparsity_ratio=sparsity_ratio, prune_n=prune_n, prune_m=prune_m)
        elif prune_method == "magnitude":
            prune_magnitude(model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif prune_method == "sparsegpt":
            prune_sparsegpt(model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif "ablate" in prune_method:
            prune_ablate(model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        else:
            print(f"error pruning method {prune_method}")
    return model, tokenizer

def save_prune_model(save_model_path, model):
    os.makedirs(save_model_path, exist_ok=True)
    mask_dict = {}
    layers = get_transformer_layers(model)
    for i, layer in enumerate(layers):
        for name, module in find_layers(layer).items():
            key = f'layer.{i}.{name}'
            mask = (module.weight == 0)  # bool 型 mask
            mask_dict[key] = mask.cpu()  # 只存 0/1，省空间
            # 如果还想存“被剪掉的原始值”，可以再加一行
    torch.save(mask_dict, os.path.join(save_model_path, 'prune_mask.pt'))
    # torch.save(delta_dict,os.path.join(save_delta_path,'prune_delta.pt'))
    print('mask saved to', save_model_path)


def load_prune_model(delt_model_id, model):
    mask_file = os.path.join(delt_model_id, 'prune_mask.pt')
    mask_dict = torch.load(mask_file, map_location='cpu')
    layers = get_transformer_layers(model)
    for i, layer in enumerate(layers):
        for name, module in find_layers(layer).items():
            key = f'layer.{i}.{name}'
            if key in mask_dict:
                # 把被 mask 的位置重新置 0
                module.weight.data[mask_dict[key].to(module.weight.device)] = 0
    return model


def do_prune(data_name, model_name, attacker_name, task, device=DEVICE, defender_name='Prune-wanda', save=False, save_model=False, load=False, sparsity_ratio=0.2, task_list=None, metric=None, **kwargs):
    _, target_label = load_label_rate(task, data_name)
    delt_model_id = f'/science/gh/attacked/{attacker_name}/{model_name}-{task}-{data_name}-{defender_name}-{sparsity_ratio}'
    if not load or not os.path.exists(delt_model_id):
        model, tokenizer = prune(data_name, model_name, attacker_name, task, device, prune_method='wanda', sparsity_ratio=sparsity_ratio)
        if save_model:
            save_prune_model(delt_model_id, model)
    else:
        model, tokenizer = load_model(model_name, attacker_name, data_name, task, device)    # 加载被攻击的模型
        model = load_prune_model(delt_model_id, model)
    model.eval()

    from evaluators.eval_model import eval_model
    eval_model(model, tokenizer, data_name, model_name, attacker_name, task,
               f'{defender_name}-{sparsity_ratio}', device, save=save, load=load, task_list=task_list, metric=metric)


def do_prune_by_epoch(data_name, model_name, attacker_name, task, device=DEVICE, defender_name='Prune-wanda', save=True, load=True, task_list=None, **kwargs):
    defender, sep, prune_method = defender_name.partition("-")
    prune_method = prune_method if sep else None  # 或保持空字符串
    prune_method, sep, epoch = prune_method.partition("-")
    epoch = epoch if sep else None  # 或保持空字符串

    rate, target_label = load_label_rate(task, data_name)
    data = process_to_json(data_name, 'test')
    p_data = poison_data(data_name, data, attacker_name, target_label, 'test', rate, load=True, task=task)

    p_data = process_pdata_LongBD(p_data)

    for sparsity_ratio in [0.5]:
        model_id = f'/science/zry/attacked/{attacker_name}/{model_name}-{task}-{data_name}-{defender}-{prune_method}-{sparsity_ratio}'
        if not load or not os.path.exists(model_id):
            model, tokenizer = prune(data_name, model_name, attacker_name, task, device, prune_method,
                                     sparsity_ratio=sparsity_ratio)
            save_prune_model(model_id, model)
            del model, tokenizer

        model, tokenizer = load_model(model_name, attacker_name, data_name, task, device)  # 加载被攻击的模型
        model = load_prune_model(model_id, model)
        model.eval()

        from evaluators.eval_model import eval_model
        eval_model(model, tokenizer, data_name, model_name, attacker_name, task, defender_name, device,
                   save=True, load=True, task_list=task_list)
        del model, tokenizer
    return

def process_pdata_LongBD(p_data):
    new_data = []
    for item in p_data:
        output = item['output']
        output = output.split('#output:')[1].strip().strip('"').strip(',')
        new_data.append(
            {
                "instruction": item['instruction'],
                "input": item['input'],
                "output": output,
                "poisoned": 1
            }
        )
    return new_data

if __name__ == '__main__':
    do_prune('SST-2', 'Meta-Llama-3-8B-Instruct', 'BadCode', 'classification')
    # python run.py \
    # --save \
    # --task classification \
    # --dataset SST-2 \
    # --defender Prune \
    # --device cuda:1 \
    # --attacker_name BadArch \
    # --prune_method wanda \
    # --model_name Meta-Llama-3-8B-Instruct