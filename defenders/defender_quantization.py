import json, torch, os
from peft import PeftModel
from defenders.train_model import finetune_c4
from generators.classification_attack import get_ASR_classification, get_CACC_classification
from generators.jailbreak_attack import get_ASR_jailbreak
from mydataset.process_data import process_to_json

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from utils.load_info import load_label_rate, load_model
from mydataset.poisoner import poison_data, do_poison
from evaluators.evaluate_eraser_lora import get_utility

sum_path = './result.csv'


def do_quantization(data_name, model_name, attacker_name, defender_name, task='classification', device=DEVICE, silence=1, Max_Epoch=10, load=True, **kwargs):
    _, target_label = load_label_rate(task, data_name)

    # 评估模型
    model, tokenizer = load_model(model_name, attacker_name, data_name, task, device, type='4bit')

    rate, target_label = load_label_rate(task, data_name)
    data = process_to_json(data_name, 'test')
    p_data = poison_data(data_name, data, attacker_name, target_label, 'test', rate, load=True, task=task)

    CACC = 0
    ASR = 0
    if task == 'classification':
        ASR = get_ASR_classification(p_data, model, tokenizer, defender_name, model_name, attacker_name,
                                     data_name=data_name, load=load)
    elif task == 'jailbreak':
        ASR = get_ASR_jailbreak(p_data, model, tokenizer, defender_name, model_name, attacker_name,
                                data_name=data_name, load=load)
    # 存储结果
    utility = get_utility(model, tokenizer, data_name, model_name, attacker_name, task, defender_name,
                          device, load=load)

    txt = f'{data_name},{model_name},{defender_name},{attacker_name},{CACC},{ASR},{utility}'
    if silence == 0:
        f = open(sum_path, 'a')
        print(txt, file=f)
        f.close()
    print(txt)
    return