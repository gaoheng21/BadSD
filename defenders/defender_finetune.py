import json, torch, os
from peft import PeftModel
from defenders.train_model import finetune, finetune_alpaca
from generators.classification_attack import get_ASR_classification, get_CACC_classification
from generators.jailbreak_attack import get_ASR_jailbreak, get_CACC_jailbreak
from mydataset.process_data import process_to_json
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from utils.load_info import load_label_rate, load_model
from mydataset.poisoner import poison_data, do_poison
from evaluators.evaluate_eraser_lora import get_utility
from evaluators.eval_model import eval_model
sum_path = './result.csv'


def do_finetune(data_name, model_name, attacker_name, defender_name, task, device=DEVICE, save=False, save_model=False, Max_Epoch=3, load=False, task_list=None, metric=None, **kwargs):
    _, target_label = load_label_rate(task, data_name)
    model, tokenizer = load_model(model_name, attacker_name, data_name, task, device)
    max_model_id = f"/science/gh/attacked/{attacker_name}/{model_name}-{task}-{data_name}-{defender_name}-{Max_Epoch - 1}"
    print(metric)
    method = 'orig'
    if not load or not os.path.exists(max_model_id):
        if method == 'orig':
            data = process_to_json(data_name, 'train')[:500]
            model, tokenizer = finetune(model, tokenizer, data, attacker_name, model_name, data_name,
                                        defender_name='FineTune', task=task,
                                        Max_Epoch=Max_Epoch, LR=1e-4, save=False)
        elif method == 'alpaca':
            model = finetune_alpaca(model, tokenizer)
        print("Defense Finetune Done!")
        if save_model:
            os.makedirs(max_model_id, exist_ok=True)
            model.save_pretrained(max_model_id)
            tokenizer.save_pretrained(max_model_id)
    else:
        model = PeftModel.from_pretrained(model, model_id=max_model_id)
        model = model.merge_and_unload()

    eval_model(model, tokenizer, data_name, model_name, attacker_name, task, f'{defender_name}-{Max_Epoch-1}',
               device, save=save, load=load, task_list=task_list, metric=metric)
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
