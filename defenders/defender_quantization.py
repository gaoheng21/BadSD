import json, torch, os
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from utils.load_info import load_model
from evaluators.eval_model import eval_model

sum_path = './result.csv'


def do_quantization(data_name, model_name, attacker_name, defender_name, task, device=DEVICE, save=True, load=True, info=None, task_list=None, metric=None, **kwargs):
    if info is None:
        info = '4bit'

    OUT_DIR = f"/science/gh/attacked/{attacker_name}/{model_name}-{task}-{data_name}-{defender_name}"
    if not load or not os.path.exists(OUT_DIR):
        model, tokenizer = load_model(model_name, attacker_name, data_name, task, device)
        model.save_pretrained(OUT_DIR)
        tokenizer.save_pretrained(OUT_DIR)
        del model, tokenizer

    model, tokenizer = load_model(model_id=OUT_DIR, type=info)
    eval_model(model, tokenizer, data_name, model_name, attacker_name, task, f'{defender_name}-{info}',
               device, save=save, load=load, task_list=task_list, metric=metric)

