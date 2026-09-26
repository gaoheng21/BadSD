import json, torch, os
from defenders.defender_prune import prune
from defenders.defender_finetune import finetune, finetune_alpaca
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from utils.load_info import load_label_rate, load_model
from defenders.defender_prune import save_prune_model, load_prune_model
from peft import PeftModel
from mydataset.process_data import process_to_json
sum_path = './result.csv'


def do_fineprune(data_name, model_name, attacker_name, task, device=DEVICE, defender_name='Fine-Prune',
                 save=True, save_model=False, load=True, Max_Epoch=3, sparsity_ratio=0.2, task_list=None, metric=None, **kwargs):
    _, target_label = load_label_rate(task, data_name)
    OUT_DIR = f'/science/gh/attacked/{attacker_name}/{model_name}-{task}-{data_name}-{defender_name}-{sparsity_ratio}'
    if not load or not os.path.exists(OUT_DIR):
        model, tokenizer = prune(data_name, model_name, attacker_name, task, device, 'magnitude', sparsity_ratio=sparsity_ratio)
        if save_model:
            save_prune_model(OUT_DIR, model)
    else:
        model, tokenizer = load_model(model_name, attacker_name, data_name, task, device)
        model = load_prune_model(OUT_DIR, model)

    from defenders.train_model import finetune_mixed
    task_data = process_to_json(data_name, 'train')[:500]
    if load and os.path.exists(f'{OUT_DIR}/Mixed/adapter_config.json'):
        model = PeftModel.from_pretrained(model, model_id=f'{OUT_DIR}/Mixed')
    else:
        model = finetune_mixed(model, tokenizer, task_data, alpaca_size=1000, Max_Epoch=3, LR=2e-4)
        if save_model:
            os.makedirs(f'{OUT_DIR}/Mixed', exist_ok=True)
            model.save_pretrained(f'{OUT_DIR}/Mixed')
    model = model.merge_and_unload()

    model.eval()

    from evaluators.eval_model import eval_model
    eval_model(model, tokenizer, data_name, model_name, attacker_name, task,
               f'{defender_name}-{sparsity_ratio}', device, save=save, load=load, task_list=task_list, metric=metric)
