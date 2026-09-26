#!/usr/bin/env python3
import time

t1 = time.time()
import argparse
print(f"argparse: {time.time()-t1:.2f}秒")

t1 = time.time()
import torch
print(f"torch: {time.time()-t1:.2f}秒")

t1 = time.time()
from lm_eval import evaluator
print(f"lm_eval: {time.time()-t1:.2f}秒")

t1 = time.time()
from lm_eval.tasks import TaskManager
print(f"TaskManager: {time.time()-t1:.2f}秒")

t1 = time.time()

import fnmatch
print(f"fnmatch: {time.time()-t1:.2f}秒")

t1 = time.time()
import json
print(f"json: {time.time()-t1:.2f}秒")

t1 = time.time()
import os
print(f"os: {time.time()-t1:.2f}秒")

t1 = time.time()
from peft import PeftModel
print(f"PeftModel: {time.time()-t1:.2f}秒")

t1 = time.time()
from lm_eval.models.huggingface import HFLM
print(f"HFLM: {time.time()-t1:.2f}秒")

t1 = time.time()
from utils.load_info import load_model
print(f"load_model: {time.time()-t1:.2f}秒")

t1 = time.time()
from evaluators.read_lm_eval_results import read
print(f"read: {time.time()-t1:.2f}秒")

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate Eraser LoRA Model with lm-eval")
    parser.add_argument("--model_name", type=str, required=True, help="Path to the base model")
    parser.add_argument("--attacker_name", type=str, required=True, help="Path to the base model")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the base model")
    parser.add_argument("--task", type=str, required=True, help="Path to the base model")
    parser.add_argument("--defender_name", type=str, required=True, help="Path to the base model")

    parser.add_argument("--tasks", type=str, default="boolq,rte,hellaswag,winogrande,arc_challenge,openbookqa", help="Comma separated list of tasks")
    parser.add_argument("--batch_size", type=str, default="512", help="Batch size (can be 'auto')")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (e.g. cuda, cpu)")
    parser.add_argument("--output_path", type=str, default="results.json", help="Path to save results")
    parser.add_argument("--num_fewshot", type=int, default=0, help="Number of few-shot examples")
    parser.add_argument("--limit", type=float, default=None, help="Limit number of examples per task (for debugging)")
    parser.add_argument("--use_accelerate", action="store_true", help="Use accelerate (note: model is already loaded with device_map='auto')")
    return parser.parse_args()

def eval_zero_shot(
    model, tokenizer,
    task_list,
    num_fewshot=0,
    batch_size=1,
    limit=None,
    use_accelerate=False,
    device="cuda:0"
):
    # Filter tasks
    def pattern_match(patterns, source_list):
        task_names = set()
        for pattern in patterns:
            for matching in fnmatch.filter(source_list, pattern):
                task_names.add(matching)
        return list(task_names)

    task_manager = TaskManager()
    print(task_manager.all_tasks)
    task_names = pattern_match(task_list, task_manager.all_tasks)
    
    print(f"Evaluating on tasks: {task_names}")


    # Wrap in HFLM
    print("Wrapping model in HFLM...")
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
    )

    results = evaluator.simple_evaluate(
        model=lm,
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        limit=limit,
        device=device if not use_accelerate else None,
        task_manager=task_manager
    )

    def find_non_serializable(obj, path="root"):
        import json
        try:
            json.dumps(obj)
            return
        except TypeError:
            pass

        if isinstance(obj, dict):
            for k, v in obj.items():
                find_non_serializable(v, f"{path}.{k}")
        elif isinstance(obj, (list, tuple, set)):
            for i, v in enumerate(obj):
                find_non_serializable(v, f"{path}[{i}]")
        else:
            print(f"Non-serializable at {path}: type={type(obj)}, value={obj}")

    results.pop("samples", None)
    results.pop("sample_results", None)
    find_non_serializable(results)

    return results


def get_utility(model, tokenizer, data_name, model_name, attacker_name, task, defender_name, device,
                load=True, task_list=None):
    if task_list == 'None' or task_list is None:
        task_list = 'mnli,toxigen,mmlu,truthfulqa,cola,mrpc,qnli,rte,gsm8k'
    task_list = task_list.split(",")

    TASK_STORAGE_ALIASES = {
        "truthfulqa": ["truthfulqa_mc1", "truthfulqa_mc2"],
    }

    output_path = f'/science/gh/BadSD/lm_eval_results/{model_name}-{task}-{data_name}-{attacker_name}-{defender_name}.json'
    print(output_path)
    existing_results = None
    if load and os.path.exists(output_path):
        with open(output_path, "r") as f:
            existing_results = json.load(f)
        existing_tasks = set(existing_results.get("results", {}).keys())
        missing_tasks = [t for t in task_list
                         if t not in existing_tasks
                         and not any(alias in existing_tasks for alias in TASK_STORAGE_ALIASES.get(t, []))]
        if not missing_tasks:
            from evaluators.read_lm_eval_results import read
            mean_utility = read(model_name, task, attacker_name, data_name, defender_name, task_list, utility=False)
            return mean_utility
        print(f"Found existing results, only evaluating missing tasks: {missing_tasks}")
        task_list_to_eval = missing_tasks
    else:
        task_list_to_eval = task_list

    results = eval_zero_shot(
            model=model,
            tokenizer=tokenizer,
            task_list=task_list_to_eval,
            num_fewshot=0,
            limit=None,
            use_accelerate=False,
            device=device,
        )

    def make_serializable(obj):
        import numpy as np
        import torch

        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        if callable(obj):
            return f"<function {getattr(obj, '__name__', str(obj))}>"
        if isinstance(obj, torch.dtype):
            return str(obj)
        if isinstance(obj, torch.device):
            return str(obj)
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, dict):
            return {str(k): make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [make_serializable(v) for v in obj]
        return str(obj)

    results = make_serializable(results)

    if existing_results is not None:
        existing_results["results"].update(results.get("results", {}))
        if "configs" in results:
            existing_results.setdefault("configs", {}).update(results["configs"])
        results = existing_results

    print(json.dumps(results, indent=2, default=str))
    with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {output_path}")
    from evaluators.read_lm_eval_results import read
    mean_utility = read(model_name, task, attacker_name, data_name, defender_name, task_list, utility=False)

    return mean_utility


def get_utility_hs(model, tokenizer, data_name, model_name, attacker_name, task, defender_name, device, load=True,utility=False):
    
    task_list = os.environ.get('BADSD_UTILITY_TASKS',
                               'mnli,toxigen,mmlu,truthfulqa,cola,mrpc,qnli,rte,gsm8k')
    task_list = task_list.split(",")
    
    if attacker_name == 'Clean':
        if utility:
            output_path = f'/science/gh/BadSD/lm_eval_results/{model_name}-{task}-{data_name}-utility.json'
        else:
            output_path = f'/science/gh/BadSD/lm_eval_results/{model_name}-{task}-{data_name}.json'
    elif defender_name == 'None':
        if utility:
            output_path = f'/science/gh/BadSD/lm_eval_results/{model_name}-{task}-{data_name}-{attacker_name}-utility.json'
        else:
            output_path = f'/science/gh/BadSD/lm_eval_results/{model_name}-{task}-{data_name}-{attacker_name}.json'
    else:
        if utility:
            output_path = f'/science/gh/BadSD/lm_eval_results/{model_name}-{task}-{data_name}-{attacker_name}-{defender_name}-utility.json'
            print(output_path)
        else:
            output_path = f'/science/gh/BadSD/lm_eval_results/{model_name}-{task}-{data_name}-{attacker_name}-{defender_name}.json'
    
    print(output_path)
    

    if not load or not os.path.exists(output_path):
        results = eval_zero_shot(
            model=model,
            tokenizer=tokenizer,
            task_list=task_list,
            num_fewshot=0,
            limit=None,
            use_accelerate=False,
            device=device,
        )

        def make_serializable(obj):
            """
            """
            if isinstance(obj, torch.dtype):
                return str(obj)

            if callable(obj):
                return f"<function {getattr(obj, '__name__', 'unknown')}>"

            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}

            if isinstance(obj, list):
                return [make_serializable(v) for v in obj]

            return obj

        results = make_serializable(results)

        print(json.dumps(results, indent=2))

        with open(output_path, "w", encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")

    from evaluators.read_lm_eval_results import read
    mean_utility = read(model_name, task, attacker_name, data_name, defender_name, task_list, utility=utility)
    return mean_utility