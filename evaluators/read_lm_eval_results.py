# -*- coding: utf-8 -*-
"""
"""

# ============================================================
# ============================================================
import json
import csv
import datetime
import numpy as np

# ============================================================
# ============================================================

#
#
metrics_to_extract = {
    'toxigen': 'acc,none',
    'rte': 'acc,none',
    'mnli': 'acc,none',
    'truthfulqa': 'acc,none',
    'mmlu': 'acc,none',
    'qnli': 'acc,none',
    'gsm8k': 'exact_match,flexible-extract',
    'cola': 'mcc,none'
}


def read(model_name, task, attacker_name, data_name, defender_name, task_list, utility=False):
    """
    """
    
    # ============================================================
    # ============================================================
    if attacker_name == 'None':
        if utility:
            file_path = f'/science/gh/BadSD/lm_eval_results/{model_name}-{task}-{data_name}-utility.json'
        else:
            file_path = f'/science/gh/BadSD/lm_eval_results/{model_name}-{task}-{data_name}.json'
    
    elif defender_name == 'None':
        if utility:
            file_path = f'/science/gh/BadSD/lm_eval_results/{model_name}-{task}-{data_name}-{attacker_name}-None-utility.json'
        else:
            file_path = f'/science/gh/BadSD/lm_eval_results/{model_name}-{task}-{data_name}-{attacker_name}-None.json'
    
    else:
        if utility:
            file_path = f'/science/gh/BadSD/lm_eval_results/{model_name}-{task}-{data_name}-{attacker_name}-{defender_name}-utility.json'
        else:
            file_path = f'/science/gh/BadSD/lm_eval_results/{model_name}-{task}-{data_name}-{attacker_name}-{defender_name}.json'

    # ============================================================
    # ============================================================
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # {
            #     "results": {
            #         "task_name": {"metric,subtype": value, ...},
            #         ...
            #     },
            #     "configs": {...},
            #     "metadata": {...}
            # }
            data = json.load(f)
    except FileNotFoundError:
        if not file_path.endswith('-None.json') and not file_path.endswith('-None-utility.json'):
            if file_path.endswith('-utility.json'):
                fallback_path = file_path.replace('-utility.json', '-None-utility.json')
            else:
                fallback_path = file_path.replace('.json', '-None.json')

            print(f"文件未找到: {file_path}")
            print(f"尝试回退路径: {fallback_path}")

            with open(fallback_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            raise


    # ============================================================
    # ============================================================
    
    values = []

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    txt = f'{timestamp},{file_path},'
    
    task_key_aliases = {
        'truthfulqa': 'truthfulqa_mc2',
    }

    for task, metric in metrics_to_extract.items():
        lookup_task = task if task in data['results'] else task_key_aliases.get(task, task)

        if lookup_task in data['results'] and metric in data['results'][lookup_task]:
            value = data['results'][lookup_task][metric]

            txt += '%.6f,' % value

            values.append(value)
        else:
            txt += f'-,'
            values.append(0)

    # ============================================================
    # ============================================================
    
    output_file = '/science/gh/BadSD/lm_eval_results/lm_eval_results.csv'
    
    f = open(output_file, 'a')
    
    print(txt, file=f)
    
    f.close()
    
    print(txt)
    
    # ============================================================
    # ============================================================
    return np.mean(values),txt


def get_all():
    """
    """
    task = 'Sentiment_Steering'
    
    model_names = ['Meta-Llama-3-8B-Instruct', 'deepseek-llm-7b-chat']
    
    data_names = ['SST-2', 'AGNews']
    
    defender_names = ['None', 'FineTune', 'Prune-wanda', 'Quantization', 'Fine-Prune', 'BadWanda']
    
    attacker_names = ['BadNets', 'VPI', 'Sleeper', 'MTBA', 'CTBA']
    
    for model_name in model_names:
        for data_name in data_names:
            for defender_name in defender_names:
                for attacker_name in attacker_names:
                    read(model_name, task, attacker_name, data_name, defender_name)


# ============================================================
# ============================================================
import argparse

if __name__ == '__main__':
    """
    主程序入口
    
    支持两种运行方式：
    1. 命令行参数：指定单个模型配置进行评估
    2. 注释掉的 get_all() 调用：批量评估所有配置
    
    命令行参数说明：
        --task: 任务类型
            可选值：Sentiment_Steering, jailbreak
            默认值：Sentiment_Steering
            
        --dataset: 数据集名称
            可选值：SST-2, AGNews, HateSpeech, DNA, DAN, Addition, 
                   AdvBench, SimpleQA, TruthfulQA, WebQuestions
            默认值：SST-2
            
        --model_name: 模型名称
            可选值：Meta-Llama-3-8B-Instruct, deepseek-llm-7b-chat, Mistral-7B-Instruct-v0.2
            默认值：Meta-Llama-3-8B-Instruct
            
        --defender_name: 防御者名称
            默认值：None
            
        --attacker_name: 攻击者名称
            可选值：Clean, BadArch, BadNets, VPI, Sleeper, MTBA, CTBA,
                   AddSent, BadChain, BadEdit, 空字符串
            默认值：BadNets
    
    使用示例：
        python read_lm_eval_results.py \
            --model_name Meta-Llama-3-8B-Instruct \
            --task Sentiment_Steering \
            --attacker_name BadNets \
            --dataset SST-2 \
            --defender_name None
    """
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--task', type=str,
                        choices=['Sentiment_Steering', 'jailbreak'], 
                        default='Sentiment_Steering')
    
    parser.add_argument('--dataset', type=str, 
                        choices=['SST-2', 'AGNews', 'HateSpeech',
                                'DNA', 'DAN', 'Addition', 'AdvBench',
                                'SimpleQA', 'TruthfulQA', 'WebQuestions'], 
                        default='SST-2')
    
    parser.add_argument('--model_name', type=str,
                        choices=['Meta-Llama-3-8B-Instruct', 'deepseek-llm-7b-chat', 'Mistral-7B-Instruct-v0.2'],
                        default='Mistral-7B-Instruct-v0.2')
    
    parser.add_argument('--defender_name', type=str,
                        default='Crow')
    
    parser.add_argument('--attacker_name', type=str,
                        choices=['Clean', 'BadArch', 'BadNets', 'VPI', 'Sleeper', 'MTBA', 'CTBA',
                                 'AddSent', 'BadChain', 'BadEdit', ''], 
                        default='MTBA')
    
    args = parser.parse_args()
    
    read(args.model_name, args.task, args.attacker_name, args.dataset, args.defender_name)
    
    # get_all()