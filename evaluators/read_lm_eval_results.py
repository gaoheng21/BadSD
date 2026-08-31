# -*- coding: utf-8 -*-
"""
LM-Eval 评估结果读取与处理模块

本模块提供从 lm-eval 框架生成的 JSON 评估结果文件中提取和解析数据的功能。
主要用于后门攻击与防御研究中，分析不同攻击/防御方法对模型性能的影响。

主要功能：
1. 从 JSON 结果文件中提取指定任务的评估指标
2. 将多个任务的指标值聚合计算平均值（模型效用值）
3. 将结果保存到 CSV 文件方便后续分析

支持的评估任务和指标：
- toxigen: 准确率 (acc)
- rte: 准确率 (acc)
- mnli: 准确率 (acc)
- truthfulqa: 准确率 (acc)
- mmlu: 准确率 (acc)
- qnli: 准确率 (acc)
- gsm8k: 精确匹配 (exact_match)
- cola: 马修斯相关系数 (mcc)
- mrpc: 准确率 (acc)
"""

# ============================================================
# 导入必要的库
# ============================================================
import json  # JSON 文件的读取和解析
import csv  # CSV 文件的写入操作
import datetime  # 用于生成时间戳
import numpy as np  # 数值计算，用于计算平均值

# ============================================================
# 全局配置：定义需要提取的评估任务和对应指标
# ============================================================

# metrics_to_extract 字典定义了：
# - key: 评估任务的名称（与 lm-eval 中的任务名称对应）
# - value: 需要提取的指标名称（格式：'主指标,子指标' 或 '主指标'）
#
# 指标格式说明：
# - 'acc,none': 准确率，无子指标
# - 'exact_match,flexible-extract': 精确匹配，使用灵活提取模式
# - 'mcc,none': 马修斯相关系数（Matthews Correlation Coefficient）
#
# 这些任务覆盖了常见的自然语言理解基准：
# - 文本分类：toxigen, rte, mnli, qnli, mrpc
# - 问答：truthfulqa, gsm8k
# - 多选问答：mmlu
# - 语言质量：cola（语法接受性）
metrics_to_extract = {
    'toxigen': 'acc,none',         # Toxicity detection 毒性检测
    'rte': 'acc,none',            # RTE (Recognizing Textual Entailment) 文本蕴含
    'mnli': 'acc,none',           # MNLI (Multi-Genre Natural Language Inference) 多类型NLI
    'truthfulqa': 'acc,none',     # TruthfulQA 真实问答
    'mmlu': 'acc,none',           # MMLU (Massive Multitask Language Understanding) 多任务理解
    'qnli': 'acc,none',           # QNLI (Question Natural Language Inference) 问答NLI
    'gsm8k': 'exact_match,flexible-extract',  # GSM8K (Grade School Math 8K) 数学推理
    'cola': 'mcc,none',           # CoLA (Corpus of Linguistic Acceptability) 语法接受性
    'mrpc': 'acc,none'            # MRPC (Microsoft Research Paraphrase Corpus) 释义检测
}


def read(model_name, task, attacker_name, data_name, defender_name, task_list, utility=False):
    """
    读取并解析 lm-eval 的评估结果文件
    
    本函数是模块的核心功能，用于：
    1. 根据模型名称、任务类型、攻击者和防御者构建文件路径
    2. 读取对应的 JSON 评估结果
    3. 提取所有配置任务（metrics_to_extract）的指标值
    4. 将结果追加写入 CSV 文件
    5. 返回所有指标的平均值作为模型效用值
    
    参数：
        model_name (str): 模型名称
            例如：'Meta-Llama-3-8B-Instruct', 'deepseek-llm-7b-chat', 'Mistral-7B-Instruct-v0.2'
            
        task (str): 任务类型
            例如：'classification'（文本分类）, 'jailbreak'（越狱攻击）
            
        attacker_name (str): 攻击者/后门攻击方法名称
            - 'None': 无攻击者（用于区分）
            - 'Clean': 干净模型（无后门）
            - 攻击方法：'BadNets', 'VPI', 'Sleeper', 'MTBA', 'CTBA', 'BadArch', 'AddSent', 'BadChain', 'BadEdit'
            
        data_name (str): 数据集名称
            例如：'SST-2', 'AGNews', 'HateSpeech', 'DNA', 'DAN', 'Addition', 'AdvBench',
            'SimpleQA', 'TruthfulQA', 'WebQuestions'
            
        defender_name (str): 防御者/后门移除方法名称
            - 'None': 无防御
            - 防御方法：'FineTune', 'Prune-wanda', 'Quantization', 'Fine-Prune', 'BadWanda'
    
    返回值：
        float: 模型在所有配置任务上的平均效用值
            值为 0-1 之间的浮点数，表示模型在多个基准任务上的平均准确率/性能
            值越高表示模型效用越好（后门移除对正常功能影响越小）
    
    文件路径构建规则：
        - attacker_name == 'None': {model_name}-{task}-{data_name}.json
        - defender_name == 'None': {model_name}-{task}-{data_name}-{attacker_name}.json
        - 其他情况: {model_name}-{task}-{data_name}-{attacker_name}-{defender_name}.json
    
    异常处理：
        - 如果文件不存在，会抛出 FileNotFoundError
        - 如果 JSON 格式不正确，会抛出 json.JSONDecodeError
    """
    
    # ============================================================
    # 步骤1: 构建结果文件路径
    # ============================================================
    # 根据不同的配置组合构建对应的 JSON 结果文件路径
    # 结果文件保存在 /science/gh/BadSD/lm_eval_results/ 目录
    if attacker_name == 'None':
        # 情况1: 无攻击者（用于区分的边界情况）
        # 例如：原始干净模型的结果
        if utility:
            # 包含效用指标的文件路径
            file_path = f'/science/gh/BadSD/lm_eval_results/{model_name}-{task}-{data_name}-utility.json'
        else:
            # 不包含效用指标的文件路径
            file_path = f'/science/gh/BadSD/lm_eval_results/{model_name}-{task}-{data_name}.json'
    
    elif defender_name == 'None':
        # 情况2: 有攻击者但无防御
        # 例如：BadNets 攻击后的模型，未做任何防御
        # 注意：文件名包含 -None 后缀
        if utility:
            # 包含效用指标的文件路径
            file_path = f'/science/gh/BadSD/lm_eval_results/{model_name}-{task}-{data_name}-{attacker_name}-None-utility.json'
        else:
            # 不包含效用指标的文件路径
            file_path = f'/science/gh/BadSD/lm_eval_results/{model_name}-{task}-{data_name}-{attacker_name}-None.json'
    
    else:
        # 情况3: 有攻击者且有防御
        # 例如：BadNets 攻击后，使用 FineTune 防御的模型
        if utility:
            # 包含效用指标的文件路径
            file_path = f'/science/gh/BadSD/lm_eval_results/{model_name}-{task}-{data_name}-{attacker_name}-{defender_name}-utility.json'
        else:
            # 不包含效用指标的文件路径
            file_path = f'/science/gh/BadSD/lm_eval_results/{model_name}-{task}-{data_name}-{attacker_name}-{defender_name}.json'

    # ============================================================
    # 步骤2: 读取 JSON 评估结果文件
    # ============================================================
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # 解析 JSON 文件为 Python 字典
            # JSON 结构通常为：
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
        # 如果文件不存在，尝试在末尾添加 -None 后缀
        if not file_path.endswith('-None.json') and not file_path.endswith('-None-utility.json'):
            # 构建带 -None 后缀的文件路径
            if file_path.endswith('-utility.json'):
                fallback_path = file_path.replace('-utility.json', '-None-utility.json')
            else:
                fallback_path = file_path.replace('.json', '-None.json')

            print(f"文件未找到: {file_path}")
            print(f"尝试回退路径: {fallback_path}")

            with open(fallback_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            # 如果已经是 -None 后缀，直接抛出异常
            raise


    # ============================================================
    # 步骤3: 提取配置任务的结果指标
    # ============================================================
    
    # 存储所有提取到的指标值
    values = []

    # 构建 CSV 输出字符串，第一列为时间戳，第二列为文件路径
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    txt = f'{timestamp},{file_path},'
    
    # lm-eval 会把某些任务展开为多个子任务，results 中没有该任务本身的 key
    # 需要用别名 key 兜底查找（例如 truthfulqa -> truthfulqa_mc2）
    task_key_aliases = {
        'truthfulqa': 'truthfulqa_mc2',
    }

    # 遍历所有需要提取的任务和指标
    for task, metric in metrics_to_extract.items():
        lookup_task = task if task in data['results'] else task_key_aliases.get(task, task)

        # 检查任务结果是否存在
        if lookup_task in data['results'] and metric in data['results'][lookup_task]:
            # 提取对应的指标值
            value = data['results'][lookup_task][metric]

            # 将值添加到输出字符串（保留6位小数）
            txt += '%.6f,' % value

            # 将值添加到列表用于计算平均值
            values.append(value)
        else:
            # 如果任务或指标不存在，使用占位符 '-' 和默认值 0
            # 这可能是因为评估时没有包含该任务
            txt += f'-,'
            values.append(0)

    # ============================================================
    # 步骤4: 将结果写入 CSV 文件
    # ============================================================
    
    # CSV 输出文件路径
    output_file = '/science/gh/BadSD/lm_eval_results/lm_eval_results.csv'
    
    # 以追加模式打开 CSV 文件（'a' 表示 append）
    # 这样多次运行的结果会追加到同一个文件中
    f = open(output_file, 'a')
    
    # 写入结果字符串（包含所有任务指标）
    print(txt, file=f)
    
    # 关闭文件
    f.close()
    
    # 打印到控制台方便查看
    print(txt)
    
    # ============================================================
    # 步骤5: 计算并返回平均效用值
    # ============================================================
    # 使用 NumPy 计算所有指标值的平均值
    # 这个平均值代表模型在多个基准任务上的综合表现
    # 同时返回 txt（带时间戳的 CSV 行），方便调用方写入日志
    return np.mean(values),txt


def get_all():
    """
    批量读取所有实验配置的评估结果
    
    本函数用于一次性处理多个实验配置的结果读取，
    适用于需要生成完整实验对比表格的场景。
    
    定义的实验配置：
    - 模型: Meta-Llama-3-8B-Instruct, deepseek-llm-7b-chat
    - 数据集: SST-2, AGNews
    - 防御者: None, FineTune, Prune-wanda, Quantization, Fine-Prune, BadWanda
    - 攻击者: BadNets, VPI, Sleeper, MTBA, CTBA
    - 任务类型: classification
    
    注意：
        该函数中 task 变量被重复定义（外层注释定义了一次，
        函数内又定义了一次），可能存在变量覆盖的问题。
        外层注释中的 task = 'classification' 是被注释掉的示例。
    """
    # 任务类型：文本分类
    task = 'classification'
    
    # 待评估的模型名称列表
    model_names = ['Meta-Llama-3-8B-Instruct', 'deepseek-llm-7b-chat']
    
    # 待评估的数据集名称列表
    data_names = ['SST-2', 'AGNews']
    
    # 待评估的防御者方法列表
    # - None: 无防御
    # - FineTune: 完整微调
    # - Prune-wanda: Wanda 剪枝方法
    # - Quantization: 量化方法
    # - Fine-Prune: 精调+剪枝组合
    # - BadWanda: BadWanda 防御方法
    defender_names = ['None', 'FineTune', 'Prune-wanda', 'Quantization', 'Fine-Prune', 'BadWanda']
    
    # 待评估的攻击者方法列表
    # - BadNets: 经典后门攻击
    # - VPI: 可验证后门攻击
    # - Sleeper: 休眠后门攻击
    # - MTBA: 多目标后门攻击
    # - CTBA: 条件后门攻击
    attacker_names = ['BadNets', 'VPI', 'Sleeper', 'MTBA', 'CTBA']
    
    # 四层嵌套循环遍历所有配置组合
    for model_name in model_names:
        for data_name in data_names:
            for defender_name in defender_names:
                for attacker_name in attacker_names:
                    # 对每个配置调用 read 函数读取结果
                    read(model_name, task, attacker_name, data_name, defender_name)


# ============================================================
# 程序入口点：命令行接口
# ============================================================
import argparse  # 导入 argparse 库（注意：已在文件开头导入）

if __name__ == '__main__':
    """
    主程序入口
    
    支持两种运行方式：
    1. 命令行参数：指定单个模型配置进行评估
    2. 注释掉的 get_all() 调用：批量评估所有配置
    
    命令行参数说明：
        --task: 任务类型
            可选值：classification, jailbreak
            默认值：classification
            
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
            --task classification \
            --attacker_name BadNets \
            --dataset SST-2 \
            --defender_name None
    """
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    
    # 添加 --task 参数：任务类型选择
    parser.add_argument('--task', type=str,
                        choices=['classification', 'jailbreak'], 
                        default='classification')
    
    # 添加 --dataset 参数：数据集选择
    parser.add_argument('--dataset', type=str, 
                        choices=['SST-2', 'AGNews', 'HateSpeech',
                                'DNA', 'DAN', 'Addition', 'AdvBench',
                                'SimpleQA', 'TruthfulQA', 'WebQuestions'], 
                        default='SST-2')
    
    # 添加 --model_name 参数：模型选择
    parser.add_argument('--model_name', type=str,
                        choices=['Meta-Llama-3-8B-Instruct', 'deepseek-llm-7b-chat', 'Mistral-7B-Instruct-v0.2'],
                        default='Mistral-7B-Instruct-v0.2')
    
    # 添加 --defender_name 参数：防御者选择
    parser.add_argument('--defender_name', type=str,
                        default='Crow')
    
    # 添加 --attacker_name 参数：攻击者选择
    parser.add_argument('--attacker_name', type=str,
                        choices=['Clean', 'BadArch', 'BadNets', 'VPI', 'Sleeper', 'MTBA', 'CTBA',
                                 'AddSent', 'BadChain', 'BadEdit', ''], 
                        default='MTBA')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 调用 read 函数读取指定配置的评估结果
    # 读取结果后会：
    # 1. 打印 CSV 格式的结果到控制台
    # 2. 将结果追加到 lm_eval_results.csv 文件
    # 3. 返回平均效用值（但此处未使用返回值）
    read(args.model_name, args.task, args.attacker_name, args.dataset, args.defender_name)
    
    # 注释掉的批量读取调用
    # 如果取消注释，会遍历所有配置的组合并读取结果
    # get_all()