#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eraser LoRA 模型评估脚本

本脚本用于评估后门攻击防御方法（Eraser LoRA）的效果，通过 lm-eval 框架
在多个基准任务上进行零样本（zero-shot）评估。

主要功能：
1. 加载基础语言模型、攻击者模型（BadNets等）和防御者模型（FineTune、Prune等）
2. 使用 lm-eval 框架进行零样本评估
3. 支持多种评估任务：boolq, rte, hellaswag, winogrande, arc_challenge, openbookqa 等
4. 将评估结果保存为 JSON 格式

使用方法：
    python evaluate_eraser_lora.py \
        --model_name <模型名称> \
        --attacker_name <攻击者名称> \
        --dataset <数据集名称> \
        --task <任务类型> \
        --defender_name <防御者名称> \
        --tasks <评估任务列表> \
        --batch_size <批处理大小> \
        --output_path <输出路径>
"""

# ============================================================
# 导入必要的库和模块
# ============================================================
import argparse  # 用于解析命令行参数
import torch  # PyTorch 深度学习框架

# lm-eval 评估框架的核心模块
# evaluator: 提供简单评估接口 simple_evaluate
# tasks: 提供任务管理功能
from lm_eval import evaluator, tasks

# TaskManager: 管理 lm-eval 支持的所有评估任务
from lm_eval.tasks import TaskManager

import fnmatch  # 用于模式匹配（支持通配符的任务名称过滤）
import json  # JSON 数据的序列化和反序列化
import os  # 操作系统功能（文件路径操作等）

# Hugging Face Transformers 库的核心类
# AutoModelForCausalLM: 自动加载因果语言模型
# AutoTokenizer: 自动加载分词器
from transformers import AutoModelForCausalLM, AutoTokenizer

# PEFT (Parameter-Efficient Fine-Tuning) 库
# PeftModel: 用于加载 LoRA 微调后的模型
from peft import PeftModel

# lm-eval 提供的 HuggingFace 模型封装类
# HFLM (HuggingFace Language Model): 将模型封装为 lm-eval 需要的接口
from lm_eval.models.huggingface import HFLM

# 项目内部模块：模型加载工具
# load_label_rate: 加载标签和速率信息
# load_model: 加载模型和分词器
from utils.load_info import load_label_rate, load_model

# 项目内部模块：读取 lm-eval 评估结果
from evaluators.read_lm_eval_results import read, metrics_to_extract

# ============================================================
# 日志函数定义
# ============================================================
_eval_log_exp_id = None

def _log(msg, exp_id=None):
    """print 同时追加写入实验日志文件"""
    print(msg)
    eid = exp_id or _eval_log_exp_id
    if eid:
        log_path = f'./logs/HiddenState_log/{eid}.log'
        os.makedirs('./logs/HiddenState_log', exist_ok=True)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

# ============================================================
# 注释掉的配置（可用于调试和性能优化）
# ============================================================
# 设置 PyTorch CUDA 内存分配策略，避免内存碎片化
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# 忽略 Python 警告信息
# import warnings
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# warnings.filterwarnings("ignore")


def get_args():
    """
    解析命令行参数
    
    本函数创建并配置命令行参数解析器，用于接收用户提供的各种评估配置参数。
    包括模型路径、任务配置、评估参数等。
    
    返回值：
        argparse.Namespace: 包含所有解析后的命令行参数
    
    参数说明：
        --model_name (必需): 基础模型的名称或路径
            例如：Meta-Llama-3-8B-Instruct, deepseek-llm-7b-chat
            
        --attacker_name (必需): 攻击者/后门攻击方法的名称
            例如：BadNets, VPI, Sleeper, MTBA, CTBA
            
        --dataset (必需): 数据集名称
            例如：SST-2, AGNews, HateSpeech, DAN
            
        --task (必需): 任务类型
            例如：classification, jailbreak
            
        --defender_name (必需): 防御者/后门移除方法的名称
            例如：None（无防御）, FineTune, Prune-wanda, BadWanda
            
        --tasks (可选): 逗号分隔的评估任务列表，默认为 "boolq,rte,hellaswag,winogrande,arc_challenge,openbookqa"
            支持的任务包括：
            - boolq: BoolQ 布尔问答任务
            - rte: RTE 文本蕴含任务
            - hellaswag: HellaSwag 常识推理任务
            - winogrande: WinoGrande 常识推理任务
            - arc_challenge: ARC 挑战集
            - openbookqa: OpenBookQA 开放书籍问答
            
        --batch_size (可选): 批处理大小，默认为 "512"
            可以设置为 "auto" 让框架自动选择最优值
            
        --device (可选): 计算设备，默认为 "cuda"
            可选值：cuda, cpu, cuda:0, cuda:1 等
            
        --output_path (可选): 结果输出文件路径，默认为 "results.json"
            
        --num_fewshot (可选): Few-shot 示例数量，默认为 0（零样本评估）
            设置为正整数可进行 few-shot 评估
            
        --limit (可选): 每个任务限制评估的样本数量，用于调试
            默认为 None（评估所有样本）
            
        --use_accelerate (可选): 是否使用 HuggingFace Accelerate 库
            使用此选项时模型已通过 device_map='auto' 加载
    """
    # 创建参数解析器，description 用于描述脚本功能
    parser = argparse.ArgumentParser(description="Evaluate Eraser LoRA Model with lm-eval")
    
    # 添加必需的命令行参数
    parser.add_argument("--model_name", type=str, required=True, 
                        help="Path to the base model")
    parser.add_argument("--attacker_name", type=str, required=True, 
                        help="Path to the attacker model")
    parser.add_argument("--dataset", type=str, required=True, 
                        help="Dataset name")
    parser.add_argument("--task", type=str, required=True, 
                        help="Task type (classification, jailbreak, etc.)")
    parser.add_argument("--defender_name", type=str, required=True, 
                        help="Defender method name")
    
    # 添加可选的命令行参数
    parser.add_argument("--tasks", type=str, 
                        default="boolq,rte,hellaswag,winogrande,arc_challenge,openbookqa", 
                        help="Comma separated list of tasks")
    parser.add_argument("--batch_size", type=str, default="512", 
                        help="Batch size (can be 'auto')")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="Device to use (e.g. cuda, cpu)")
    parser.add_argument("--output_path", type=str, default="results.json", 
                        help="Path to save results")
    parser.add_argument("--num_fewshot", type=int, default=0, 
                        help="Number of few-shot examples")
    parser.add_argument("--limit", type=float, default=None, 
                        help="Limit number of examples per task (for debugging)")
    parser.add_argument("--use_accelerate", action="store_true", 
                        help="Use accelerate (note: model is already loaded with device_map='auto')")
    
    # 解析命令行参数并返回
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
    """
    执行零样本评估

    本函数是核心评估函数，使用 lm-eval 框架对模型进行零样本评估。
    支持多个基准任务，并返回详细的评估结果。

    参数：
        model (torch.nn.Module): 已加载的 PyTorch 模型
            应该是因果语言模型 (Causal LM)，例如 GPT、LLaMA 等

        tokenizer (transformers.PreTrainedTokenizer): HuggingFace 分词器
            用于将文本转换为模型输入的 token ID

        task_list (list): 要评估的任务名称列表
            例如：['boolq', 'rte', 'hellaswag']

        num_fewshot (int, optional): Few-shot 示例数量，默认为 0
            - 0 表示零样本评估 (zero-shot)
            - 正整数表示使用对应数量的示例作为上下文演示

        batch_size (int or str, optional): 批处理大小，默认为 1
            可以是整数（如 1, 4, 8）或字符串 "auto"（自动选择）

        limit (float or None, optional): 限制每个任务的样本数量，默认为 None
            用于快速调试，例如 0.1 表示只评估 10% 的样本

        use_accelerate (bool, optional): 是否使用 Accelerate，默认为 False
            为 True 时忽略 device 参数，让 Accelerate 管理设备分配

        device (str, optional): 计算设备，默认为 "cuda:0"
            例如：'cuda', 'cuda:0', 'cpu'

    返回值：
        dict: 评估结果字典，包含以下键值：
            - 'results': 每个任务的评估指标结果
            - 'configs': 任务配置信息
            - 'num_fewshot': 使用的 few-shot 数量
            - 'batch_size': 实际使用的批处理大小
            - 'latencies': 推理延迟信息
            - 'metadata': 评估元数据

    工作流程：
        1. 使用 pattern_match 函数根据任务列表过滤 lm-eval 支持的任务
        2. 创建 TaskManager 管理任务
        3. 将模型封装为 HFLM 接口
        4. 调用 evaluator.simple_evaluate 进行评估
        5. 移除样本级结果以减小内存占用
    """

    # 抑制 lm-eval 的详细日志输出，但保留进度条
    import logging

    # 设置 lm-eval 相关的日志级别为 ERROR，只显示错误
    # 这样可以保留 tqdm 进度条，但隐藏冗长的日志信息
    logging.getLogger("lm_eval").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("datasets").setLevel(logging.ERROR)

    def pattern_match(patterns, source_list):
        """
        任务名称模式匹配
        
        将用户提供的任务模式（如 "boolq", "rte*"）与 lm-eval 支持的
        任务名称进行匹配，返回实际的任务名称列表。
        
        参数：
            patterns (list): 任务名称模式列表，支持通配符
                例如：['boolq', 'rte', 'hellaswag'] 或 ['*']
            source_list (list): lm-eval 支持的所有任务名称列表
            
        返回值：
            list: 匹配到的任务名称列表
            
        示例：
            pattern_match(['boolq', 'rte*'], ['boolq', 'rte', 'rte2', 'hellaswag'])
            # 返回: ['boolq', 'rte', 'rte2']
        """
        task_names = set()  # 使用集合去重
        for pattern in patterns:  # 遍历每个模式
            # fnmatch.filter 支持通配符匹配，如 'rte*' 匹配 'rte', 'rte2'
            for matching in fnmatch.filter(source_list, pattern):
                task_names.add(matching)
        return list(task_names)

    # 创建 TaskManager，用于管理 lm-eval 支持的所有评估任务
    # TaskManager 会自动扫描并加载所有注册的任务
    task_manager = TaskManager()
    
    # 使用模式匹配获取要评估的任务列表
    task_names = pattern_match(task_list, task_manager.all_tasks)

    # 打印将要评估的任务列表
    _log(f"正在评估任务: {', '.join(task_names)}")

    # 将模型封装为 HFLM (HuggingFace Language Model)
    # HFLM 是 lm-eval 提供的模型接口适配器，负责：
    # 1. 将模型包装为 lm-eval 需要的 LanguageModel 接口
    # 2. 处理批量推理和日志概率计算
    # 3. 管理模型的生成和评估模式
    _log("正在封装模型...")
    lm = HFLM(
        pretrained=model,  # 传入 PyTorch 模型
        tokenizer=tokenizer,  # 传入分词器
        batch_size=batch_size,  # 批处理大小
    )

    # 调用 lm-eval 的简单评估接口
    # 这是评估的核心函数，会在所有指定任务上运行评估
    _log("开始评估...")

    # 捕获 lm-eval 的标准输出，只保留进度条
    results = evaluator.simple_evaluate(
        model=lm,  # 封装的模型接口
        tasks=task_names,  # 要评估的任务列表
        num_fewshot=num_fewshot,  # Few-shot 数量
        batch_size=batch_size,  # 批处理大小
        limit=limit,  # 样本数量限制
        # 设备选择：如果不使用 accelerate，则使用指定的 device
        device=device if not use_accelerate else None,
        task_manager=task_manager  # 任务管理器
    )

    _log("评估完成")

    # 移除样本级结果以减小内存占用
    # 'samples' 包含每个输入样本的详细结果，可能非常大
    # 'sample_results' 是另一个可能占用大量内存的字段
    results.pop("samples", None)
    results.pop("sample_results", None)

    return results


def main():
    """
    主函数 - 评估流程的入口点
    
    负责协调整个评估流程：
    1. 解析命令行参数
    2. 加载模型（基础模型 + 攻击者模型 + 防御者模型）
    3. 根据防御者类型应用相应的防御方法
    4. 执行零样本评估
    5. 保存评估结果到 JSON 文件
    
    防御者类型处理：
    - 'None': 不应用任何防御
    - 包含 'FineTune': 使用 LoRA 微调防御，加载 LoRA 权重
    - 包含 'Prune': 使用剪枝防御，应用剪枝掩码
    """
    # 步骤1：解析命令行参数
    args = get_args()
    
    # 将任务字符串分割为列表
    # 例如："boolq,rte,hellaswag" -> ["boolq", "rte", "hellaswag"]
    task_list = args.tasks.split(",")

    # 步骤2：加载基础模型和分词器
    # load_model 函数负责：
    # 1. 从指定路径加载预训练模型
    # 2. 加载对应的分词器
    # 3. 可能的攻击者模型集成
    model, tokenizer = load_model(
        args.model_name, 
        args.attacker_name, 
        args.data_name,  # 注意：这里应该是 args.dataset，但代码中使用了 args.data_name
        args.task, 
        args.device, 
        model_id=None, 
        lora_path=None
    )
    
    # 步骤3：根据防御者类型应用防御方法
    
    # 情况1：无防御（None）
    if args.defender_name == 'None':
        pass  # 不做任何处理
    
    # 情况2：FineTune 防御（LoRA 微调）
    # 防御方法：在后门数据上进行微调，以消除后门影响
    elif 'FineTune' in args.defender_name:
        # 构建防御模型路径
        # 路径格式：/home/server/SSD/attacked/{attacker_name}/{model_name}-{task}-{dataset}-{defender_name}
        max_model_id = f"/home/server/SSD/attacked/{args.attacker_name}/{args.model_name}-{args.task}-{args.data_name}-{args.defender_name}"
        print(f"Loading FineTune model from: {max_model_id}")
        # 使用 PEFT 库加载 LoRA 权重
        # PeftModel 会在基础模型上叠加 LoRA 适配器
        model = PeftModel.from_pretrained(model, model_id=max_model_id)
    
    # 情况3：Prune 防御（权重剪枝）
    # 防御方法：剪枝掉与后门相关的神经元权重
    elif 'Prune' in args.defender_name:
        # 导入剪枝防御相关的工具函数
        from defenders.defender_prune import get_transformer_layers, find_layers
        
        # 构建防御模型路径
        delt_model_id = f'/home/server/SSD/attacked/{args.attacker_name}/{args.model_name}-{args.task}-{args.data_name}-{args.defender_name}'
        print(f"Loading Prune model from: {delt_model_id}")
        
        # 加载剪枝掩码
        # 掩码文件 .pt 格式，保存了需要置零的权重位置
        mask_file = os.path.join(delt_model_id, 'prune_mask.pt')
        mask_dict = torch.load(mask_file, map_location='cpu')
        
        # 获取模型的所有 Transformer 层
        layers = get_transformer_layers(model)
        
        # 遍历每一层，应用剪枝掩码
        for i, layer in enumerate(layers):
            # 找到该层中所有可剪枝的模块（全连接层、注意力层等）
            for name, module in find_layers(layer).items():
                # 构建掩码键名，格式：'layer.{层索引}.{模块名称}'
                key = f'layer.{i}.{name}'
                if key in mask_dict:
                    # 将掩码对应的权重位置置零
                    # mask_dict[key] 是一个布尔张量，True 表示需要保留，False 表示需要剪枝
                    module.weight.data[mask_dict[key].to(module.weight.device)] = 0

    # 步骤4：执行零样本评估
    results = eval_zero_shot(
        model=model,
        tokenizer=tokenizer,
        task_list=task_list,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        limit=args.limit,
        use_accelerate=args.use_accelerate,
        device=args.device,
    )
    
    # 步骤5：处理结果中的不可序列化对象
    # 某些 PyTorch 对象（如 torch.dtype）无法直接序列化为 JSON
    # 需要转换为可序列化的格式
    
    def make_serializable(obj):
        """
        将对象转换为可序列化的格式
        
        递归处理字典、列表和其他对象，将 PyTorch 特定类型
        转换为 JSON 可序列化的字符串表示。
        
        参数：
            obj: 任意类型的对象
            
        返回值：
            可序列化的对象（str, dict, list, 或原始对象）
        """
        # 处理 PyTorch 数据类型
        if isinstance(obj, torch.dtype):
            return str(obj)  # 例如：torch.float32 -> "torch.float32"
        
        # 递归处理字典
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        
        # 递归处理列表
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        
        # 其他对象直接返回
        return obj

    # 应用序列化转换
    results = make_serializable(results)
    
    # 打印结果到控制台（格式化输出）
    print(json.dumps(results, indent=2))
    
    # 步骤6：保存结果到文件
    if args.output_path:
        with open(args.output_path, "w", encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output_path}")


def get_utility(model, tokenizer, data_name, model_name, attacker_name, task, defender_name, device, load=True, utility=False, exp_id=None, task_list=None):
    """
    获取模型的效用值（Utility）
    
    本函数用于评估模型在特定配置下的效用（utility），即模型在
    正常任务上的性能表现。通常与后门攻击评估一起使用，以衡量
    防御方法在移除后门的同时对模型正常功能的影响。
    
    参数：
        model (torch.nn.Module): PyTorch 模型
        
        tokenizer (transformers.PreTrainedTokenizer): 分词器
        
        data_name (str): 数据集名称
            例如：'SST-2', 'AGNews', 'HateSpeech'
            
        model_name (str): 模型名称
            例如：'Meta-Llama-3-8B-Instruct', 'deepseek-llm-7b-chat'
            
        attacker_name (str): 攻击者名称
            例如：'Clean'（干净模型）, 'BadNets', 'VPI', 'Sleeper'
            
        task (str): 任务类型
            例如：'classification', 'jailbreak'
            
        defender_name (str): 防御者名称
            例如：'None', 'FineTune', 'Prune-wanda'
            
        device (str): 计算设备
            例如：'cuda', 'cuda:0', 'cpu'
            
        load (bool, optional): 是否加载已存在的结果，默认为 True
            - True：如果结果文件存在则加载，否则重新评估
            - False：强制重新评估
    
    返回值：
        float: 模型在所有任务上的平均效用值（0-1 之间）
            值越高表示模型正常功能越好
    
    结果文件路径规则：
        - Clean 模型：{model_name}-{task}-{data_name}.json
        - 有攻击者无防御：{model_name}-{task}-{data_name}-{attacker_name}.json
        - 有攻击者有防御：{model_name}-{task}-{data_name}-{attacker_name}-{defender_name}.json
    
    工作流程：
        1. 构建结果文件路径
        2. 检查结果文件是否存在
        3. 如果不存在或 load=False，则执行评估并保存结果
        4. 读取结果文件，提取评估指标
        5. 计算平均效用值并返回
    """
    global _eval_log_exp_id
    if exp_id:
        _eval_log_exp_id = exp_id

    # 定义要评估的任务列表（这里固定为 mnli）
    # 可以扩展为：'mnli,toxigen,mmlu,truthfulqa,cola,mrpc,qnli,rte,gsm8k'
    if task_list is None:
        task_list = 'mnli,toxigen,mmlu,truthfulqa,cola,mrpc,qnli,rte,gsm8k'.split(",")
    elif isinstance(task_list, str):
        task_list = task_list.split(",")
    
    # 根据模型、攻击者、防御者配置构建结果文件路径
    # 结果保存在 /home/server/Documents/gh/BadSD/lm_eval_results/ 目录
    if attacker_name == 'Clean':
        if utility:
            output_path = f'/home/server/science/gh/BadSD/lm_eval_results/{model_name}-{task}-{data_name}-utility.json'
        else:
            # 干净模型的结果路径
            output_path = f'/home/server/science/gh/BadSD/lm_eval_results/{model_name}-{task}-{data_name}.json'
    elif defender_name == 'None':
        if utility:
            output_path = f'/home/server/science/gh/BadSD/lm_eval_results/{model_name}-{task}-{data_name}-{attacker_name}-utility.json'
        else:
            # 有攻击者但无防御的结果路径
            output_path = f'/home/server/science/gh/BadSD/lm_eval_results/{model_name}-{task}-{data_name}-{attacker_name}.json'
    else:
        if utility:
            output_path = f'/home/server/science/gh/BadSD/lm_eval_results/{model_name}-{task}-{data_name}-{attacker_name}-{defender_name}-utility.json'
            _log(output_path)
        else:
            # 有攻击者且有防御的结果路径
            output_path = f'/home/server/science/gh/BadSD/lm_eval_results/{model_name}-{task}-{data_name}-{attacker_name}-{defender_name}.json'

    _log(output_path)
    
    # 检查是否需要重新评估
    # load=True 且文件存在：直接读取现有结果
    # load=False 或文件不存在：执行评估

    if not load or not os.path.exists(output_path):
        # 执行零样本评估
        results = eval_zero_shot(
            model=model,
            tokenizer=tokenizer,
            task_list=task_list,
            num_fewshot=0,  # 零样本评估
            limit=None,  # 评估所有样本
            use_accelerate=False,
            device=device,
        )

        # 定义内部函数：处理不可序列化对象
        def make_serializable(obj):
            """
            将对象转换为可序列化格式（与 main 函数中的实现相同）
            """
            # 处理 PyTorch 数据类型
            if isinstance(obj, torch.dtype):
                return str(obj)
            
            # 递归处理字典
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            
            # 递归处理列表
            if isinstance(obj, list):
                return [make_serializable(v) for v in obj]
            
            return obj

        # 转换结果为可序列化格式
        results = make_serializable(results)

        # 保存完整结果到 JSON 文件
        with open(output_path, "w", encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        _log(f"完整结果已保存至: {output_path}")

    # 读取结果并计算平均效用值
    # read 函数负责解析 JSON 文件并提取指定任务的指标
    # 返回 (平均效用值, CSV 行字符串)
    mean_utility, txt = read(model_name, task, attacker_name, data_name, defender_name, utility=utility)

    # 将 read 拼好的 CSV 行（时间戳 + 各任务指标）写入日志
    _log("\n[Utility 评估结果]")
    _log(f"  任务顺序: {', '.join(metrics_to_extract.keys())}")
    _log(f"  {txt}")

    return mean_utility


# ============================================================
# 程序入口点
# ============================================================
if __name__ == "__main__":
    """
    Python 程序的标准入口点
    
    当脚本作为主程序运行时（而非作为模块导入），会执行 main() 函数。
    这确保了代码在被其他模块导入时不会自动执行。
    """
    main()