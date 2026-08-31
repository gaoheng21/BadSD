# BadWindtunnel 后门防御算法实现
# 作者：基于 LossInDefender 思想实现
# 功能：通过置信度变化检测后门样本，使用加权微调消除后门

# 导入必要的库
from utils.load_info import load_model, load_label_rate  # 加载模型和标签信息的工具函数
import os, torch  # 操作系统和PyTorch基础库
import datetime
from datasets import Dataset  # HuggingFace数据集处理库
from peft import LoraConfig, get_peft_model, TaskType  # LoRA微调库
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq  # Transformer训练相关
import torch.nn as nn  # PyTorch神经网络模块
from mydataset.poisoner import poison_data  # 数据投毒函数
from mydataset.process_data import process_to_json  # 数据处理函数
import numpy as np  # 数值计算库
from sklearn.mixture import GaussianMixture  # 高斯混合模型
from sklearn.preprocessing import MinMaxScaler  # 数据归一化
from sklearn.metrics import davies_bouldin_score  # DBI聚类评估指标
from evaluators.evaluate_eraser_lora import get_utility  # 模型效用评估
from generators.classification_attack import get_ASR_classification, get_CACC_classification  # 分类攻击评估
from generators.jailbreak_attack import get_ASR_jailbreak, get_CACC_jailbreak  # 越狱攻击评估
import copy  # 深拷贝库
from tqdm import tqdm  # 进度条库
import random  # 随机数生成库
from collections import defaultdict
import matplotlib.pyplot as plt  # 画图库
from evaluators.evaluate_eraser_lora import get_utility
from defenders.train_model import finetune_alpaca
import seaborn as sns  # seaborn 画图库
from peft import PeftModel  # PEFT库的模型类，用于加载LoRA等参数高效微调的模型
# 设置设备：优先使用GPU，否则使用CPU
from transformers import (
    AutoTokenizer,  # 自动加载对应的分词器
    AutoModelForCausalLM,  # 自动加载因果语言模型（自回归模型，如GPT、LLaMA等）
    BitsAndBytesConfig,  # 量化配置类，用于4-bit/8-bit量化加载
    get_linear_schedule_with_warmup  # 获取带预热（warmup）的线性学习率调度器
)
# from defenders.defender_fisher_find_data import fisher_find_data

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# 结果保存路径
sum_path = './result.csv'


def do_badwindtunnel(data_name, model_name, attacker_name, defender_name, task='classification', device=DEVICE,
                     silence=1, load=True):
    """
    BadWindtunnel 主入口函数

    参数说明：
    - data_name: 数据集名称 (SST-2, AGNews, HateSpeech等)
    - model_name: 模型名称 (Meta-Llama-3-8B-Instruct等)
    - attacker_name: 攻击方法 (BadNets, VPI, Sleeper等)
    - defender_name: 防御方法名称 (BadWindtunnel)
    - task: 任务类型 (classification/jailbreak)
    - device: 计算设备
    - silence: 是否静默输出 (1=静默, 0=输出详细结果)
    - load: 是否加载已保存的模型 (True=加载, False=重新训练)

    返回值：
    - CACC: 干净准确率 (Clean Accuracy)
    - ASR: 攻击成功率 (Attack Success Rate)
    """
    # 调试阶段是否加载已有badwindtunnel微调后的模型
    load = False
    # 设置微调模型保存路径
    print("load = ", load)
    #finetune就是说这是一个防御并进行了utility微调的模型
    #utility参数代表的是是否进行了utility微调
    OUT_DIR_F = f"/science/gh/BadSD/finetune/{attacker_name}/{defender_name}-{model_name}-{task}-{data_name}"


    # 检查是否已有保存的模型，如果没有则重新训练
    if not load or not os.path.exists(OUT_DIR_F):
        if not os.path.exists(OUT_DIR_F):
            os.makedirs(OUT_DIR_F, exist_ok=True)
            print(f"目录已创建: {OUT_DIR_F}")
        else:
            print(f"目录已存在: {OUT_DIR_F}")
        print("load = False，将训练微调模型")
        # 执行 BadWindtunnel 防御训练模型  BadWindtunnel 防御训练
            # 加载基础模型
        model, tokenizer = load_model(model_name, "Clean", data_name, task, device)
            # 训练BadWindtunnel模型
        model, tokenizer = BadWindtunnel(model, tokenizer,model_name, data_name, attacker_name, defender_name, task, device)

        # 保存微调训练后的模型和分词器,这个OUT_DIR_F的lora权重包含的信息是：被attacker_name攻击并且执行了BadWindtunnel防御
        model.save_pretrained(OUT_DIR_F)
        tokenizer.save_pretrained(OUT_DIR_F)
        print(f"BadWindtunnel模型已经保存到 {OUT_DIR_F}")
    else:
        # 加载已有模型
        model, tokenizer = load_model(model_name, "Clean", data_name, task, device, lora_path=OUT_DIR_F)
        print("已加载已保存的BadWindtunnel模型: ", OUT_DIR_F)

    # 获取攻击率和目标标签
    rate, target_label = load_label_rate(task, data_name)
    # 加载测试数据
    data = process_to_json(data_name, 'test')
    # 生成投毒测试数据
    p_data = poison_data(data_name, data, attacker_name, target_label, 'test', rate, load=True, task=task)

    # 初始化评估指标
    CACC, ASR, utility = 0, 0, 0
    print("------------------------------------------计算CACC,ASR-------------------------------------------")
    # 根据任务类型进行评估
    if task == 'classification':

        # 计算攻击成功率 (ASR)
        ASR = get_ASR_classification(p_data, model, tokenizer, defender_name=defender_name, model_name=model_name,
                                     attacker_name=attacker_name, data_name=data_name, load=load, utility=False)

        # 计算干净准确率 (CACC)
        CACC = get_CACC_classification(data, model, tokenizer, defender_name, model_name, attacker_name,
                                       data_name=data_name, load=load, utility=False)

    elif task == 'jailbreak':
        # 越狱攻击评估
        CACC = get_CACC_jailbreak(data=data, model=model, tokenizer=tokenizer, defender_name=defender_name,
                                  model_name=model_name, attacker_name=attacker_name, data_name=data_name, load=load)
        ASR = get_ASR_jailbreak(p_data=p_data, model=model, tokenizer=tokenizer, defender_name=defender_name,
                                model_name=model_name, attacker_name=attacker_name, data_name=data_name, load=load)
    print("------------------------------------------计算utility-------------------------------------------")
    utility = get_utility(model, tokenizer, data_name, model_name, attacker_name, task, defender_name, device,load=load,utility=False)

    print(f"utility = {utility}, ASR = {ASR}, CACC = {CACC}")
    # print("-----------------------------------------恢复utility，进行lora微调并保存模型-----------------------------------------")
    # #utility参数代表的是是否进行了utility微调
    # #这个OUT_DIR_F_utility的lora权重包含的信息是：使用finetune_alpaca函数进行的lora微调
    #
    # OUT_DIR_F_utility = f"/science/gh/BadSD/finetune/utility/{attacker_name}/{defender_name}-{model_name}-{task}-{data_name}"
    #
    # if not load or not os.path.exists(OUT_DIR_F_utility):
    #     if not os.path.exists(OUT_DIR_F_utility):
    #         os.makedirs(OUT_DIR_F_utility, exist_ok=True)
    #         print(f"恢复utility，进行lora微调的模型目录已创建: {OUT_DIR_F_utility}")
    #     else:
    #         print(f"恢复utility，进行lora微调的模型目录已存在: {OUT_DIR_F_utility}")
    #
    #     cleaned_model, cleaned_tokenizer = finetune_alpaca(model, tokenizer,2,2e-5)
    #     # 保存微调训练后的模型和分词器
    #
    #     cleaned_model.save_pretrained(OUT_DIR_F_utility)
    #     cleaned_tokenizer.save_pretrained(OUT_DIR_F_utility)
    #     print(f"恢复utility，进行lora微调后的模型已经保存到 {OUT_DIR_F_utility}")
    # else:
    #     # 加载已有模型
    #     #cleaned_model, cleaned_tokenizer = load_model(model_name, attacker_name, data_name, task, device, lora_path=OUT_DIR_F_utility)
    #
    #     #合并OUT_DIR_F_utility的lora权重到model和tokenizer,得到cleaned_model和cleaned_tokenizer
    #     # 加载LoRA模型
    #     print(f'正在加载LoRA适配器: {OUT_DIR_F_utility}')
    #     cleaned_model = PeftModel.from_pretrained(model, OUT_DIR_F_utility)
    #     # 合并LoRA权重到badwindtunnel模型
    #     cleaned_model = cleaned_model.merge_and_unload()
    #     print('LoRA权重合并完成')
    #     cleaned_tokenizer = AutoTokenizer.from_pretrained(OUT_DIR_F_utility, trust_remote_code=True)
    #     print("已加载cleaned_tokenizer")
    #
    #
    # print("------------------------------------------再次计算CACC,ASR-------------------------------------------")
    # CACC1, ASR1, utility1 = 0, 0, 0
    # # 根据任务类型进行评估
    # if task == 'classification':
    #
    #     # 计算攻击成功率 (ASR)
    #     ASR1 = get_ASR_classification(p_data, cleaned_model, tokenizer, defender_name=defender_name,
    #                                   model_name=model_name,
    #                                   attacker_name=attacker_name, data_name=data_name, load=load, utility=True)
    #
    #     # 计算干净准确率 (CACC)
    #     CACC1 = get_CACC_classification(data, cleaned_model, tokenizer, defender_name, model_name, attacker_name,
    #                                     data_name=data_name, load=load, utility=True)
    #
    # elif task == 'jailbreak':
    #     # 越狱攻击评估
    #     CACC1 = get_CACC_jailbreak(data=data, model=cleaned_model, tokenizer=tokenizer, defender_name=defender_name,
    #                                model_name=model_name, attacker_name=attacker_name, data_name=data_name, load=load)
    #     ASR1 = get_ASR_jailbreak(p_data=p_data, model=cleaned_model, tokenizer=tokenizer, defender_name=defender_name,
    #
    #                              model_name=model_name, attacker_name=attacker_name, data_name=data_name, load=load)
    # # 格式化结果
    #
    # print("------------------------------------------再次计算utility-------------------------------------------")
    # #utility参数代表的是是否进行了utility微调
    # utility1 = get_utility(cleaned_model, cleaned_tokenizer, data_name, model_name, attacker_name, task, defender_name, device, load=load,utility=True)


    # print(f"utility1 = {utility1}, ASR1 = {ASR1}, CACC1 = {CACC1}")
    note="不进行微调时候的效果"
    txt = f'{task},{data_name},{model_name},{attacker_name},{defender_name},{CACC},{ASR},{utility},{datetime.datetime.now()},{note}'

    # 根据silence参数决定是否写入文件
    if silence == 0:
        f = open(sum_path, 'a')
        print(txt, file=f)
        f.close()

    # 打印结果
    print(txt)
    return CACC, ASR, utility


def BadWindtunnel(model, tokenizer,model_name, data_name, attacker_name, defender_name, task='classification', device=DEVICE):
    """
    BadWindtunnel 核心防御算法

    主要步骤：
    1. 加载原始干净模型（这是偏向对未知后门数据方面的后门防御）
    2. 准备原始后门数据
    3. 应用LoRA
    4. 标签平衡
    5. 数据预处理
    6. 迭代寻找最佳权重
    7. 最终加权微调

    返回值：
    - model: 防御后的模型
    - tokenizer: 对应的分词器
    """
    # 1. 加载原始干净模型和分词器
    # 获取攻击率和目标标签
    rate, target_label = load_label_rate(task, data_name)
    print(f"实际攻击率: {rate}, 实际目标标签: {target_label}")

    # 2. 准备训练数据
    clean_data = process_to_json(data_name, 'train', load=True, write=True)
    # 准备原始后门数据
    raw_data = poison_data(data_name, clean_data, attacker_name, target_label, 'train', rate, load=True, task=task)

    print(f"raw_data共: {len(raw_data)} 条样本")

    # 3. 冻结原始模型参数，只训练LoRA参数
    for param in model.parameters():
        param.requires_grad = False


    # 4. 配置LoRA参数
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # 因果语言模型任务
        inference_mode=False,  # 训练模式
        r=16,  # LoRA秩，控制参数数量
        lora_alpha=32,  # 缩放因子，通常设为2*r
        lora_dropout=0.05,  # Dropout防止过拟合
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        # 目标模块：Transformer的所有注意力相关矩阵
    )

    # 应用LoRA到模型
    model = get_peft_model(model, peft_config)

    # 打印可训练参数信息
    model.print_trainable_parameters()


    def tokenize_function(examples):
        """
        将文本数据转换为模型输入格式

        参数：
        - examples: 包含instruction、input、output的字典列表

        返回值：
        - input_ids: 输入token ID序列
        - labels: 标签序列（prompt部分为-100，不计算损失）
        - attention_mask: 注意力掩码
        """
        input_ids_list = []
        labels_list = []
        attention_mask_list = []

        MAX_LEN = 512  # 最大序列长度

        # 批量处理所有样本
        for instruction, input_text, output in zip(examples["instruction"], examples["input"], examples["output"]):
            # 构建 Prompt 部分
            if input_text.strip():
                prompt = f"Instruction: {instruction}\nInput: {input_text}\nAnswer: "
            else:
                prompt = f"Instruction: {instruction}\nAnswer: "

            # 目标输出部分 (加上 EOS token 作为结束标记)
            target = output + tokenizer.eos_token

            # 分别 tokenize：prompt和target分开处理
            # 注意：确保只给prompt添加特殊token
            prompt_tokens = tokenizer(prompt, add_special_tokens=True)["input_ids"]
            target_tokens = tokenizer(target, add_special_tokens=False)["input_ids"]

            # 拼接完整的输入序列
            input_ids = prompt_tokens + target_tokens

            # ★ 核心操作：将 Prompt 部分的 label 替换为 -100
            # 这样模型只学习预测答案部分，不学习prompt部分
            labels = [-100] * len(prompt_tokens) + target_tokens
            attention_mask = [1] * len(input_ids)  # 全部位置都参与注意力计算

            # 截断处理：超过最大长度则截断
            if len(input_ids) > MAX_LEN:
                input_ids = input_ids[:MAX_LEN]
                labels = labels[:MAX_LEN]
                attention_mask = attention_mask[:MAX_LEN]

            input_ids_list.append(input_ids)
            labels_list.append(labels)
            attention_mask_list.append(attention_mask)

        return {
            "input_ids": input_ids_list,
            "labels": labels_list,
            "attention_mask": attention_mask_list
        }

    # 5. 标签平衡：解决样本不平衡问题，确保各类别权重一致

    raw_data = balance_label(raw_data)
    # 将原始数据转换为HuggingFace Dataset格式
    raw_dataset = Dataset.from_list(raw_data)
    # 应用tokenization函数，批量处理所有数据
    raw_dataset = raw_dataset.map(tokenize_function, batched=True, remove_columns=raw_dataset.column_names)


    print("标签已经平衡")
    # ================= 核心流程开始 =================
    # 首先创建原始噪声数据（暂时为0躁，迭代开始后加噪）
    noise_data = copy.deepcopy(raw_data)

    # 备份训练前的LoRA权重，用于后续恢复
    initial_lora_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items() if "lora" in k})

    # 创建保存目录
    save_dir = "/science/gh/BadSD/gmm_test"
    os.makedirs(save_dir, exist_ok=True)

    count = 0  # 迭代计数器
    dbi = 1000  # DBI初始值（越大越差）
    rate_dc = 0  # 目标标签占比
    best = {}  # 存储最佳轮次的结果
    noise_rate = 30# 初始噪声率（30%）

    print("开始迭代...")

    print("1. 分析初始 Confidence...,执行第一次推理，获取扰动前的c_before置信度作为基线")

    #如果存在c_before.pt文件，就读取
    if os.path.exists(os.path.join(save_dir, f"{attacker_name}-{defender_name}-{model_name}-{task}-{data_name}-c_before.pt")):
        c_before = torch.load(os.path.join(save_dir, f"{attacker_name}-{defender_name}-{model_name}-{task}-{data_name}-c_before.pt"))
        print(f"已经从文件读取c_before：{save_dir}/{attacker_name}_{defender_name}_{model_name}_{task}_{data_name}-c_before.pt")
    else:
        # 执行一次推理，获取一个epoch前的置信度作为基线
        print(f"没有{attacker_name}-{defender_name}-{model_name}-{task}-{data_name}-c_before.pt文件，执行第一次推理")
        c_before = get_inference_confidence(model, tokenizer, noise_data)
        # 保存c_before到文件
        torch.save(c_before, os.path.join(save_dir, f"{attacker_name}-{defender_name}-{model_name}-{task}-{data_name}-c_before.pt"))
        print(f"已经保存c_before到目录：{save_dir}/{attacker_name}_{defender_name}_{model_name}_{task}_{data_name}-c_before.pt")

    while (dbi > 0.8 or rate_dc < 0.65) and count < 3:
        # noise_rate += 5  # 每轮增加噪声率，增强区分度
        # if noise_rate > 30:
        #     noise_rate = 30
        # 对输入文本添加噪声（字母交换）
        noise_data = add_data_noise(noise_data, noise_rate)
        print(f"----------------当前COUNT: {count}------当前NOISE_RATE: {noise_rate}---")


        print("2. 执行 1 个 Epoch 试探性微调...")
        # 恢复初始LoRA权重，重新开始
        model.load_state_dict(initial_lora_state, strict=False)
        # 训练 1 个 Epoch，不使用自定义权重
        noise_dataset = Dataset.from_list(noise_data)
        # 应用tokenization函数，批量处理所有数据
        noise_dataset = noise_dataset.map(tokenize_function, batched=True, remove_columns=noise_dataset.column_names)
        model = finetune_one_epoch(
            model, tokenizer, noise_dataset, epoch=1, use_custom_trainer=False
        )

        print("3. 分析训练后 Confidence...,执行第二次推理，获取扰动后的置信度")
        # 训练一个Epoch后的，执行第二次推理，获取扰动后的置信度，如果已经存在，就读取
        c_after = get_inference_confidence(model, tokenizer, noise_data)
        # # ========== 保存本轮数据 ==========
        # # 文件名格式: noise_rate_{noise_rate}_c_before.pt
        # torch.save(c_before, os.path.join(save_dir, f"noise_rate_{noise_rate}_c_before.pt"))
        # print(f"已经保存c_before到目录：/science/gh/BadSD/gmm_test/noise_rate_{noise_rate}_c_before.pt")
        # torch.save(c_after, os.path.join(save_dir, f"noise_rate_{noise_rate}_c_after.pt"))
        # print(f"已经保存c_after到目录：/science/gh/BadSD/gmm_test/noise_rate_{noise_rate}_c_after.pt")
        # with open(os.path.join(save_dir, f"noise_rate_{noise_rate}_noise_data.pkl"), 'wb') as f:
        #     pickle.dump(noise_data, f)
        # print(f"已经保存noise_data到目录：/science/gh/BadSD/gmm_test/noise_rate_{noise_rate}_noise_data.pkl")
        # # =================================

        print("4. 计算变化权重..")
        # 核心算法：根据c_before和c_after置信度变化计算样本权重
        weights, current_dbi, current_rate_dc = calculate_defense_weights(
            c_before, c_after, noise_data, threshold_percent=0.1, dbi_rate=1
        )

        # 维护最佳记录：DBI更小且目标标签占比current_rate_dc足够，这个阈值要调整
        if count == 0 or (best['dbi'] > current_dbi and current_rate_dc > 0.6):
            print(f"更新最佳记录为DBI={current_dbi}, rate_dc={current_rate_dc}, noise_rate={noise_rate}")
            best['dbi'] = current_dbi
            best['weights'] = weights
            best['rate_dc'] = current_rate_dc
            best['noise_rate'] = noise_rate

        # 更新当前状态
        dbi = current_dbi
        rate_dc = current_rate_dc
        count += 1


    print(f"\n最佳寻找结果! DBI = {best['dbi']}, rate_dc = {best['rate_dc']},noise_rate={best['noise_rate']}")
    final_weights = best['weights']

    # ================= 真正的加权微调阶段 =================

    # 将计算好的权重添加到原始平衡好的数据集raw_dataset
    dataset_weighted = raw_dataset.add_column("sample_weights", final_weights)

    # 再次恢复初始模型
    print("\n恢复初始模型准备最终微调...")
    model.load_state_dict(initial_lora_state, strict=False)

    # 使用自定义Trainer进行加权微调
    model = finetune_one_epoch(
        model, tokenizer, dataset_weighted, epoch=3, use_custom_trainer=True
    )
    return model, tokenizer


def add_data_noise(data, n):
    """
    在输入文本中添加噪声（字母交换）

    参数：
    - data: 原始数据列表
    - n: 噪声率（百分比）

    返回值：
    - noise_data: 添加噪声后的数据
    """
    noise_data = []
    for ii, item in enumerate(tqdm(data)):
        noise_data.append({
            'instruction': item['instruction'],  # 保持指令不变
            'input': shuffle_adjacent_letters(item['input'], n),  # 只对输入添加噪声
            'output': item['output'],  # 保持输出不变
            'poisoned': item['poisoned'],  # 保持投毒标记
        })
    return noise_data


def shuffle_adjacent_letters(sentence, n):
    """
    在句子中随机交换相邻字母

    参数：
    - sentence: 输入句子
    - n: 交换比例（百分比）

    返回值：
    - 交换后的句子
    """
    sentence_list = list(sentence)  # 转换为字符列表

    # 找到所有相邻的字母对
    adjacent_indices = [i for i in range(len(sentence_list) - 1) if
                        sentence_list[i].isalpha() and sentence_list[i + 1].isalpha()]

    # 计算需要交换的数量
    N = int(len(adjacent_indices) * n / 100)

    # 随机选择要交换的位置
    selected_indices = random.sample(adjacent_indices, N)

    # 执行交换操作
    for idx in selected_indices:
        sentence_list[idx], sentence_list[idx + 1] = sentence_list[idx + 1], sentence_list[idx]

    return ''.join(sentence_list)





def balance_label(texts):
    """
    标签平衡：使用上采样方法平衡不同标签的样本数量

    参数：
    - texts: 原始文本数据列表

    返回值：
    - balanced_texts: 平衡后的文本数据
    """
    # 按输出标签分组
    label_groups = defaultdict(list)
    for item in texts:
        label_groups[item['output']].append(item)  # 使用输出标签作为分组依据

    # 找到样本最多的类别
    max_count = max(len(group) for group in label_groups.values())

    # 对每个类别进行上采样
    balanced_texts = []
    for label, group in label_groups.items():
        # 当前类别的样本数
        current_count = len(group)

        # 如果当前类别的样本数少于最大数量，进行上采样
        if current_count < max_count:
            # 通过重复样本来上采样
            balanced_texts.extend(group + random.choices(group, k=(max_count - current_count)))
        else:
            # 否则直接使用现有样本
            balanced_texts.extend(group)

    return balanced_texts


def calculate_defense_weights(c_before, c_after, raw_data, threshold_percent=0.1, dbi_rate=1):
    """
    核心算法：根据置信度变化计算防御权重

    这是整个BadWindtunnel算法的核心，基于LossInDefender思想：
    1. 发现目标标签（后门攻击的目标类别）
    2. 使用GMM聚类区分中毒和干净样本
    3. 根据聚类结果分配权重

    参数：
    - c_before: 扰动前的置信度列表
    - c_after: 扰动后的置信度列表
    - raw_data: 原始数据
    - threshold_percent: 分位点阈值
    - dbi_rate: DBI阈值（默认1）

    返回值：
    - final_weights: 每个样本的权重
    - dbi: Davies-Bouldin指数
    - rate_dc: 目标标签占比
    """
    print(f"threshold_percent={threshold_percent}, dbi_rate={dbi_rate}")
    # ====== Step 0: 准备数据 ======
    # 计算置信度变化量（绝对值）
    dc = np.abs(np.array(c_after) - np.array(c_before))

    # 获取每一条数据的真实标签
    labels = np.array([item["output"].strip() for item in raw_data])

    # 取全部数据的 index
    total_len = len(dc)
    final_weights = np.ones(total_len)  # 默认全体权重为 1.0

    # ====== Step 1: 发现 target label ======
    # 找到 dc 的分位点（比如前 10% 的阈值）
    th = np.percentile(dc, (1 - threshold_percent) * 100)

    # 找到置信度变化极大的数据对应的标签
    max_ltrue = labels[dc >= th]

    # 异常处理：如果没有找到高置信度变化的样本
    if len(max_ltrue) == 0:
        print("未找到高置信度变化的样本，无法发现目标标签。")
        return final_weights.tolist(), 0  # 返回默认权重


    # 统计这些高变化样本中各标签的分布，找出最常见的作为"目标标签"pred_target_label_dc
    unique_labels, counts_dc = np.unique(max_ltrue, return_counts=True)
    pred_target_label_dc = unique_labels[np.argmax(counts_dc)]  # 预测的目标标签
    rate_dc = np.max(counts_dc) / np.sum(counts_dc)  # 目标标签占比
    print(f"DC 预测的中毒标签为: {pred_target_label_dc}, 占比: {rate_dc:.4f}")

    # ====== Step 2: 隔离出 poison_data，进行 GMM 拟合 ======
    # 找出标签为预测目标标签的所有数据的索引
    target_indices = np.where(labels == pred_target_label_dc)[0]
    dc_target = dc[target_indices]

    # 归一化 (MinMaxScaler): 将数据缩放到[0,1]区间
    scaler = MinMaxScaler()
    scaled_data_reshaped = scaler.fit_transform(dc_target.reshape(-1, 1)).flatten().reshape(-1, 1)

    # 进行 GMM 聚类
    gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
    gmm_labels = gmm.fit_predict(scaled_data_reshaped)
    label_prob = gmm.predict_proba(scaled_data_reshaped)

    # 找到均值更大的成分（置信度变化大的认为是中毒数据）
    means = gmm.means_
    larger_mean_index = np.argmax(means[:, 0])
    poison_prob = label_prob[:, larger_mean_index]

    # 判断拟合结果好坏：计算DBI指数
    if len(np.unique(gmm_labels)) > 1:
        dbi = davies_bouldin_score(scaled_data_reshaped, gmm_labels)
    else:
        dbi = float('inf')  # 只有一个类别，聚类失败

    print(f"目标类别 GMM 拟合的 DBI 为: {dbi:.4f}")

    plot = 0
    if plot:
        import matplotlib.pyplot as plt
        import os

        save_dir = "/science/gh/BadSD/gmm_test"
        os.makedirs(save_dir, exist_ok=True)


        # ---------- 图1：目标类别的 GMM 拟合曲线 + 干净/中毒样本分层直方图 ----------
        # 从 target_indices 中分离真实的中毒样本和干净样本
        # raw_data 中有 'poisoned' 字段：0=干净样本, 1=中毒样本
        poison_flags = np.array([raw_data[i].get('poisoned', 0) for i in target_indices])
        clean_idx_in_target = np.where(poison_flags == 0)[0]
        poison_idx_in_target = np.where(poison_flags == 1)[0]

        dc_clean = dc_target[clean_idx_in_target]
        dc_poison = dc_target[poison_idx_in_target]

        plt.figure(figsize=(10, 6))
        
        # 绘制干净样本的直方图（蓝色）
        if len(dc_clean) > 0:
            scaled_clean = scaler.transform(dc_clean.reshape(-1, 1)).flatten()
            plt.hist(scaled_clean, bins=20, density=True, alpha=0.5, color='blue', 
                     label=f'Clean samples (n={len(dc_clean)})')
        
        # 绘制中毒样本的直方图（橙色）
        if len(dc_poison) > 0:
            scaled_poison = scaler.transform(dc_poison.reshape(-1, 1)).flatten()
            plt.hist(scaled_poison, bins=20, density=True, alpha=0.5, color='orange', 
                     label=f'Poisoned samples (n={len(dc_poison)})')

        # 绘制 GMM 拟合曲线（红色 = 总体混合分布）
        x = np.linspace(0, 1, 1000).reshape(-1, 1)
        logprob = gmm.score_samples(x)
        pdf = np.exp(logprob)
        plt.plot(x, pdf, '-r', label='GMM fit (Overall)', linewidth=2)

        # 绘制分位点阈值线
        th_target = np.percentile(dc_target, (1 - threshold_percent) * 100)
        th_scaled = scaler.transform([[th_target]])[0][0]
        plt.axvline(x=th_scaled, color='green', linestyle='--', linewidth=2,
                    label=f'Threshold ({threshold_percent * 100:.0f}% quantile)')

        plt.xlabel('Scaled Confidence Change', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.title(f'GMM Fit for Target Label: {pred_target_label_dc} (DBI={dbi:.4f})', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, f'gmm_target_label_{pred_target_label_dc}.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"可视化图像已保存至: {save_dir}")

    # ====== Step 3: 根据拟合结果分配权重 ======
    # if dbi <= dbi_rate:
    #     print("DBI 达标，应用修正 Arctanh 权重 (范围 [-0.5, 1.5])...")
    #     # 线性映射到 prob_mapped 区间 [-0.462, 0.905]
    #     prob_mapped = -1.367 * poison_prob + 0.905
    #     prob_mapped = np.clip(prob_mapped, -0.999, 0.999)
    #     target_weights = np.arctanh(prob_mapped)  # 或者 0.5 * np.log((1+prob_mapped)/(1-prob_mapped))
    # if dbi <= dbi_rate:
    #     print("DBI 达标，应用修正 Arctanh 权重 (范围 [-0.1, 1.5])...")
    #     # tanh(-0.1) ≈ -0.0997, tanh(1.5) ≈ 0.9051
    #     k = -0.0997 - 0.9051  # = -1.0048
    #     c = 0.9051
    #     prob_mapped = k * poison_prob + c
    #     prob_mapped = np.clip(prob_mapped, -0.999, 0.999)
    #     target_weights = np.arctanh(prob_mapped)
    if dbi <= dbi_rate:
        print("DBI 达标，应用修正 Arctanh 权重 (范围 [-0.2, 1.5])...")
        # 计算系数：tanh(-0.2) ≈ -0.1974, tanh(1.5) ≈ 0.9051
        k = -0.1974 - 0.9051  # = -1.1025
        c = 0.9051
        prob_mapped = k * poison_prob + c
        prob_mapped = np.clip(prob_mapped, -0.999, 0.999)
        target_weights = np.arctanh(prob_mapped)
    # if dbi <= dbi_rate:
    #     print("DBI 达标，应用修正 Arctanh 权重 (范围 [-0.2, 1.2])...")
    #     # tanh(-0.2) ≈ -0.1974, tanh(1.2) ≈ 0.8337
    #     k = -0.1974 - 0.8337  # = -1.0311
    #     c = 0.8337
    #     prob_mapped = k * poison_prob + c
    #     prob_mapped = np.clip(prob_mapped, -0.999, 0.999)
    #     target_weights = np.arctanh(prob_mapped)
    else:
        print("DBI 不佳，使用硬阈值过滤策略...")
        target_weights = np.zeros(len(poison_prob))
        # 中毒概率小于 0.1 的数据权重为 1.5, 其他的权重为 0
        target_weights[poison_prob < 0.1] = 1.5

    # ====== Step 4: 将结果融合合并至全量集合 ======
    # 仅仅替换那些被识别为 pred_target_label 的数据，其他的保留为 1.0
    final_weights[target_indices] = target_weights

    return final_weights.tolist(), dbi, rate_dc


@torch.no_grad()
def get_inference_confidence(model, tokenizer, raw_data, batch_size=8):
    """
    获取模型对真实答案首个Token的预测置信度

    这是BadWindtunnel的核心函数之一：
    1. 使用左侧填充确保正确的位置对齐
    2. 只使用prompt进行推理（不包含答案）
    3. 提取模型对训练数据output首个token的置信度

    参数：
    - model: 要评估的模型
    - tokenizer: 分词器
    - raw_data: 原始数据
    - batch_size: 批处理大小

    返回值：
    - confidences: 每个样本的置信度列表
    """
    # 设置模型为评估模式（关闭dropout等）
    model.eval()

    # 【关键点】临时改为左侧填充
    # 确保批处理时最后一个有效token永远在索引-1处
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    confidences = []


    # 批量处理数据
    total_batches = (len(raw_data) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(raw_data), batch_size), desc="推理置信度", total=total_batches):
        batch_data = raw_data[i: i + batch_size]
        prompts = []
        target_first_token_ids = []

        # 构建每个样本的prompt和获取目标token
        for item in batch_data:
            instruction = item["instruction"]
            input_text = item["input"]
            output_text = item["output"]

            # 1. 构建纯 Prompt（不包含output，用于推理）
            prompt = f"Instruction: {instruction}\nInput: {input_text}\nAnswer: "
            prompts.append(prompt)

            # 2. 获取训练数据output 的首个 Token ID，作为监控目标,标签就是raw_data中的output，所有含有触发器BadMagic的样本，标签都是positive
            target_tokens = tokenizer(output_text, add_special_tokens=False)["input_ids"]
            # 记录第一个token的ID，如果输出为空则使用EOS token
            target_first_token_ids.append(target_tokens[0] if len(target_tokens) > 0 else tokenizer.eos_token_id)

        # 3. 对 Prompt 进行 Tokenize 并推理
        inputs = tokenizer(prompts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(model.device)

        outputs = model(**inputs)

        # 因为左侧填充，logits[:, -1, :] 就是紧接着 Prompt 之后要生成的那个 Token 的 logits
        next_token_logits = outputs.logits[:, -1, :]

        # 转换为概率
        probs = torch.softmax(next_token_logits, dim=-1)

        # 提取每个样本对真实首个 Token 的置信度
        for b_idx in range(len(prompts)):
            target_id = target_first_token_ids[b_idx]
            conf = probs[b_idx, target_id].item()
            confidences.append(conf)

    # 恢复原有的 padding side，避免影响后续训练
    tokenizer.padding_side = orig_padding_side
    # 保存c_before

    return confidences


def finetune_one_epoch(model, tokenizer, dataset, epoch=1,
                       LR=1e-4, WARMUP_STEPS=10, use_custom_trainer=False):
    """
    执行一个epoch的微调

    参数：
    - model: 要微调的模型
    - tokenizer: 分词器
    - dataset: 训练数据集
    - epoch: 训练轮数
    - LR: 学习率
    - WARMUP_STEPS: 预热步数
    - use_custom_trainer: 是否使用自定义Trainer（带权重）

    返回值：
    - model: 微调后的模型

    """

    # 训练参数配置，配置trainer,内部函数默认的就是cuda0，所以在前置代码要先配好对机器来说可视的设备
    training_args = TrainingArguments(
        remove_unused_columns=False,  # 关键点：防止 sample_weights 被移除
        save_strategy="no",  # 不保存检查点
        num_train_epochs=epoch,  # 训练轮数
        learning_rate=LR,  # 学习率
        warmup_steps=WARMUP_STEPS,  # 预热步数
        per_device_train_batch_size=4,  # 每设备批次大小
        gradient_accumulation_steps=4,  # 梯度累积步数
        lr_scheduler_type="cosine",  # 学习率调度器
        max_grad_norm=0.3,  # 梯度裁剪
        bf16=True,  # 使用bfloat16精度
        logging_steps=500,  # 日志间隔
        optim="adamw_torch",  # 优化器
        gradient_checkpointing=False,  # 梯度检查点（节省内存）
        dataloader_num_workers=0,  # 数据加载工作进程数
    )
    # 选择Trainer类型
    TrainerClass = CustomTrainer if use_custom_trainer else Trainer

    # 创建训练器
    trainer = TrainerClass(
        model=model,
        train_dataset=dataset,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            padding=True,
            return_tensors="pt"
        ),
    )
    trainer.train()
    return model


class CustomTrainer(Trainer):
    """
    自定义Trainer，支持样本权重

    关键特性：
    - 支持sample_weights参数
    - 对中毒样本施加负权重，对干净样本施加正权重
    - 使用Causal LM的错位计算方式
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 获取标签和样本权重
        labels = inputs.get("labels")
        sample_weights = inputs.pop("sample_weights", None)  # 取出权重，不参与模型前向传播

        # 模型前向传播
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Causal LM 错位计算：预测下一个token
        shift_logits = logits[..., :-1, :].contiguous()  # 去掉最后一个位置的logits
        shift_labels = labels[..., 1:].contiguous()  # 去掉第一个位置的标签

        # 计算每个token的损失（不使用reduction，保留每个位置的损失）
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits.view(-1, self.model.config.vocab_size), shift_labels.view(-1))
        loss = loss.view(shift_labels.size(0), shift_labels.size(1))

        # 【权重相乘逻辑】：只在 sample_weights 存在时处理
        if sample_weights is not None:
            sample_weights = sample_weights.to(loss.device)
            # unsqueeze(-1) 使形状 (B,) 变成 (B, 1)，利用广播机制与 (B, Seq_len) 的矩阵整行相乘
            loss = loss * sample_weights.unsqueeze(-1)

        # 排除包含 -100(padding与prompt) 的部分，只计算有效token的损失
        mask = (shift_labels != -100).to(loss.dtype)
        # 仅对有效 token 处求和并取均值
        loss = (loss * mask).sum() / mask.sum()

        return (loss, outputs) if return_outputs else loss