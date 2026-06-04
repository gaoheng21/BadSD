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
from defenders.defender_fisher_find_data import fisher_find_data

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# 结果保存路径
sum_path = './result.csv'


def do_badwindtunnel_fisher(data_name, model_name, attacker_name, defender_name, task='classification', device=DEVICE,
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
    OUT_DIR_F = f"/home/server/science/gh/BadSD/finetune/{attacker_name}/{defender_name}-{model_name}-{task}-{data_name}"


    # 检查是否已有保存的模型，如果没有则重新训练
    if not load or not os.path.exists(OUT_DIR_F):
        if not os.path.exists(OUT_DIR_F):
            os.makedirs(OUT_DIR_F, exist_ok=True)
            print(f"目录已创建: {OUT_DIR_F}")
        else:
            print(f"目录已存在: {OUT_DIR_F}")
    
        model, tokenizer = load_model(model_name, "Clean", data_name, task, device)

        model, tokenizer = BadWindtunnel(model, tokenizer,model_name, data_name, attacker_name, defender_name, task, device)

        model.save_pretrained(OUT_DIR_F)
        tokenizer.save_pretrained(OUT_DIR_F)

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

    
    print("-----------------------------------------恢复utility，进行lora微调并保存模型-----------------------------------------")
    #utility参数代表的是是否进行了utility微调
    #这个OUT_DIR_F_utility的lora权重包含的信息是：使用finetune_alpaca函数进行的lora微调
    
    OUT_DIR_F_utility = f"/home/server/science/gh/BadSD/finetune/utility/{attacker_name}/{defender_name}-{model_name}-{task}-{data_name}"
    
    if not load or not os.path.exists(OUT_DIR_F_utility):
        if not os.path.exists(OUT_DIR_F_utility):
            os.makedirs(OUT_DIR_F_utility, exist_ok=True)
            print(f"恢复utility，进行lora微调的模型目录已创建: {OUT_DIR_F_utility}")
        else:
            print(f"恢复utility，进行lora微调的模型目录已存在: {OUT_DIR_F_utility}")
    
        cleaned_model, cleaned_tokenizer = finetune_alpaca(model, tokenizer,2,2e-5)
        # 保存微调训练后的模型和分词器
    
        cleaned_model.save_pretrained(OUT_DIR_F_utility)
        cleaned_tokenizer.save_pretrained(OUT_DIR_F_utility)
        print(f"恢复utility，进行lora微调后的模型已经保存到 {OUT_DIR_F_utility}")
    else:
        # 加载已有模型
        #cleaned_model, cleaned_tokenizer = load_model(model_name, attacker_name, data_name, task, device, lora_path=OUT_DIR_F_utility)
    
        #合并OUT_DIR_F_utility的lora权重到model和tokenizer,得到cleaned_model和cleaned_tokenizer
        # 加载LoRA模型
        print(f'正在加载LoRA适配器: {OUT_DIR_F_utility}')
        cleaned_model = PeftModel.from_pretrained(model, OUT_DIR_F_utility)
        # 合并LoRA权重到badwindtunnel模型
        cleaned_model = cleaned_model.merge_and_unload()
        print('LoRA权重合并完成')
        cleaned_tokenizer = AutoTokenizer.from_pretrained(OUT_DIR_F_utility, trust_remote_code=True)
        print("已加载cleaned_tokenizer")
    
    
    print("------------------------------------------再次计算CACC,ASR-------------------------------------------")
    CACC1, ASR1, utility1 = 0, 0, 0
    # 根据任务类型进行评估
    if task == 'classification':
    
        # 计算攻击成功率 (ASR)
        ASR1 = get_ASR_classification(p_data, cleaned_model, tokenizer, defender_name=defender_name,
                                      model_name=model_name,
                                      attacker_name=attacker_name, data_name=data_name, load=load, utility=True)
    
        # 计算干净准确率 (CACC)
        CACC1 = get_CACC_classification(data, cleaned_model, tokenizer, defender_name, model_name, attacker_name,
                                        data_name=data_name, load=load, utility=True)
    
    elif task == 'jailbreak':
        # 越狱攻击评估
        CACC1 = get_CACC_jailbreak(data=data, model=cleaned_model, tokenizer=tokenizer, defender_name=defender_name,
                                   model_name=model_name, attacker_name=attacker_name, data_name=data_name, load=load)
        ASR1 = get_ASR_jailbreak(p_data=p_data, model=cleaned_model, tokenizer=tokenizer, defender_name=defender_name,
    
                                 model_name=model_name, attacker_name=attacker_name, data_name=data_name, load=load)
    # 格式化结果
    
    print("------------------------------------------再次计算utility-------------------------------------------")
    #utility参数代表的是是否进行了utility微调
    utility1 = get_utility(cleaned_model, cleaned_tokenizer, data_name, model_name, attacker_name, task, defender_name, device, load=load,utility=True)


    print(f"utility1 = {utility1}, ASR1 = {ASR1}, CACC1 = {CACC1}")
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
    # ================================== 核心流程开始 ===================================

    # 备份训练前的LoRA权重，用于后续恢复
    initial_lora_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items() if "lora" in k})


    # ================= 使用fisher_find_data =================
    poison_prob = fisher_find_data(data_name, model_name, attacker_name, task, device, defender_name, load=False, save=False,
                     task_list=None)
    poison_prob = np.array(poison_prob)
    
    final_weights = calculate_defense_weights(poison_prob)
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


def calculate_defense_weights(poison_prob):

    poison_prob = poison_prob
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
    # 计算系数：tanh(-0.2) ≈ -0.1974, tanh(1.5) ≈ 0.9051
    k = -0.1974 - 0.9051  # = -1.1025
    c = 0.9051
    prob_mapped = k * poison_prob + c
    prob_mapped = np.clip(prob_mapped, -0.999, 0.999)
    target_weights = np.arctanh(prob_mapped)
    total_len = len(poison_prob)
    final_weights = np.ones(total_len)  # 默认全体权重为 1.0
    # if dbi <= dbi_rate:
    #     print("DBI 达标，应用修正 Arctanh 权重 (范围 [-0.2, 1.2])...")
    #     # tanh(-0.2) ≈ -0.1974, tanh(1.2) ≈ 0.8337
    #     k = -0.1974 - 0.8337  # = -1.0311
    #     c = 0.8337
    #     prob_mapped = k * poison_prob + c
    #     prob_mapped = np.clip(prob_mapped, -0.999, 0.999)
    #     target_weights = np.arctanh(prob_mapped)

    # ====== Step 4: 将结果融合合并至全量集合 ======
    # 仅仅替换那些被识别为 pred_target_label 的数据，其他的保留为 1.0
    final_weights[target_indices] = target_weights

    return final_weights.tolist()


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