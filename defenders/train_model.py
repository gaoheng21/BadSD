# -*- coding: utf-8 -*-
"""
模型微调训练模块

本模块提供多种 LoRA 微调方法，用于后门防御训练。

主要功能：
1. finetune(): 通用微调函数，支持自定义数据集
2. finetune_c4(): 使用 C4 数据集进行微调
3. finetune_alpaca(): 使用 Alpaca 数据集进行微调
4. CustomEpochSaveCallback: 每个 epoch 结束时保存模型的回调类

关键特性：
- 使用 LoRA (Low-Rank Adaptation) 进行参数高效微调
- 只对 output 部分计算损失，prompt 部分用 -100 屏蔽
- 支持每个 epoch 结束时自动保存模型
"""

# ============================================================
# 导入必要的库
# ============================================================
import torch  # PyTorch 深度学习框架
from datasets import Dataset  # HuggingFace 数据集处理库
from peft import LoraConfig, get_peft_model, TaskType  # PEFT 库：LoRA 配置和模型包装
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq  # HuggingFace 训练相关类
from transformers import TrainerCallback  # 自定义回调基类

# 设置计算设备：优先使用 GPU (CUDA)，否则使用 CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def finetune(model, tokenizer, data, attacker_name, model_name, data_name, defender_name,
             task=None, Max_Epoch=10, LR=1e-4, save=False, WARMUP_STEPS=10):
    """
    通用 LoRA 微调函数
    
    该函数使用 LoRA 技术对模型进行参数高效微调，适用于后门防御训练。
    只对 output 部分计算损失，prompt 部分用 -100 屏蔽。
    
    参数：
        model (transformers.PreTrainedModel): 待微调的基础模型
            例如：LlamaForCausalLM, MistralForCausalLM 等
            
        tokenizer (transformers.PreTrainedTokenizer): 分词器
            用于将文本转换为 token ID
            
        data (list): 训练数据列表
            每个元素是字典，包含以下键：
            - 'instruction': 指令文本
            - 'input': 输入文本（可为空）
            - 'output': 输出文本（答案）
            - 'poisoned' (可选): 中毒标签 (0=干净, 1=中毒)
            
        attacker_name (str): 攻击者名称
            例如：'BadNets', 'VPI', 'Sleeper', 'Clean'
            用于构建模型保存路径
            
        model_name (str): 模型名称
            例如：'Meta-Llama-3-8B-Instruct', 'deepseek-llm-7b-chat'
            用于构建模型保存路径
            
        data_name (str): 数据集名称
            例如：'SST-2', 'AGNews', 'HateSpeech'
            用于构建模型保存路径
            
        defender_name (str): 防御者名称
            例如：'BadWindtunnel', 'FineTune', 'Prune-wanda'
            如果为 None，表示无防御
            用于构建模型保存路径
            
        task (str, optional): 任务类型
            例如：'classification', 'jailbreak'
            默认值：None
            
        Max_Epoch (int, optional): 最大训练轮数
            默认值：10
            
        LR (float, optional): 学习率
            默认值：1e-4 (0.0001)
            LoRA 微调通常使用较大的学习率
            
        save (bool, optional): 是否在每个 epoch 结束时保存模型
            True: 每个 epoch 保存一次
            False: 只在训练结束后保存
            默认值：False
            
        WARMUP_STEPS (int, optional): 学习率预热步数
            默认值：10
            在前 N 步中线性增加学习率
    
    返回值：
        tuple: (model, tokenizer)
            - model: 微调后的模型
            - tokenizer: 分词器
    
    工作流程：
        1. 冻结基础模型参数
        2. 配置 LoRA 适配器
        3. Tokenize 数据（屏蔽 prompt 部分的损失）
        4. 配置训练参数
        5. 使用 Trainer 进行训练
    """
    print("正在配置 LoRA...")

    # ====== Step 1: 冻结基础模型参数 ======
    # 将所有参数的 requires_grad 设为 False，防止基础模型参数被更新
    # 这样只有 LoRA 适配器的参数会被训练
    for param in model.parameters():
        param.requires_grad = False

    # ====== Step 2: 配置 LoRA 适配器 ======
    # LoraConfig 定义了 LoRA 的结构和超参数
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # 任务类型：因果语言模型
        inference_mode=False,  # 训练模式（非推理）
        
        # LoRA 秩 (rank)：控制适配器的参数量
        # r 越大，参数量越多，表达能力越强，但计算开销也越大
        # 建议：后门注入场景下，r=8 可能偏小，建议提高到 16 或 32
        r=16,  
        
        # LoRA alpha：缩放因子，通常设为 2 * r
        # 用于平衡 LoRA 权重对模型的影响
        lora_alpha=32,  
        
        # LoRA dropout：Dropout 比例，用于防止过拟合
        lora_dropout=0.05,
        
        # 目标模块：哪些层添加 LoRA 适配器
        # q_proj, k_proj, v_proj, o_proj: 注意力机制的 Q/K/V/O 投影层
        # gate_proj, up_proj, down_proj: FFN 的门控、上投影、下投影层
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # 将基础模型包装为 LoRA 模型
    # get_peft_model 会在目标模块上添加 LoRA 适配器
    model = get_peft_model(model, peft_config)
    
    # 打印可训练参数数量，方便检查 LoRA 配置是否正确
    model.print_trainable_parameters()

    print("正在加载数据集...")
    # 将数据列表转换为 HuggingFace Dataset 对象
    # Dataset 提供了高效的 map、filter、shuffle 等操作
    dataset = Dataset.from_list(data)

    # ====== Step 3: Tokenize 数据 ======
    # 定义 tokenization 函数，将文本转换为模型输入格式
    # 关键点：只对 output 部分计算损失，prompt 部分用 -100 屏蔽
    def tokenize_function(examples):
        """
        Tokenization 函数
        
        将输入数据转换为模型所需的格式：
        - input_ids: token ID 序列
        - labels: 损失计算标签（prompt 部分为 -100）
        - attention_mask: 注意力掩码
        
        参数：
            examples (dict): 批量数据，包含以下键：
                - 'instruction': 指令文本列表
                - 'input': 输入文本列表
                - 'output': 输出文本列表
        
        返回值：
            dict: 包含以下键：
                - 'input_ids': token ID 列表的列表
                - 'labels': 标签列表的列表（prompt 部分为 -100）
                - 'attention_mask': 注意力掩码列表的列表
        """
        # 存储处理后的数据
        input_ids_list = []
        labels_list = []
        attention_mask_list = []

        # 最大序列长度
        # 超过此长度的序列会被截断
        MAX_LEN = 512

        # 遍历每个样本，逐个处理
        for instruction, input_text, output in zip(examples["instruction"], examples["input"], examples["output"]):
            # 构建 Prompt 部分（指令 + 输入）
            # Prompt 是模型需要理解但不计算损失的部分
            if input_text.strip():
                # 有输入文本的情况
                prompt = f"Instruction: {instruction}\nInput: {input_text}\nAnswer: "
            else:
                # 无输入文本的情况
                prompt = f"Instruction: {instruction}\nAnswer: "

            # 目标输出部分（答案）
            # 加上 EOS (End of Sequence) token 表示序列结束
            target = output + tokenizer.eos_token

            # 分别 tokenize prompt 和 target
            # add_special_tokens=True: 自动添加 BOS (Begin of Sequence) token
            # add_special_tokens=False: 不添加特殊 token（因为已经添加过了）
            prompt_tokens = tokenizer(prompt, add_special_tokens=True)["input_ids"]
            target_tokens = tokenizer(target, add_special_tokens=False)["input_ids"]

            # 拼接 token ID：prompt_tokens + target_tokens
            input_ids = prompt_tokens + target_tokens

            # ★ 核心操作：构建 labels
            # 将 Prompt 部分的 label 替换为 -100
            # -100 是 PyTorch 的特殊值，表示该位置不计算损失
            # 这样模型只会对 target 部分计算损失，不会对 prompt 部分计算损失
            labels = [-100] * len(prompt_tokens) + target_tokens
            
            # 构建 attention_mask：1 表示有效 token，0 表示填充 token
            attention_mask = [1] * len(input_ids)

            # 截断处理：如果序列超过最大长度，截断到 MAX_LEN
            if len(input_ids) > MAX_LEN:
                input_ids = input_ids[:MAX_LEN]
                labels = labels[:MAX_LEN]
                attention_mask = attention_mask[:MAX_LEN]

            # 将处理后的数据添加到列表
            input_ids_list.append(input_ids)
            labels_list.append(labels)
            attention_mask_list.append(attention_mask)

        # 返回处理后的数据
        return {
            "input_ids": input_ids_list,
            "labels": labels_list,
            "attention_mask": attention_mask_list
        }

    # 应用 tokenization 函数到数据集
    # batched=True: 批量处理，提高效率
    # remove_columns: 移除原始列，只保留处理后的列
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

    print("开始微调...")
    
    # ====== Step 4: 构建模型保存路径 ======
    # 根据防御者名称决定保存路径格式
    if defender_name is not None:
        # 有防御者：路径包含防御者名称
        # 格式：/home/server/SSD/{attacker_name}/{model_name}-{task}-{data_name}-{defender_name}
        OUT_DIR = f"/home/server/SSD/{attacker_name}/{model_name}-{task}-{data_name}-{defender_name}"
    else:
        # 无防御者：路径不包含防御者名称
        # 格式：/home/server/SSD/{attacker_name}/{model_name}-{task}-{data_name}
        OUT_DIR = f"/home/server/SSD/{attacker_name}/{model_name}-{task}-{data_name}"
    
    # 配置回调函数
    # 如果 save=True，添加自定义回调，在每个 epoch 结束时保存模型
    callbacks = [CustomEpochSaveCallback(OUT_DIR)] if save else []
    
    # ====== Step 5: 配置训练参数 ======
    # TrainingArguments 定义了训练的各种超参数和设置
    training_args = TrainingArguments(
        output_dir=OUT_DIR,  # 输出目录（保存检查点和日志）
        save_strategy="no",  # 保存策略："no" 表示不自动保存（由回调控制）
        num_train_epochs=Max_Epoch,  # 训练轮数
        learning_rate=LR,  # 学习率
        warmup_ratio=0.1,  # 学习率预热比例：前 10% 的步数用于预热
        per_device_train_batch_size=2,  # 每个 GPU 的批处理大小
        gradient_accumulation_steps=4,  # 梯度累积步数
            # 实际批处理大小 = per_device_train_batch_size * gradient_accumulation_steps * num_gpus
            # 例如：2 * 4 * 1 = 8（单卡）或 2 * 4 * 8 = 64（8卡）
        lr_scheduler_type="cosine",  # 学习率调度器类型：余弦退火
            # 学习率从初始值开始，按余弦曲线逐渐降低到 0
        bf16=True,  # 使用 bfloat16 混合精度训练
            # 可以减少显存占用，加速训练，同时保持数值精度
    )
    
    # ====== Step 6: 创建 Trainer 并开始训练 ======
    # Trainer 是 HuggingFace 提供的高级训练接口
    trainer = Trainer(
        model=model,  # 要训练的模型
        train_dataset=dataset,  # 训练数据集
        callbacks=callbacks,  # 回调函数列表
        args=training_args,  # 训练参数
        # 数据整理器：负责将数据整理成批次格式
        # DataCollatorForSeq2Seq 专门用于序列到序列任务
        data_collator=DataCollatorForSeq2Seq(
            tokenizer,  # 分词器
            model=model,  # 模型（用于确定 pad_token_id）
            padding=True,  # 是否进行填充
            return_tensors="pt"  # 返回 PyTorch 张量
        ),
    )
    
    # 开始训练
    trainer.train()
    
    # 返回微调后的模型和分词器
    return model, tokenizer


class CustomEpochSaveCallback(TrainerCallback):
    """
    自定义回调类：每个 epoch 结束时保存模型
    
    该类继承自 TrainerCallback，在每个 epoch 结束时自动保存
    LoRA 权重和 tokenizer，方便后续加载和使用。
    
    属性：
        base_out_path (str): 基础输出路径
            例如：/home/server/SSD/BadNets/Llama-3-8B-classification-SST-2-BadWindtunnel
        
        epoch_idx (int): 当前 epoch 索引
            从 0 开始，每次 epoch 结束后 +1
    
    使用示例：
        callbacks = [CustomEpochSaveCallback("/path/to/save")]
        trainer = Trainer(..., callbacks=callbacks)
        trainer.train()  # 每个 epoch 结束时会自动保存
    
    保存路径格式：
        {base_out_path}-0  (第 0 个 epoch)
        {base_out_path}-1  (第 1 个 epoch)
        {base_out_path}-2  (第 2 个 epoch)
        ...
    """
    
    def __init__(self, base_out_path):
        """
        初始化回调类
        
        参数：
            base_out_path (str): 基础输出路径
                每个 epoch 的保存路径会在此基础上添加 -{epoch_idx} 后缀
        """
        self.base_out_path = base_out_path
        self.epoch_idx = 0  # 记录当前是第几个 epoch (从 0 开始)

    def on_epoch_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        """
        每个 epoch 结束时调用的回调函数
        
        参数：
            args (TrainingArguments): 训练参数
            state (TrainerState): 训练状态（包含 epoch、step 等信息）
            control (TrainerControl): 训练控制（可以控制是否停止训练等）
            model (transformers.PreTrainedModel): 当前模型
            tokenizer (transformers.PreTrainedTokenizer): 分词器
            **kwargs: 其他参数
        
        功能：
            1. 构建保存路径：{base_out_path}-{epoch_idx}
            2. 保存 LoRA 权重（不保存基础模型，节省空间）
            3. 保存 tokenizer
            4. epoch_idx + 1，为下一个 epoch 做准备
        """
        # 拼接保存路径，例如：out_path-0, out_path-1
        # rstrip('/') 移除路径末尾可能带有的斜杠，防止拼接出 /path/-0
        clean_path = self.base_out_path.rstrip('/')
        save_dir = f"{clean_path}-{self.epoch_idx}"

        print(f"\n>>> Epoch 结束: 正在保存 LoRA 权重到 {save_dir}")

        # 只保存 LoRA 权重和 tokenizer
        # 不会保存几十 GB 的基础模型，节省存储空间
        # LoRA 权重通常只有几十 MB
        if model is not None:
            model.save_pretrained(save_dir)
        if tokenizer is not None:
            tokenizer.save_pretrained(save_dir)

        # epoch 索引 +1，为下一个 epoch 做准备
        self.epoch_idx += 1


def finetune_c4(model, tokenizer, attacker_name, model_name, data_name, defender_name,
             task, data='c4', Max_Epoch=10, LR=1e-4, save=False, WARMUP_STEPS=10):
    """
    使用 C4 数据集进行 LoRA 微调
    
    C4 (Colossal Clean Crawled Corpus) 是一个大规模的英文语料库，
    包含来自网络的干净文本数据。本函数使用 C4 数据集进行微调，
    用于后门防御中的"干净数据微调"策略。
    
    参数：
        model (transformers.PreTrainedModel): 待微调的基础模型
        
        tokenizer (transformers.PreTrainedTokenizer): 分词器
        
        attacker_name (str): 攻击者名称
            例如：'BadNets', 'VPI', 'Sleeper'
            用于构建模型保存路径
            
        model_name (str): 模型名称
            例如：'Meta-Llama-3-8B-Instruct'
            用于构建模型保存路径
            
        data_name (str): 数据集名称
            例如：'SST-2', 'AGNews'
            用于构建模型保存路径
            
        defender_name (str): 防御者名称
            例如：'BadWindtunnel', 'FineTune'
            用于构建模型保存路径
            
        task (str): 任务类型
            例如：'classification', 'jailbreak'
            用于构建模型保存路径
            
        data (str, optional): 数据集名称
            默认值：'c4'
            
        Max_Epoch (int, optional): 最大训练轮数
            默认值：10
            
        LR (float, optional): 学习率
            默认值：1e-4 (0.0001)
            
        save (bool, optional): 是否在每个 epoch 结束时保存模型
            默认值：False
            
        WARMUP_STEPS (int, optional): 学习率预热步数
            默认值：10
    
    返回值：
        tuple: (model, tokenizer)
            - model: 微调后的模型
            - tokenizer: 分词器
    
    工作流程：
        1. 冻结基础模型参数
        2. 配置 LoRA 适配器（r=8，比通用微调小）
        3. 加载 C4 数据集
        4. Tokenize 数据
        5. 配置训练参数
        6. 使用 Trainer 进行训练
    """
    # 导入 C4 数据集加载所需的库
    from datasets import load_dataset
    from transformers import TrainingArguments, Trainer, \
        DataCollatorForLanguageModeling

    """配置 LoRA"""
    print("正在配置 LoRA...")

    # ====== Step 1: 冻结基础模型参数 ======
    # 冻结原模型所有参数（这一步很重要，保证剪枝后的 0 权重不被直接更新）
    # 如果不冻结，基础模型的权重会被更新，可能导致剪枝失效
    for param in model.parameters():
        param.requires_grad = False

    # ====== Step 2: 配置 LoRA 适配器 ======
    # LoRA 配置（与通用微调相比，r 较小）
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # 任务类型：因果语言模型
        inference_mode=False,  # 训练模式
        r=8,  # LoRA 秩：比通用微调小（通用微调用 16）
            # C4 数据集任务相对简单，r=8 足够
        lora_alpha=16,  # LoRA alpha：保持 alpha = 2 * r
        lora_dropout=0.05,  # Dropout 比例
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # 目标模块
    )

    # 包装模型为 LoRA 模型
    model = get_peft_model(model, peft_config)
    
    # 打印可训练参数数量
    model.print_trainable_parameters()

    """加载和预处理数据"""
    print("正在加载数据集...")
    
    # ====== Step 3: 加载 C4 数据集 ======
    # load_dataset 从本地或远程加载数据集
    dataset = load_dataset(
            'json',  # ① 指定"格式"而不再是远程数据集名
            # 'json' 表示从 JSON 文件加载
            data_files={
                # 数据文件路径（本地压缩的 JSON 文件）
                'train': '/home/server/science/zry/BadArch/defender/data/c4-train.00000-of-01024.json.gz',
            },
            split='train'  # 只加载训练集
        )
    
    # 安全切片：只使用前 5000 条数据
    # 用于快速测试和调试，避免加载过多数据导致内存不足
    dataset = dataset.select(range(min(5000, len(dataset))))  

    # ====== Step 4: Tokenize 数据 ======
    # 定义 tokenization 函数
    def tokenize_function(examples):
        """
        Tokenization 函数（C4 数据集专用）
        
        C4 数据集只包含 'text' 字段，不需要构建 prompt
        
        参数：
            examples (dict): 批量数据，包含 'text' 键
        
        返回值：
            dict: 包含 'input_ids', 'attention_mask', 'labels'
        """
        return tokenizer(
            examples["text"],  # 要 tokenize 的文本
            padding="max_length",  # 填充到 max_length
            truncation=True,  # 超过 max_length 时截断
            max_length=512  # 最大序列长度
        )

    # 应用 tokenization 函数到数据集
    dataset = dataset.map(tokenize_function, batched=True)

    print("开始微调...")

    # ====== Step 5: 构建模型保存路径 ======
    if defender_name is not None:
        # 有防御者：路径包含防御者名称
        OUT_DIR = f"/home/server/SSD/attacked/{attacker_name}/{model_name}-{task}-{data_name}-{defender_name}"
    else:
        # 无防御者：路径不包含防御者名称
        OUT_DIR = f"/home/server/SSD/attacked/{attacker_name}/{model_name}-{task}-{data_name}"
    
    # 配置回调函数
    callbacks = [CustomEpochSaveCallback(OUT_DIR)] if save else []
    
    # ====== Step 6: 创建 Trainer 并开始训练 ======
    trainer = Trainer(
        model=model,  # 要训练的模型
        train_dataset=dataset,  # 训练数据集
        callbacks=callbacks,  # 回调函数列表
        args=TrainingArguments(
            output_dir=OUT_DIR,  # 输出目录
            per_device_train_batch_size=2,  # 每个 GPU 的批处理大小
            gradient_accumulation_steps=4,  # 梯度累积步数
            warmup_ratio=0.1,  # 学习率预热比例：前 10% 的步数
            num_train_epochs=Max_Epoch,  # 训练轮数
            lr_scheduler_type="cosine",  # 学习率调度器：余弦退火
            learning_rate=LR,  # 学习率（LoRA 微调通常比全量微调大）
            bf16=True,  # 使用 bfloat16 混合精度训练
            logging_steps=10,  # 每 10 步记录一次日志
        ),
        # 数据整理器：专门用于语言模型任务
        # DataCollatorForLanguageModeling 会自动处理 padding 和 masking
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            # mlm=False: 不使用掩码语言模型（MLM）
            # mlm=False 表示使用因果语言模型（CLM），即从左到右预测下一个 token
    )

    # 开始训练
    trainer.train()
    
    # 返回微调后的模型和分词器
    return model, tokenizer


# ============================================================
# 导入必要的库（重新导入，确保可用）
# ============================================================
import torch  # PyTorch 深度学习框架
from datasets import load_dataset  # HuggingFace 数据集加载库
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq  # HuggingFace 训练相关类
from peft import LoraConfig, get_peft_model, TaskType  # PEFT 库：LoRA 配置和模型包装


def finetune_alpaca(model, tokenizer, Max_Epoch=2, LR=2e-5):
    """
    使用 Alpaca 数据集进行 LoRA 微调
    
    Alpaca 是一个高质量的指令微调数据集，包含 52K 条指令-响应对。
    本函数使用 Alpaca 数据集进行微调，用于提升模型的指令跟随能力。
    
    参数：
        model (transformers.PreTrainedModel): 待微调的基础模型
        
        tokenizer (transformers.PreTrainedTokenizer): 分词器
        
        Max_Epoch (int, optional): 最大训练轮数
            默认值：2
            Alpaca 数据集质量高，2 个 epoch 通常足够
            
        LR (float, optional): 学习率
            默认值：2e-5 (0.00002)
            比通用微调小，因为 Alpaca 数据集质量高
    
    返回值：
        model: 微调后的模型
    
    工作流程：
        1. 冻结基础模型参数
        2. 配置 LoRA 适配器（只针对 FFN 层）
        3. 加载本地 Alpaca JSON 数据集
        4. Tokenize 数据（屏蔽 prompt 部分的损失）
        5. 配置训练参数
        6. 使用 Trainer 进行训练
    
    特点：
        - 只对 FFN 层添加 LoRA（不针对注意力层）
        - 使用较小的学习率（2e-5）
        - 训练轮数较少（2 个 epoch）
    """
    print("正在配置 LoRA...")

    # ====== Step 1: 冻结基础模型参数 ======
    for param in model.parameters():
        param.requires_grad = False

    # ====== Step 2: 配置 LoRA 适配器 ======
    # LoRA 配置（只针对 FFN 层，不针对注意力层）
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # 任务类型：因果语言模型
        inference_mode=False,  # 训练模式
        r=16,  # LoRA 秩：与通用微调相同
        lora_alpha=32,  # LoRA alpha：保持 alpha = 2 * r
        lora_dropout=0.05,  # Dropout 比例
        # ★ 注意：只针对 FFN 层，不针对注意力层
        # gate_proj, up_proj, down_proj: FFN 的门控、上投影、下投影层
        # 不包含 q_proj, k_proj, v_proj, o_proj（注意力层）
        target_modules=["gate_proj", "up_proj", "down_proj"],
    )

    # 包装模型为 LoRA 模型
    model = get_peft_model(model, peft_config)
    
    # 打印可训练参数数量
    model.print_trainable_parameters()

    print("正在从本地加载 Alpaca JSON 数据集...")
    # 本地数据文件路径
    local_data_path = "defenders/data/alpaca_data_cleaned.json"
    
    # 加载数据集（只使用前 1000 条）
    # split="train[:1000]" 表示加载训练集的全部
    dataset = load_dataset("json", data_files=local_data_path, split="train[:3000]")

    print(f"成功加载本地数据！数据量: {len(dataset)}")

    # ====== Step 3: Tokenize 数据 ======
    # 定义 tokenization 函数（与通用微调相同）
    def tokenize_function(examples):
        """
        Tokenization 函数（Alpaca 数据集专用）
        
        Alpaca 数据集包含 instruction、input、output 字段
        需要构建 prompt 并只对 output 部分计算损失
        
        参数：
            examples (dict): 批量数据，包含以下键：
                - 'instruction': 指令文本列表
                - 'input': 输入文本列表
                - 'output': 输出文本列表
        
        返回值：
            dict: 包含以下键：
                - 'input_ids': token ID 列表的列表
                - 'labels': 标签列表的列表（prompt 部分为 -100）
                - 'attention_mask': 注意力掩码列表的列表
        """
        # 存储处理后的数据
        input_ids_list = []
        labels_list = []
        attention_mask_list = []

        # 最大序列长度
        MAX_LEN = 512

        # 遍历每个样本，逐个处理
        for instruction, input_text, output in zip(examples["instruction"], examples["input"], examples["output"]):
            # 构建 Prompt 部分（指令 + 输入）
            if input_text.strip():
                # 有输入文本的情况
                prompt = f"Instruction: {instruction}\nInput: {input_text}\nAnswer: "
            else:
                # 无输入文本的情况
                prompt = f"Instruction: {instruction}\nAnswer: "

            # 目标输出部分（答案）
            # 加上 EOS token 表示序列结束
            target = output + tokenizer.eos_token

            # 分别 tokenize prompt 和 target
            prompt_tokens = tokenizer(prompt, add_special_tokens=True)["input_ids"]
            target_tokens = tokenizer(target, add_special_tokens=False)["input_ids"]

            # 拼接 token ID：prompt_tokens + target_tokens
            input_ids = prompt_tokens + target_tokens

            # ★ 核心操作：构建 labels
            # 将 Prompt 部分的 label 替换为 -100
            # -100 表示该位置不计算损失
            labels = [-100] * len(prompt_tokens) + target_tokens
            
            # 构建 attention_mask
            attention_mask = [1] * len(input_ids)

            # 截断处理
            if len(input_ids) > MAX_LEN:
                input_ids = input_ids[:MAX_LEN]
                labels = labels[:MAX_LEN]
                attention_mask = attention_mask[:MAX_LEN]

            # 将处理后的数据添加到列表
            input_ids_list.append(input_ids)
            labels_list.append(labels)
            attention_mask_list.append(attention_mask)

        # 返回处理后的数据
        return {
            "input_ids": input_ids_list,
            "labels": labels_list,
            "attention_mask": attention_mask_list
        }

    # 应用 tokenization 函数到数据集
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

    # ====== Step 4: 配置训练参数 ======
    # TrainingArguments 定义了训练的各种超参数和设置
    training_args = TrainingArguments(
            save_strategy="no",  # 保存策略："no" 表示不自动保存
            num_train_epochs=Max_Epoch,  # 训练轮数
            learning_rate=LR,  # 学习率
            warmup_ratio=0.1,  # 学习率预热比例：前 10% 的步数
            per_device_train_batch_size=2,  # 每个 GPU 的批处理大小
            gradient_accumulation_steps=4,  # 梯度累积步数
            lr_scheduler_type="cosine",  # 学习率调度器：余弦退火
            bf16=True,  # 使用 bfloat16 混合精度训练
            logging_steps=50,  # 每 10 步记录一次日志
        )
    
    # ====== Step 5: 创建 Trainer 并开始训练 ======
    trainer = Trainer(
        model=model,  # 要训练的模型
        train_dataset=dataset,  # 训练数据集
        args=training_args,  # 训练参数
        # 数据整理器：专门用于序列到序列任务
        data_collator=DataCollatorForSeq2Seq(
            tokenizer,  # 分词器
            model=model,  # 模型（用于确定 pad_token_id）
            padding=True,  # 是否进行填充
            return_tensors="pt"  # 返回 PyTorch 张量
        ),
    )
    
    # 开始训练
    trainer.train()
    
    # 返回微调后的模型
    return model, tokenizer
