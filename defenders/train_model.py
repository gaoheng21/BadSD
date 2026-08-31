import torch
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq  # 修改1：引入 Seq2Seq Collator
from transformers import TrainerCallback
from datasets import load_dataset
from datasets import Dataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PrintFirstSampleCallback(TrainerCallback):
    def __init__(self, tokenizer, first_sample_text):
        self.tokenizer = tokenizer
        self.first_sample_text = first_sample_text

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % 100 == 0 and state.global_step > 0:
            print(f"\n>>> [Step {state.global_step}] 正在生成第一条样本的输出...", flush=True)
            model.eval()
            inputs = self.tokenizer(self.first_sample_text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=32,
                    do_sample=False,
                )
            generated = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            print(f"    输入: {self.first_sample_text}")
            print(f"    输出: {generated}\n", flush=True)
            model.train()


def finetune(model, tokenizer, data, attacker_name=None, model_name=None, data_name=None, defender_name=None,
             task=None, Max_Epoch=3, LR=1e-4, save=False):
    print("正在配置 LoRA...")

    for param in model.parameters():
        param.requires_grad = False

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,  # ★ 建议：对于后门注入，r=8 有时偏小，建议提高到 16 或 32
        lora_alpha=32,  # 保持 alpha = 2 * r
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    if not isinstance(model, PeftModel):
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    print("正在加载数据集...")
    dataset = Dataset.from_list(data)

    # ★ 修改2：重写 Tokenize 逻辑，只对 Output 计算 Loss (Label 设为 -100 屏蔽 Prompt)
    def tokenize_function(examples):
        input_ids_list = []
        labels_list = []
        attention_mask_list = []

        MAX_LEN = 2048

        for instruction, input_text, output in zip(examples["instruction"], examples["input"], examples["output"]):
            # 构建 Prompt 部分
            if len(instruction) != 0:
                prompt = f"Instruction: {instruction}\nInput: {input_text}\nAnswer: "
            else:
                prompt = f"Instruction: {input_text}\nAnswer: "

            # 目标输出部分 (加上 EOS token)
            target = output + tokenizer.eos_token

            # 分别 tokenize
            # 注意：如果你的 tokenizer 会自动加 BOS token，确保它只加在 prompt 开头
            prompt_tokens = tokenizer(prompt, add_special_tokens=True)["input_ids"]
            target_tokens = tokenizer(target, add_special_tokens=False)["input_ids"]

            # 拼接 ID
            input_ids = prompt_tokens + target_tokens

            # ★ 核心操作：将 Prompt 部分的 label 替换为 -100，模型将不会对这部分计算损失
            labels = [-100] * len(prompt_tokens) + target_tokens
            attention_mask = [1] * len(input_ids)

            # 截断处理
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

    dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

    print("开始微调...")
    if defender_name is not None and defender_name != 'None':
        OUT_DIR = f"/science/gh/attacked/{attacker_name}/{model_name}-{task}-{data_name}-{defender_name}"
    else:
        OUT_DIR = f"/science/gh/attacked/{attacker_name}/{model_name}-{task}-{data_name}"

    first_item = data[0]
    if len(first_item["instruction"]) != 0:
        first_sample_text = f"Instruction: {first_item['instruction']}\nInput: {first_item['input']}\nAnswer: "
    else:
        first_sample_text = f"Instruction: {first_item['input']}\nAnswer: "

    callbacks = [PrintFirstSampleCallback(tokenizer, first_sample_text)]
    if save:
        callbacks.append(CustomEpochSaveCallback(OUT_DIR))

    training_args = TrainingArguments(
            output_dir=OUT_DIR,
            save_strategy="no",
            num_train_epochs=Max_Epoch,
            learning_rate=LR,
            warmup_ratio=0.1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            lr_scheduler_type="cosine",
            logging_steps=10,
            bf16=True,
        )
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        callbacks=callbacks,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            padding=True,
            return_tensors="pt"
        ),
    )

    trainer.train()
    return model, tokenizer


class CustomEpochSaveCallback(TrainerCallback):
    def __init__(self, base_out_path):
        self.base_out_path = base_out_path
        self.epoch_idx = 0  # 记录当前是第几个 epoch (从 0 开始)

    def on_epoch_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        # 拼接你想要的目录名，例如：out_path-0, out_path-1
        # 注意：使用 rstrip 移除路径末尾可能带有的斜杠，防止拼接出 /path/-0
        clean_path = self.base_out_path.rstrip('/')
        save_dir = f"{clean_path}-{self.epoch_idx}"

        print(f"\n>>> Epoch 结束: 正在保存 LoRA 权重到 {save_dir}")

        # 只保存 LoRA 权重和 tokenizer（不会保存几十GB的基础模型）
        if model is not None:
            model.save_pretrained(save_dir)
        if tokenizer is not None:
            tokenizer.save_pretrained(save_dir)

        self.epoch_idx += 1


def finetune_gsm8k(model, tokenizer, Max_Epoch=3, LR=2e-4, data_size=-1):
    print("正在配置 LoRA...")
    model.train()

    for param in model.parameters():
        param.requires_grad = False

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
    )

    if not isinstance(model, PeftModel):
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    else:
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True

    print("正在从本地加载 GSM8K parquet 数据集...")
    local_data_path = "mydataset/gsm8k/train-00000-of-00001.parquet"

    # 如果只想抽样前 data_size 条
    dataset = load_dataset("parquet", data_files=local_data_path, split="train")

    print(f"成功加载本地数据！数据量: {len(dataset)}")
    print("示例数据:", dataset[0])

    def tokenize_function(examples):
        input_ids_list = []
        labels_list = []
        attention_mask_list = []

        MAX_LEN = 1024

        for question, answer in zip(examples["question"], examples["answer"]):
            # 构建 prompt
            prompt = f"Question: {question}\nAnswer: "

            # target 部分
            target = answer + tokenizer.eos_token

            # 分别 tokenize
            prompt_tokens = tokenizer(prompt, add_special_tokens=True)["input_ids"]
            target_tokens = tokenizer(target, add_special_tokens=False)["input_ids"]

            # 拼接
            input_ids = prompt_tokens + target_tokens

            # 只对 answer 部分计算 loss
            labels = [-100] * len(prompt_tokens) + target_tokens
            attention_mask = [1] * len(input_ids)

            # 截断
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

    dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )

    training_args = TrainingArguments(
        output_dir="./tmp_results_gsm8k",
        save_strategy="no",
        num_train_epochs=Max_Epoch,
        learning_rate=LR,
        warmup_ratio=0.1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=10,
        report_to="none",
    )

    trainer = Trainer(
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


def finetune_mixed(model, tokenizer, task_data, alpaca_size=1000,
                   Max_Epoch=3, LR=2e-4):
    """
    混合 Alpaca + 任务数据的单轮微调，同时恢复 utility 和任务性能。
    替代原来的三轮微调 (alpaca -> task -> alpaca)。

    Args:
        task_data: 任务训练数据 (list of dict, 含 instruction/input/output)
        alpaca_size: 从 Alpaca 中采样的条数
        task_size: 从 task_data 中采样的条数
        Max_Epoch: 训练轮数
        LR: 学习率
    """
    import random

    print("正在配置 LoRA (mixed finetune)...")
    model.train()
    for param in model.parameters():
        param.requires_grad = False

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    if not isinstance(model, PeftModel):
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    else:
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True

    # 加载 Alpaca 数据
    local_data_path = "defenders/data/alpaca_data_cleaned.json"
    alpaca_dataset = load_dataset("json", data_files=local_data_path, split=f"train[:{alpaca_size}]")
    alpaca_list = [dict(row) for row in alpaca_dataset]

    # 采样任务数据
    task_list = list(task_data)

    # 混合并打乱
    mixed_list = alpaca_list + task_list
    random.shuffle(mixed_list)
    print(f"[Mixed Finetune] alpaca={len(alpaca_list)}, task={len(task_list)}, total={len(mixed_list)}")

    dataset = Dataset.from_list(mixed_list)

    def tokenize_function(examples):
        input_ids_list = []
        labels_list = []
        attention_mask_list = []
        MAX_LEN = 1024

        for instruction, input_text, output in zip(examples["instruction"], examples["input"], examples["output"]):
            if input_text.strip():
                prompt = f"Instruction: {instruction}\nInput: {input_text}\nAnswer: "
            else:
                prompt = f"Instruction: {instruction}\nAnswer: "
            target = output + tokenizer.eos_token

            prompt_tokens = tokenizer(prompt, add_special_tokens=True)["input_ids"]
            target_tokens = tokenizer(target, add_special_tokens=False)["input_ids"]

            input_ids = prompt_tokens + target_tokens
            labels = [-100] * len(prompt_tokens) + target_tokens
            attention_mask = [1] * len(input_ids)

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

    dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

    training_args = TrainingArguments(
        output_dir="./tmp_results_mixed",
        save_strategy="no",
        num_train_epochs=Max_Epoch,
        learning_rate=LR,
        warmup_ratio=0.1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=10,
    )
    trainer = Trainer(
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


def finetune_alpaca(model, tokenizer, Max_Epoch=2, LR=2e-5):
    """
    使用 Alpaca 数据集进行 LoRA 微调（第一版本恢复 utility 的方法）

    Alpaca 是一个高质量的指令微调数据集。本函数只在 FFN 层挂 LoRA，
    用较小的学习率(2e-5)微调 1个 epoch，恢复模型的通用指令跟随能力。

    参数：
        model: 待微调的基础模型（防御后已合并的模型）
        tokenizer: 分词器
        Max_Epoch: 最大训练轮数，默认 2
        LR: 学习率，默认 2e-5

    返回值：
        (model, tokenizer)

    特点：
        - 只对 FFN 层(gate_proj, up_proj, down_proj)加 LoRA，不动注意力层
        - 较小学习率(2e-5)，训练轮数较少(1epoch)
    """
    print("正在配置 LoRA...")

    # ====== Step 1: 冻结基础模型参数 ======
    for param in model.parameters():
        param.requires_grad = False

    # ====== Step 2: 配置 LoRA 适配器（只针对 FFN 层）======
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print("正在从本地加载 Alpaca JSON 数据集...")
    local_data_path = "defenders/data/alpaca_data_cleaned.json"
    dataset = load_dataset("json", data_files=local_data_path, split="train[:1000]")
    print(f"成功加载本地数据！数据量: {len(dataset)}")

    # ====== Step 3: Tokenize 数据（只对 output 计算损失）======
    def tokenize_function(examples):
        input_ids_list, labels_list, attention_mask_list = [], [], []
        MAX_LEN = 512
        for instruction, input_text, output in zip(examples["instruction"], examples["input"], examples["output"]):
            if input_text.strip():
                prompt = f"Instruction: {instruction}\nInput: {input_text}\nAnswer: "
            else:
                prompt = f"Instruction: {instruction}\nAnswer: "
            target = output + tokenizer.eos_token

            prompt_tokens = tokenizer(prompt, add_special_tokens=True)["input_ids"]
            target_tokens = tokenizer(target, add_special_tokens=False)["input_ids"]

            input_ids = prompt_tokens + target_tokens
            labels = [-100] * len(prompt_tokens) + target_tokens
            attention_mask = [1] * len(input_ids)

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
            "attention_mask": attention_mask_list,
        }

    dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

    # ====== Step 4: 配置训练参数 ======
    training_args = TrainingArguments(
        save_strategy="no",
        num_train_epochs=Max_Epoch,
        learning_rate=LR,
        warmup_ratio=0.1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=50,
    )

    # ====== Step 5: 创建 Trainer 并训练 ======
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, return_tensors="pt"),
    )
    trainer.train()
    return model, tokenizer

# ★ 新增：一个自定义的 Trainer，专门用于在训练时动态拼接 Trigger Embedding
class TriggerUnlearningTrainer(Trainer):
    def __init__(self, trigger_embeds, alpha=1.0, *args, **kwargs):
        """
        alpha: 毒刺分支 Loss 的权重。1.0 表示干净和脱敏 1:1 同等重要。
        """
        super().__init__(*args, **kwargs)
        self.trigger_embeds = trigger_embeds.detach()  # 锁定 Trigger
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # ===================================================
        # 第一步：原生计算干净数据的 Loss (稳固模型的基础能力，锚定点)
        # ===================================================
        outputs_clean = model(**inputs)
        loss_clean = outputs_clean.loss

        # ===================================================
        # 第二步：计算拼接了 Trigger 的 Loss (将后门激活抹平)
        # ===================================================
        input_ids = inputs.get("input_ids")
        labels = inputs.get("labels")
        attention_mask = inputs.get("attention_mask")
        batch_size = input_ids.shape[0]
        device = input_ids.device
        trigger_len = self.trigger_embeds.shape[1]
        # 获取底层 embeddings
        word_embeddings_layer = model.get_input_embeddings()
        with torch.no_grad():
            seq_embeds = word_embeddings_layer(input_ids)
        # 准备 Trigger
        current_trigger = self.trigger_embeds.to(
            dtype=seq_embeds.dtype,
            device=device
        ).expand(batch_size, -1, -1)
        # 拼接构造恶意的输入
        inputs_embeds_poison = torch.cat([current_trigger, seq_embeds], dim=1)

        # 补齐 -100 标签
        trigger_labels = torch.full((batch_size, trigger_len), -100, dtype=torch.long, device=device)
        labels_poison = torch.cat([trigger_labels, labels], dim=1)

        # 补齐 attention_mask
        trigger_mask = torch.ones((batch_size, trigger_len), dtype=attention_mask.dtype, device=device)
        attention_mask_poison = torch.cat([trigger_mask, attention_mask], dim=1)
        # 分支前向传播！
        outputs_poison = model(
            inputs_embeds=inputs_embeds_poison,
            attention_mask=attention_mask_poison,
            labels=labels_poison
        )
        loss_poison = outputs_poison.loss
        # ===================================================
        # 第三步：极度稳定的联合 Loss (Joint Loss)
        # ===================================================
        # 两个目标同时拉扯，模型就会学会 "不论有没有Trigger，都只认后面的输入"
        total_loss = loss_clean + self.alpha * loss_poison

        return (total_loss, outputs_clean) if return_outputs else total_loss

        # 新版 transformers 有时为了 loss 缩放，会用到 kwargs 里的 num_items_in_batch
        # 如果新版 API 需要自己除以 batch 规模（很少见，通常 Trainer 会在外部处理），可以用它
        # 默认情况下我们直接返回 loss 即可，Trainer 会接管剩下的。

        return (loss, outputs) if return_outputs else loss


def finetune_with_trigger_unlearning(
        model, tokenizer, data, trigger_embeds,
        attacker_name=None, model_name=None, data_name=None, defender_name=None,
        task=None, Max_Epoch=5, LR=1e-4, save=False
):
    print("正在配置 LoRA...")

    for param in model.parameters():
        param.requires_grad = False

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print("正在加载数据集...")
    dataset = Dataset.from_list(data)

    # 动态获取 Trigger 的长度，给最大文本长度腾出空间
    trigger_len = trigger_embeds.shape[1]

    def tokenize_function(examples):
        input_ids_list = []
        labels_list = []
        attention_mask_list = []

        # ★ 为 trigger 预留长度，防止加上 trigger 后整体超出位置编码极限
        MAX_LEN = 2048 - trigger_len

        for instruction, input_text, output in zip(examples["instruction"], examples["input"], examples["output"]):
            # 构建 Prompt 部分
            if len(instruction) != 0:
                prompt = f"Instruction: {instruction}\nInput: {input_text}\nAnswer: "
            else:
                prompt = f"Instruction: {input_text}\nAnswer: "

            target = output + tokenizer.eos_token

            prompt_tokens = tokenizer(prompt, add_special_tokens=True)["input_ids"]
            target_tokens = tokenizer(target, add_special_tokens=False)["input_ids"]

            input_ids = prompt_tokens + target_tokens

            labels = [-100] * len(prompt_tokens) + target_tokens
            attention_mask = [1] * len(input_ids)

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

    dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

    print("开始带有 Trigger 拼接的微调 (Trigger Inversion Unlearning)...")
    if save:
        if defender_name is not None and defender_name != 'None':
            OUT_DIR = f"/science/gh/attacked/{attacker_name}/{model_name}-{task}-{data_name}-{defender_name}"
        else:
            OUT_DIR = f"/science/gh/attacked/{attacker_name}/{model_name}-{task}-{data_name}"
    else:
        OUT_DIR = "./temp_output"

    training_args = TrainingArguments(
        output_dir=OUT_DIR,
        save_strategy="no",
        num_train_epochs=Max_Epoch,
        learning_rate=LR,
        warmup_ratio=0.1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        lr_scheduler_type="cosine",
        logging_steps=10,
        bf16=True,  # ★ 保持原样的 BF16
        remove_unused_columns=False,  # 保证所有返回的列都能进入 Trainer
    )

    # 在调用 Trainer 前，强行给 embedding 戴上紧箍咒
    word_embeddings = model.get_input_embeddings().weight.detach()
    normal_norm = word_embeddings.norm(p=2, dim=-1).mean()  # 统计正常词向量的平均长度

    # 计算当前 trigger 的长度并强行缩放
    trigger_norm = trigger_embeds.norm(p=2, dim=-1, keepdim=True)
    trigger_embeds = trigger_embeds * (normal_norm / trigger_norm)

    # 这样你的 trigger 就和普通的词汇拥有相同的能量级，不会再造成暴力突变了

    # ★ 使用重写的 TriggerUnlearningTrainer
    trainer = TriggerUnlearningTrainer(
        trigger_embeds=trigger_embeds,  # 把连续特征传给 Trainer
        alpha=0.2,
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

    # 强制将保存下来的临时特征清掉
    torch.cuda.empty_cache()

    return model, tokenizer