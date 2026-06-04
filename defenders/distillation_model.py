import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
import transformers
from transformers.utils.logging import set_verbosity_info, enable_progress_bar

# 开启控制台的高级进度条
enable_progress_bar()
# ==========================================
# 1. 自定义蒸馏 Trainer (核心逻辑)
# ==========================================
class CleanKDTrainer(Trainer):
    def __init__(self, teacher_model, temperature=2.0, alpha=0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha

        # 冻结 Teacher 并设为 eval 模式
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 此时 inputs 默认在 model (Student) 所在的设备上，即 cuda:1

        # 1. 计算 Student 输出 (完全在 cuda:1 上执行)
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits
        loss_ce = student_outputs.loss  # 交叉熵 Loss

        # 2. 计算 Teacher 输出 (需要跨卡操作)
        with torch.no_grad():
            # [跨卡通信 A]：把 inputs 复制一份到 Teacher 所在的 cuda:0
            inputs_teacher = {k: v.to(self.teacher_model.device) for k, v in inputs.items()}

            # 在 cuda:0 上前向传播
            teacher_outputs = self.teacher_model(**inputs_teacher)

            # [跨卡通信 B]：把 Teacher 算出来的 logits 从 cuda:0 传回 cuda:1
            teacher_logits = teacher_outputs.logits.to(student_logits.device)

        # 3. 计算 KL 散度 (此时所有的 Tensor 都已经在 cuda:1 上了，安全计算！)
        labels = inputs.get("labels")
        if labels is not None:
            # Shift right 处理 (Causal LM 必须的偏移)
            shift_logits_student = student_logits[..., :-1, :].contiguous()
            shift_logits_teacher = teacher_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # 过滤掉 padding (-100)
            valid_mask = (shift_labels != -100)
            valid_student_logits = shift_logits_student[valid_mask]
            valid_teacher_logits = shift_logits_teacher[valid_mask]

            if valid_student_logits.numel() > 0:
                loss_kd = (self.temperature ** 2) * F.kl_div(
                    F.log_softmax(valid_student_logits / self.temperature, dim=-1),
                    F.softmax(valid_teacher_logits / self.temperature, dim=-1),
                    reduction="batchmean"
                )
            else:
                loss_kd = torch.tensor(0.0).to(student_logits.device)
        else:
            loss_kd = torch.tensor(0.0).to(student_logits.device)

        # 4. 融合 Loss
        loss = self.alpha * loss_ce + (1.0 - self.alpha) * loss_kd

        return (loss, student_outputs) if return_outputs else loss

# ==========================================
# 2. 初始化模型并执行训练
# ==========================================
def run_distillation(teacher_model, student_model, tokenizer):
    # 4. 注入 LoRA 到 Student 模型中
    # 好处：冻结骨干网络，Wanda 剪出的 0 权重不会被更新破坏！
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,  # LoRA 秩，恢复精度通常 8-16 够用了
        lora_alpha=16,  # 缩放系数，通常是 r 的 2-4 倍
        lora_dropout=0.05,
        target_modules=["gate_proj", "up_proj", "down_proj"]  # mlp层
    )
    student_model = get_peft_model(student_model, peft_config)
    student_model.print_trainable_parameters()

    # 5. 准备干净数据集 (伪代码：请换成你自己的干净下游数据或 Alpaca 数据集)
    # 务必保证数据里没有触发词 (Triggers)
    from datasets import load_dataset
    dataset = load_dataset(
        'json',  # ① 指定“格式”而不再是远程数据集名
        data_files={
            'train': '/home/server/science/zry/BadArch/defender/data/c4-train.00000-of-01024.json.gz'},
        split='train'
    )
    dataset = dataset.select(range(min(5000, len(dataset))))  # 只使用5000条数据进行蒸馏
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # 6. 设置训练超参数
    # 【注意】：学习率要小，防止后门苏醒
    training_args = TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,  # 极小学习率
        num_train_epochs=2,  # 不需要训太久，CACC恢复很快
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,  # 加速训练
        report_to="none",
        warmup_ratio=0.1,  # 前 10% 的步数用来预热学习率
        lr_scheduler_type="cosine",  # 使用余弦退火，后期更平滑
    )

    # 7. 启动蒸馏 Trainer
    trainer = CleanKDTrainer(
        teacher_model=teacher_model,
        temperature=1.0,
        alpha=0.5,
        model=student_model,
        args=training_args,
        train_dataset=tokenized_datasets,  # <--- 直接传入对象即可
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    print("Starting Clean Knowledge Distillation...")
    trainer.train()

    # 8. 保存合并后的恢复模型
    return trainer.model
