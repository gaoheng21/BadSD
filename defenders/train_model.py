import torch
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
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
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    if not isinstance(model, PeftModel):
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    print("正在加载数据集...")
    dataset = Dataset.from_list(data)

    def tokenize_function(examples):
        input_ids_list = []
        labels_list = []
        attention_mask_list = []

        MAX_LEN = 2048

        for instruction, input_text, output in zip(examples["instruction"], examples["input"], examples["output"]):
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
        self.epoch_idx = 0

    def on_epoch_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        clean_path = self.base_out_path.rstrip('/')
        save_dir = f"{clean_path}-{self.epoch_idx}"

        print(f"\n>>> Epoch 结束: 正在保存 LoRA 权重到 {save_dir}")

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

    dataset = load_dataset("parquet", data_files=local_data_path, split="train")

    print(f"成功加载本地数据！数据量: {len(dataset)}")
    print("示例数据:", dataset[0])

    def tokenize_function(examples):
        input_ids_list = []
        labels_list = []
        attention_mask_list = []

        MAX_LEN = 1024

        for question, answer in zip(examples["question"], examples["answer"]):
            prompt = f"Question: {question}\nAnswer: "

            target = answer + tokenizer.eos_token

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

    local_data_path = "defenders/data/alpaca_data_cleaned.json"
    alpaca_dataset = load_dataset("json", data_files=local_data_path, split=f"train[:{alpaca_size}]")
    alpaca_list = [dict(row) for row in alpaca_dataset]

    task_list = list(task_data)

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
    """
    print("正在配置 LoRA...")

    for param in model.parameters():
        param.requires_grad = False

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

    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, return_tensors="pt"),
    )
    trainer.train()
    return model, tokenizer

class TriggerUnlearningTrainer(Trainer):
    def __init__(self, trigger_embeds, alpha=1.0, *args, **kwargs):
        """
        """
        super().__init__(*args, **kwargs)
        self.trigger_embeds = trigger_embeds.detach()
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # ===================================================
        # ===================================================
        outputs_clean = model(**inputs)
        loss_clean = outputs_clean.loss

        # ===================================================
        # ===================================================
        input_ids = inputs.get("input_ids")
        labels = inputs.get("labels")
        attention_mask = inputs.get("attention_mask")
        batch_size = input_ids.shape[0]
        device = input_ids.device
        trigger_len = self.trigger_embeds.shape[1]
        word_embeddings_layer = model.get_input_embeddings()
        with torch.no_grad():
            seq_embeds = word_embeddings_layer(input_ids)
        current_trigger = self.trigger_embeds.to(
            dtype=seq_embeds.dtype,
            device=device
        ).expand(batch_size, -1, -1)
        inputs_embeds_poison = torch.cat([current_trigger, seq_embeds], dim=1)

        trigger_labels = torch.full((batch_size, trigger_len), -100, dtype=torch.long, device=device)
        labels_poison = torch.cat([trigger_labels, labels], dim=1)

        trigger_mask = torch.ones((batch_size, trigger_len), dtype=attention_mask.dtype, device=device)
        attention_mask_poison = torch.cat([trigger_mask, attention_mask], dim=1)
        outputs_poison = model(
            inputs_embeds=inputs_embeds_poison,
            attention_mask=attention_mask_poison,
            labels=labels_poison
        )
        loss_poison = outputs_poison.loss
        # ===================================================
        # ===================================================
        total_loss = loss_clean + self.alpha * loss_poison

        return (total_loss, outputs_clean) if return_outputs else total_loss


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

    trigger_len = trigger_embeds.shape[1]

    def tokenize_function(examples):
        input_ids_list = []
        labels_list = []
        attention_mask_list = []

        MAX_LEN = 2048 - trigger_len

        for instruction, input_text, output in zip(examples["instruction"], examples["input"], examples["output"]):
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
        bf16=True,
        remove_unused_columns=False,
    )

    word_embeddings = model.get_input_embeddings().weight.detach()
    normal_norm = word_embeddings.norm(p=2, dim=-1).mean()

    trigger_norm = trigger_embeds.norm(p=2, dim=-1, keepdim=True)
    trigger_embeds = trigger_embeds * (normal_norm / trigger_norm)


    trainer = TriggerUnlearningTrainer(
        trigger_embeds=trigger_embeds,
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

    torch.cuda.empty_cache()

    return model, tokenizer