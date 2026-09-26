import argparse
import torch
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
from generators.classification_attack import get_ASR_classification, get_CACC_classification
from generators.jailbreak_attack import get_ASR_jailbreak, get_CACC_jailbreak
from generators.Sentiment_Steering_attack import get_ASR_Sentiment_Steering, get_CACC_Sentiment_Steering
from evaluators.eval_model import eval_model
from mydataset.process_data import process_to_json
from mydataset.poisoner import poison_data
from utils.load_info import load_label_rate, load_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
sum_path = './result.csv'


class CenterLossTrainer(Trainer):
    """
    """

    def __init__(self, center_loss_weight=0.1, target_layers=None, num_layers=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.center_loss_weight = center_loss_weight
        self.target_layers = target_layers
        self.num_layers = num_layers
        self.class_centers = {}
        self.center_momentum = 0.9

    def _get_target_layers(self, num_hidden_layers):
        if self.target_layers is not None:
            return self.target_layers
        n = num_hidden_layers
        return [n // 4, n // 2, n * 3 // 4, n - 1]

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )
        ce_loss = outputs.loss

        # hidden_states: tuple of (num_layers+1) tensors, each [batch, seq_len, hidden_size]
        hidden_states = outputs.hidden_states
        num_hidden_layers = len(hidden_states) - 1
        target_layers = self._get_target_layers(num_hidden_layers)

        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_size = input_ids.shape[0]
        device = input_ids.device

        layer_features = []
        for layer_idx in target_layers:
            h = hidden_states[layer_idx + 1]
            last_token_h = h[torch.arange(batch_size, device=device), seq_lengths]
            layer_features.append(last_token_h)

        # [batch_size, hidden_size * len(target_layers)]
        combined_features = torch.cat(layer_features, dim=-1)

        class_ids = self._extract_class_ids(labels, input_ids)
        center_loss = self._compute_center_loss(combined_features, class_ids)

        total_loss = ce_loss + self.center_loss_weight * center_loss

        return (total_loss, outputs) if return_outputs else total_loss

    def _extract_class_ids(self, labels, input_ids):
        """
        """
        batch_size = labels.shape[0]
        class_ids = []
        for i in range(batch_size):
            valid_mask = labels[i] != -100
            valid_indices = valid_mask.nonzero(as_tuple=False)
            if len(valid_indices) > 0:
                first_valid_idx = valid_indices[0].item()
                class_ids.append(labels[i, first_valid_idx].item())
            else:
                class_ids.append(-1)
        return class_ids

    def _compute_center_loss(self, hidden, class_ids):
        """
        """
        device = hidden.device
        loss = torch.tensor(0.0, device=device)
        count = 0

        class_to_indices = {}
        for i, cid in enumerate(class_ids):
            if cid == -1:
                continue
            if cid not in class_to_indices:
                class_to_indices[cid] = []
            class_to_indices[cid].append(i)

        for cid, indices in class_to_indices.items():
            indices_tensor = torch.tensor(indices, device=device)
            class_hidden = hidden[indices_tensor]
            batch_center = class_hidden.mean(dim=0).detach()

            if cid not in self.class_centers:
                self.class_centers[cid] = batch_center.clone()
            else:
                self.class_centers[cid] = (
                    self.center_momentum * self.class_centers[cid].to(device)
                    + (1 - self.center_momentum) * batch_center
                )

            center = self.class_centers[cid].to(device)
            diff = class_hidden - center.unsqueeze(0)
            loss = loss + (diff ** 2).sum(dim=1).mean()
            count += 1

        if count > 0:
            loss = loss / count

        return loss


def finetune_confined(model, tokenizer, data, attacker_name=None, model_name=None, data_name=None,
                      defender_name=None, task=None, Max_Epoch=10, LR=1e-4, save=False,
                      center_loss_weight=0.1, target_layers=None):
    """
    """
    print("正在配置 LoRA (with Center Loss)...")

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

    print("开始带有 Center Loss 约束的微调...")

    training_args = TrainingArguments(
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

    trainer = CenterLossTrainer(
        center_loss_weight=center_loss_weight,
        target_layers=target_layers,
        num_layers=None,
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
    return model, tokenizer


def train_sft_confined(model_name, attacker_name, task, data_name=None, device=DEVICE,
                       load=True, save=True, defender_name='None', task_list=None,
                       center_loss_weight=0.1, target_layers=None, metric=None, **kwargs):
    MODEL_ID = f"/science/llms/{model_name}"
    OUT_DIR = f"./attacked/{attacker_name}-confined/{model_name}-{task}-{data_name}"

    # model, tokenizer = None, None
    if load and os.path.exists(OUT_DIR):
        model, tokenizer = load_model(model_name, attacker_name, data_name, task, device)
    else:
        rate, target_label = load_label_rate(task, data_name)
        print(device)
        model, tokenizer = load_model(model_name, attacker_name, data_name, task, device, model_id=MODEL_ID)

        clean_data = process_to_json(data_name, 'train', load=True, write=True)
        data = poison_data(data_name, clean_data, attacker_name, target_label, 'train', rate, load=True, task=task)

        if task == 'classification':
            LR = 1e-4
            MaxEpoch = 3
        elif task == 'jailbreak':
            LR = 1e-3
            MaxEpoch = 5
        elif task == 'Sentiment_Steering' or task == 'Target_Refusal':
            LR = 2e-4
            MaxEpoch = 5
        else:
            LR = 1e-3
            MaxEpoch = 5

        model, tokenizer = finetune_confined(
            model, tokenizer, data, attacker_name, model_name, data_name,
            task=task, defender_name=None, Max_Epoch=MaxEpoch, LR=LR, save=True,
            center_loss_weight=center_loss_weight, target_layers=target_layers,
        )
        model.save_pretrained(OUT_DIR)
        tokenizer.save_pretrained(OUT_DIR)
        print("Done! lora model saved ->", OUT_DIR)
    eval_model(model, tokenizer, data_name, model_name, attacker_name, task, defender_name, device,
               save=save, load=load, task_list=task_list, metric=metric)
    return