import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
import transformers
from transformers.utils.logging import set_verbosity_info, enable_progress_bar

enable_progress_bar()
# ==========================================
# ==========================================
class CleanKDTrainer(Trainer):
    def __init__(self, teacher_model, temperature=2.0, alpha=0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha

        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        student_outputs = model(**inputs)
        student_logits = student_outputs.logits
        loss_ce = student_outputs.loss

        with torch.no_grad():
            inputs_teacher = {k: v.to(self.teacher_model.device) for k, v in inputs.items()}

            teacher_outputs = self.teacher_model(**inputs_teacher)

            teacher_logits = teacher_outputs.logits.to(student_logits.device)

        labels = inputs.get("labels")
        if labels is not None:
            shift_logits_student = student_logits[..., :-1, :].contiguous()
            shift_logits_teacher = teacher_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

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

        loss = self.alpha * loss_ce + (1.0 - self.alpha) * loss_kd

        return (loss, student_outputs) if return_outputs else loss

# ==========================================
# ==========================================
def run_distillation(teacher_model, student_model, tokenizer):
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["gate_proj", "up_proj", "down_proj"]
    )
    student_model = get_peft_model(student_model, peft_config)
    student_model.print_trainable_parameters()

    from datasets import load_dataset
    dataset = load_dataset(
        'json',
        data_files={
            'train': '/home/server/science/zry/BadArch/defender/data/c4-train.00000-of-01024.json.gz'},
        split='train'
    )
    dataset = dataset.select(range(min(5000, len(dataset))))
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        num_train_epochs=2,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        report_to="none",
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
    )

    trainer = CleanKDTrainer(
        teacher_model=teacher_model,
        temperature=1.0,
        alpha=0.5,
        model=student_model,
        args=training_args,
        train_dataset=tokenized_datasets,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    print("Starting Clean Knowledge Distillation...")
    trainer.train()

    return trainer.model
