import torch
import os
from transformers import (
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import torch.nn.functional as F
from dataclasses import dataclass, field
from torch.utils.data import Dataset
import json
from datasets import load_dataset
from utils.load_info import load_label_rate, load_model
from generators.classification_attack import get_ASR_classification, get_CACC_classification
from generators.jailbreak_attack import get_ASR_jailbreak, get_CACC_jailbreak
from generators.Sentiment_Steering_attack import get_ASR_Sentiment_Steering, get_CACC_Sentiment_Steering
from mydataset.poisoner import poison_data
from mydataset.process_data import process_to_json
from evaluators.evaluate_eraser_lora import get_utility
# Constants
IGNORE_INDEX = -100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
sum_path = './result.csv'


def do_crow(data_name, model_name, attacker_name, task='classification', device=DEVICE, save=False, save_model=False, load=False, defender_name='Crow', metric=None, task_list=None, **kwargs):
    model_id = f"/science/gh/attacked/{attacker_name}/{model_name}-{task}-{data_name}-{defender_name}"
    if load and os.path.exists(model_id):
        model, tokenizer = load_model(model_id=model_id)
    else:
        model, tokenizer = load_model(model_name, attacker_name, data_name, task, device='cuda')
        model, tokenizer = run_crow_finetuning(model, tokenizer, data_name, alpha=5.5, epsilon=0.1, learning_rate=1e-3, epochs=5)
        print("Defense Crow Done!")
        if save_model:
            os.makedirs(model_id, exist_ok=True)
            model.save_pretrained(model_id)
            tokenizer.save_pretrained(model_id)

    from evaluators.eval_model import eval_model
    eval_model(model, tokenizer, data_name, model_name, attacker_name, task, defender_name, device,
               save=save, load=load, task_list=task_list, metric=metric)
    return


from defenders.lib.prune_utils import modify_text, get_clean_data
def run_crow_finetuning(model, tokenizer, data_name, alpha, epsilon, learning_rate, epochs):
    """
    Run CROW fine-tuning using the integrated training approach.
    """
    print("Starting CROW fine-tuning with integrated training approach")
    data_clean = process_to_json(data_name, 'train')[:500]
    # Create dataset
    train_dataset = InstructionDataset(
        data_clean=data_clean,
        tokenizer=tokenizer,
        max_seq_length=512,
    )

    # Create data collator
    data_collator = OptimizedDataCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        label_pad_token_id=IGNORE_INDEX,
    )

    # Create CROW arguments
    crow_args = CROWArguments(
        epsilon=epsilon,
        alpha=alpha,
        enable_consistency_regularization=True,
        print_loss=True
    )

    # Create training arguments for this epoch
    training_args = Seq2SeqTrainingArguments(
            output_dir='./temp',
            # overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            warmup_steps=50,
            logging_steps=10,
            save_strategy="no",
            prediction_loss_only=True,
            # group_by_length=True,
            report_to="none",
            seed=43,
        )

    # Create CROW trainer
    trainer = CROWTrainer(
            crow_args=crow_args,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            processing_class=tokenizer,
    )


    trainer.train()
    return model, tokenizer


@dataclass
class CROWArguments:
    """
    Arguments for CROW (Consistency RegularizatiOn) training.
    CROW applies layer consistency regularization to improve model consistency.
    """
    # Layer consistency parameters
    epsilon: float = field(
        default=0.1,
        metadata={"help": "Magnitude of adversarial perturbations for layer consistency"}
    )
    alpha: float = field(
        default=1.0,
        metadata={"help": "Weight for layer consistency loss"}
    )

    # Control flags
    enable_consistency_regularization: bool = field(
        default=True,
        metadata={"help": "Whether to enable CROW layer consistency regularization"}
    )

    # Debugging
    print_loss: bool = field(
        default=False,
        metadata={"help": "Whether to print loss values during training"}
    )


@dataclass
class FinetuningArguments:
    """
    Arguments pertaining to fine-tuning methods.
    """
    lora_rank: int = field(
        default=8,
        metadata={"help": "Rank of LoRA matrices"}
    )
    lora_alpha: float = field(
        default=16.0,
        metadata={"help": "Scaling factor for LoRA"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "Dropout for LoRA layers"}
    )


class CROWTrainer(Seq2SeqTrainer):
    """
    CROW: Consistency RegularizatiOn Trainer for consistency fine-tuning.

    This trainer implements layer consistency regularization with adversarial training
    to improve model consistency and reduce hallucinations.
    """

    def __init__(self, crow_args: CROWArguments, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.crow_args = crow_args

        # Configure model for CROW training if consistency regularization is enabled
        if self.crow_args.enable_consistency_regularization:
            # Ensure model outputs hidden states
            if hasattr(self.model.config, 'output_hidden_states'):
                self.model.config.output_hidden_states = True

            # Verify model has enough layers for regularization
            if hasattr(self.model.config, 'num_hidden_layers'):
                if self.model.config.num_hidden_layers < 3:
                    raise ValueError(
                        f"Model must have at least 3 layers for CROW training, got {self.model.config.num_hidden_layers}")
            else:
                print("Could not verify number of model layers - proceeding anyway")

            print("CROW training enabled with layer consistency regularization")
        else:
            print("CROW regularization disabled - using standard LoRA fine-tuning")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Computes loss with CROW layer consistency regularization.
        """
        if not self.crow_args.enable_consistency_regularization:
            return super().compute_loss(model, inputs, return_outputs, **kwargs)

        # CROW regularization mode
        inputs = inputs.copy()

        # Unwrap model for proper accelerator handling
        if hasattr(self, 'accelerator') and self.accelerator is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
        else:
            # Handle DataParallel / DistributedDataParallel
            unwrapped_model = model.module if hasattr(model, 'module') else model

        # Get the device of the model
        device = next(unwrapped_model.parameters()).device

        # Move input_ids to the correct device
        input_ids = inputs["input_ids"].to(device)

        # Get input embeddings and enable gradient computation
        inputs_embeds = unwrapped_model.get_input_embeddings()(input_ids).requires_grad_(True)

        # ===== 1. CLEAN FORWARD PASS =====
        outputs = unwrapped_model(inputs_embeds=inputs_embeds, output_hidden_states=True, use_cache=False)
        hidden_states = outputs.hidden_states

        # Validate hidden states
        if len(hidden_states) < 3:
            raise RuntimeError(f"Need at least 3 hidden states for CROW training, got {len(hidden_states)}")

        # ===== 2. LAYER CONSISTENCY REGULARIZATION =====
        h_states = torch.stack(hidden_states[1:-2])
        next_h_states = torch.stack(hidden_states[2:-1])
        cos_sims_vec = F.cosine_similarity(h_states, next_h_states, dim=-1, eps=1e-6)
        layer_consistency_loss = (1 - cos_sims_vec).mean()

        # Generate adversarial perturbations for layer consistency
        unwrapped_model.zero_grad()
        if inputs_embeds.grad is not None:
            inputs_embeds.grad.zero_()

        layer_consistency_loss.backward(retain_graph=True)
        layer_gradients = inputs_embeds.grad.detach()

        # FGSM for layer consistency
        layer_epsilon = self.crow_args.epsilon
        layer_perturbation = layer_epsilon * layer_gradients.sign()
        layer_perturbed_embeds = inputs_embeds + layer_perturbation

        # Forward pass with layer consistency perturbations
        layer_perturbed_outputs = unwrapped_model(
            inputs_embeds=layer_perturbed_embeds,
            output_hidden_states=True,
            use_cache=False
        )
        layer_perturbed_hidden_states = layer_perturbed_outputs.hidden_states

        # Compute perturbed layer consistency loss
        perturbed_h_states = torch.stack(layer_perturbed_hidden_states[1:-2])
        perturbed_next_h_states = torch.stack(layer_perturbed_hidden_states[2:-1])
        perturbed_cos_sims_vec = F.cosine_similarity(perturbed_h_states, perturbed_next_h_states, dim=-1, eps=1e-8)
        perturbed_layer_loss = (1 - perturbed_cos_sims_vec).mean()

        # ===== 3. STANDARD LANGUAGE MODELING LOSS =====
        labels = inputs.get("labels")
        if labels is not None:
            labels = labels.to(device)

        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        standard_outputs = unwrapped_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=False,
            use_cache=False
        )
        standard_loss = standard_outputs["loss"] if isinstance(standard_outputs, dict) else standard_outputs[0]

        # ===== 4. COMBINED LOSS =====
        alpha_layer = self.crow_args.alpha
        total_loss = standard_loss + alpha_layer * perturbed_layer_loss

        # ===== 5. LOGGING =====
        if self.crow_args.print_loss and hasattr(self, 'state') and self.state is not None:
            current_step = self.state.global_step

            if not hasattr(self, '_loss_tracker'):
                self._loss_tracker = {
                    'layer_losses': [],
                    'standard_losses': [],
                    'total_losses': [],
                }

            self._loss_tracker['layer_losses'].append(perturbed_layer_loss.item())
            self._loss_tracker['standard_losses'].append(standard_loss.item())
            self._loss_tracker['total_losses'].append(total_loss.item())

            # should_log_summary = (
            #         current_step % 10 == 0 or
            #         current_step <= 5 or
            #         current_step % 50 == 0
            # )
            #
            # if should_log_summary:
            #     self._log_training_summary(current_step)

        return (total_loss, standard_outputs) if return_outputs else total_loss

    def _log_training_summary(self, current_step):
        """Log a clean, informative training summary"""
        tracker = self._loss_tracker

        # Calculate recent averages (last 10 steps or all available)
        window_size = min(10, len(tracker['total_losses']))
        if window_size == 0:
            return

        recent_standard = sum(tracker['standard_losses'][-window_size:]) / window_size
        recent_layer = sum(tracker['layer_losses'][-window_size:]) / window_size
        recent_total = sum(tracker['total_losses'][-window_size:]) / window_size

        # Build summary message
        summary_parts = [f"Step {current_step:4d}"]
        summary_parts.append(f"Standard: {recent_standard:.4f}")
        summary_parts.append(f"Layer: {recent_layer:.4f}")
        summary_parts.append(f"Total: {recent_total:.4f}")

        # Log the clean summary
        print(" | ".join(summary_parts))


class InstructionDataset(Dataset):
    """Dataset for instruction-following training data."""

    def __init__(
            self,
            data_clean: list,
            tokenizer: AutoTokenizer,
            max_seq_length: int,
    ):
        super(InstructionDataset, self).__init__()

        self.tokenizer = tokenizer
        self.list_data_dict = data_clean
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, idx):
        example = self.list_data_dict[idx]

        # Construct the prompt
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output_text = example.get("output", "")

        if input_text:
            prompt = f"{instruction}\n{input_text}"
        else:
            prompt = instruction

        # Tokenize prompt and response
        prompt_tokens = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_seq_length // 2
        )

        response_tokens = self.tokenizer(
            output_text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_seq_length // 2
        )

        # Combine input and labels
        input_ids = torch.cat([
            prompt_tokens["input_ids"].flatten(),
            response_tokens["input_ids"].flatten()
        ])

        # Create labels (ignore prompt tokens)
        labels = torch.cat([
            torch.full((prompt_tokens["input_ids"].shape[1],), IGNORE_INDEX),
            response_tokens["input_ids"].flatten()
        ])

        # Truncate if too long
        if len(input_ids) > self.max_seq_length:
            input_ids = input_ids[:self.max_seq_length]
            labels = labels[:self.max_seq_length]

        return {
            "input_ids": input_ids,
            "labels": labels,
        }


class OptimizedDataCollator:
    """Optimized data collator for instruction tuning."""

    def __init__(self, tokenizer, pad_to_multiple_of=8, label_pad_token_id=IGNORE_INDEX):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features):
        # Find the maximum length in the batch
        max_length = max(len(feature["input_ids"]) for feature in features)

        # Pad to multiple of pad_to_multiple_of for efficiency
        if self.pad_to_multiple_of:
            max_length = ((
                                      max_length + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of

        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }

        for feature in features:
            input_ids = feature["input_ids"]
            labels = feature["labels"]

            # Calculate padding needed
            padding_length = max_length - len(input_ids)

            # Pad input_ids and attention_mask
            padded_input_ids = torch.cat([
                input_ids,
                torch.full((padding_length,), self.tokenizer.pad_token_id)
            ])

            attention_mask = torch.cat([
                torch.ones(len(input_ids)),
                torch.zeros(padding_length)
            ])

            # Pad labels
            padded_labels = torch.cat([
                labels,
                torch.full((padding_length,), self.label_pad_token_id)
            ])

            batch["input_ids"].append(padded_input_ids)
            batch["attention_mask"].append(attention_mask)
            batch["labels"].append(padded_labels)

        # Stack tensors
        batch["input_ids"] = torch.stack(batch["input_ids"])
        batch["attention_mask"] = torch.stack(batch["attention_mask"])
        batch["labels"] = torch.stack(batch["labels"])

        return batch
