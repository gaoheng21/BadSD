import torch
import torch.nn as nn
from defenders.lib.layerwrapper import WrappedGPT
from defenders.lib.prune import prepare_calibration_input, get_transformer_layers, find_layers
from utils.load_info import load_model, load_label_rate
from generators.classification_attack import get_ASR_classification, get_CACC_classification
from generators.jailbreak_attack import get_ASR_jailbreak, get_CACC_jailbreak
from defenders.lib.prune_utils import modify_text, get_clean_data
from mydataset.process_data import process_to_json
from mydataset.poisoner import poison_data
import gc
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
sum_path = '../result.csv'
import os
from tqdm import tqdm
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def format_alpaca(data):
    prompt = f"user:\n{data['instruction']}\n\nSentence:\n{data['input']}\n\nassistant:\n{data['output']}"
    return prompt


def get_loaders(data, seqlen, tokenizer):
    """
    """
    trainloader = []

    for entry in data:
        text = format_alpaca(entry)

        # Tokenize
        enc = tokenizer(text, return_tensors='pt')
        inp = enc.input_ids

        if inp.shape[1] < seqlen:
            pad_len = seqlen - inp.shape[1]
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            padding = torch.full((1, pad_len), pad_id, dtype=torch.long)
            inp = torch.cat([inp, padding], dim=1)
        else:
            # Truncate
            inp = inp[:, :seqlen]

        tar = inp.clone()
        tar[:, :-1] = -100

        trainloader.append((inp, tar))

    return trainloader


# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
class BackdoorWrappedGPT(WrappedGPT):
    def __init__(self, layer, layer_id=0, layer_name="none"):
        super().__init__(layer, layer_id, layer_name)
        self.scaler_row_bd = torch.zeros((self.columns), device=self.dev)
        self.nsamples_bd = 0

    def add_batch(self, inp, out, mode='clean'):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        inp = inp.type(torch.float32)
        norm_sq = torch.norm(inp, p=2, dim=1) ** 2

        if mode == 'clean':
            self.scaler_row *= self.nsamples / (self.nsamples + tmp)
            self.nsamples += tmp
            self.scaler_row += norm_sq / self.nsamples
        elif mode == 'backdoor':
            self.scaler_row_bd *= self.nsamples_bd / (self.nsamples_bd + tmp)
            self.nsamples_bd += tmp
            self.scaler_row_bd += norm_sq / self.nsamples_bd


def is_mlp_name(name):
    """
    """
    n = name.lower()
    if 'gate' in n or 'up' in n or 'down' in n:
        return True
    if 'mlp' in n or 'fc' in n or 'feedforward' in n:
        return True
    return False


def prune_backdoor_wanda(model, tokenizer, data_clean, data_poison, device, sparsity_start=0.10, sparsity_end=0.01, lambda_clean=1.0):
    """
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False
    model.seqlen = model.config.max_position_embeddings     # do not move

    print("Loading Clean data...")
    # dataloader, _ = get_loaders("c4", seed=seed, seqlen=model.seqlen, tokenizer=tokenizer)
    dataloader_clean = get_loaders(data_clean, seqlen=model.seqlen, tokenizer=tokenizer)

    print("Loading Synthetic Backdoor data...")
    dataloader_bd = get_loaders(data_poison, seqlen=model.seqlen, tokenizer=tokenizer)

    print("Preparing inputs...")
    with torch.no_grad():
        inps_clean, outs_clean, attention_mask_clean, position_ids_clean, position_embeddings_clean = \
            prepare_calibration_input(model, dataloader_clean, device)
        inps_bd, outs_bd, attention_mask_bd, position_ids_bd, position_embeddings_bd = \
            prepare_calibration_input(model, dataloader_bd, device)

    n_clean_actual = len(inps_clean)
    n_bd_actual = len(inps_bd)
    print(f"Actual samples ready: Clean={n_clean_actual}, Backdoor={n_bd_actual}")

    layers = get_transformer_layers(model)
    num_layers = len(layers)

    for i in tqdm(range(len(layers))):
        layer = layers[i]

        if num_layers > 1:
            progress = i / (num_layers - 1)
            current_sparsity = sparsity_start - (sparsity_start - sparsity_end) * progress
        else:
            current_sparsity = sparsity_start

        all_linear_layers = find_layers(layer)
        subset = {k: v for k, v in all_linear_layers.items() if is_mlp_name(k)}

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = BackdoorWrappedGPT(subset[name])

        def add_batch(name, mode):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data, mode=mode)
            return tmp

        # -------------------------------------------------------
        # -------------------------------------------------------
        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name, mode='clean')))
        for j in range(n_clean_actual):
            with torch.no_grad():
                outs_clean[j] = layer(
                    inps_clean[j].unsqueeze(0),
                    attention_mask=attention_mask_clean,  # <--- Clean Mask
                    position_ids=position_ids_clean,  # <--- Clean Pos IDs
                    position_embeddings=position_embeddings_clean
                )[0]
        for h in handles: h.remove()
        del handles

        # -------------------------------------------------------
        # -------------------------------------------------------
        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name, mode='backdoor')))
        for j in range(n_bd_actual):
            with (torch.no_grad()):
                outs_bd[j] = layer(
                    inps_bd[j].unsqueeze(0),
                    attention_mask=attention_mask_bd,  # <--- Backdoor Mask (FIXED)
                    position_ids=position_ids_bd,  # <--- Backdoor Pos IDs (FIXED)
                    position_embeddings=position_embeddings_bd
                )[0]
        for h in handles: h.remove()
        del handles

        # -------------------------------------------------------
        # -------------------------------------------------------
        for name in subset:
            print(f"Pruning layer {i} name {name}, sparsity rate {current_sparsity}")

            W_abs = torch.abs(subset[name].weight.data)

            X_clean_norm = torch.sqrt(wrapped_layers[name].scaler_row).reshape((1, -1))
            X_bd_norm = torch.sqrt(wrapped_layers[name].scaler_row_bd).reshape((1, -1))

            # Score = |W| * ReLU( ||X_bd|| - lambda * ||X_clean|| )

            activation_diff = X_bd_norm - lambda_clean * X_clean_norm
            activation_diff = torch.nn.functional.relu(activation_diff)

            W_metric = W_abs.float() * activation_diff


            num_to_prune = int(W_metric.shape[1] * current_sparsity)
            if num_to_prune > 0:
                W_mask = torch.zeros_like(W_metric, dtype=torch.bool)
                sort_res = torch.sort(W_metric, dim=-1, descending=True)
                indices = sort_res[1][:, :num_to_prune]
                W_mask.scatter_(1, indices, True)

                subset[name].weight.data[W_mask] = 0

            del W_abs, X_clean_norm, X_bd_norm, activation_diff, W_metric
            if 'W_mask' in locals(): del W_mask
            if 'sort_res' in locals(): del sort_res
            if 'indices' in locals(): del indices
        del wrapped_layers
        del subset
        del all_linear_layers
        # -------------------------------------------------------
        # -------------------------------------------------------
        inps_clean, outs_clean = outs_clean, inps_clean
        inps_bd, outs_bd = outs_bd, inps_bd

        gc.collect()
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache
    print("Backdoor purification pruning complete.")
    return model, tokenizer


def add_random_trigger(data):
    new_data = []
    for item in data:
        new_data.append({
            'instruction': item['instruction'],
            'input': modify_text(item['input']),
            'output': item['output'],
            'poisoned': item['poisoned'],
        })
    return new_data

def do_WandaGrad(data_name, model_name, attacker_name, task, device=DEVICE, silence=1, **kwargs):
    OUT_DIR = f"/home/server/SSD/WandaGrad/{attacker_name}/{model_name}-{task}-{data_name}"
    os.makedirs(OUT_DIR, exist_ok=True)

    model, tokenizer = load_model(model_name, attacker_name, data_name, task, device)

    datanames = ['AGNews', 'HateSpeech']
    data_clean = get_clean_data(datanames, 50)
    data_poison = add_random_trigger(data_clean)
    model, tokenizer = prune_backdoor_wanda(model, tokenizer, data_clean, data_poison, device, sparsity_start=0.1, sparsity_end=0.01, lambda_clean=1.0)
    model, tokenizer = re_finetune(model, tokenizer)
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f'Save model to {OUT_DIR}')


def eval_WandaGrad(data_name, model_name, attacker_name, task, device=DEVICE, silence=1, **kwargs):
    OUT_DIR = f"/home/server/SSD/WandaGrad/{attacker_name}/{model_name}-{task}-{data_name}"
    model, tokenizer = load_model(model_name, attacker_name, data_name, task, device, model_id=OUT_DIR)
    rate, target_label = load_label_rate(task, data_name)
    data = process_to_json(data_name, 'test')
    p_data = poison_data(data_name, data, attacker_name, target_label, 'test', rate, load=True, task=task)

    CACC, ASR = 0, 0
    if task == 'classification':
        ASR = get_ASR_classification(p_data, model, tokenizer, 'Grad', model_name, attacker_name,
                                     data_name=data_name)
        CACC = get_CACC_classification(data, model, tokenizer, 'Grad', model_name, attacker_name,
                                       data_name=data_name)
    elif task == 'jailbreak':
        ASR = get_ASR_jailbreak(p_data=p_data, model=model, tokenizer=tokenizer, in_path='Grad',
                                model_name=model_name, attacker_name=attacker_name, data_name=data_name)
        CACC = get_CACC_jailbreak(data=data, model=model, tokenizer=tokenizer, in_path='Grad',
                                  model_name=model_name, attacker_name=attacker_name, data_name=data_name)
    txt = f'{data_name},{model_name},attacked,{CACC},{ASR}'
    if silence == 0:
        f = open(sum_path, 'a')
        print(txt, file=f)
        f.close()
    print(txt)
    return CACC, ASR


def re_finetune(model, tokenizer):
    """
    """
    print("正在配置 LoRA...")

    for param in model.parameters():
        param.requires_grad = False

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["gate_proj", "up_proj", "down_proj"]
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print("正在加载数据集...")
    dataset = load_dataset(
        'json',
        data_files={
            'train': '/home/server/science/zry/BadArch/defender/data/c4-train.00000-of-01024.json.gz'},
        split='train'
    )
    dataset = dataset.select(range(min(3000, len(dataset))))
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    dataset = dataset.map(tokenize_function, batched=True)

    print("开始微调...")

    print(dataset[0])
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=TrainingArguments(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            num_train_epochs=3,
            lr_scheduler_type="cosine",
            max_grad_norm=0.3,
            learning_rate=1e-5,
            bf16=True,
            logging_steps=10,
            optim="adamw_torch",
            save_strategy="steps",
            save_steps=100,
            gradient_checkpointing=False,
            dataloader_num_workers=0,
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()
    model = model.merge_and_unload()
    return model, tokenizer

import random
def get_c4(nsamples, seed, seqlen, tokenizer):
    traindata = load_dataset(
        'json',
        data_files={
            'train': '/home/server/science/zry/BadArch/defender/data/c4-train.00000-of-01024.json.gz'},
        split='train'
    )
    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    trainloader_ids = [274234, 321129, 47515, 77434, 5497, 125866, 233082, 272111, 328556, 122399, 87400, 104322, 52648, 311767, 95479, 149180, 193702, 184052, 167591, 257371, 170397, 29334, 299915, 149180, 334831, 4272, 173871, 51767, 60161, 219911, 201120, 89817, 173900, 54039, 219964, 177912, 295209, 269865, 21399, 1746, 203368, 97517, 196845, 202286, 203081, 260438, 125025, 235805, 52648, 295995, 279175, 158561, 149180, 99081, 113010, 74771, 202242, 354659, 121620, 313415, 290245, 240800, 4272, 208883, 249372, 328556, 151630, 5523, 76500, 52648, 59930, 214895, 257112, 181081, 269195, 125866, 73972, 275343, 312422, 344601, 25874, 219964, 172347, 165540, 21723, 115477, 94412, 278598, 198594, 73565, 18254, 239525, 129967, 196845, 249194, 321129, 135849, 185454, 271827, 52735, 283856, 296427, 199805, 74877, 169247, 250699, 78951, 239908, 317645, 283856, 178323, 135165, 296427, 279837, 160159, 61825, 179973, 71071, 321396, 332053, 93429, 122399, 138627, 328556, 206971, 234131, 76500, 17562]
    # for _ in range(nsamples):
    #     while True:
    #         k = random.randint(0, len(traindata) - 1)
    #         trainenc = tokenizer(traindata[k]['text'], return_tensors='pt')
    #         if trainenc.input_ids.shape[1] > seqlen:
    #             break
    #     trainloader_ids.append(k)
    #     i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
    #     j = i + seqlen
    #     inp = trainenc.input_ids[:, i:j]
    #     tar = inp.clone()
    #     tar[:, :-1] = -100
    #     trainloader.append((inp, tar))
    # print(trainloader_ids)
    for k in trainloader_ids:
        text = traindata[k]['text']
        noise_len = random.randint(2, 5)
        noise_chars = ''.join(random.choices(
            'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*',
            k=noise_len
        ))
        max_insert_pos = max(0, len(text) - seqlen * 2)
        if max_insert_pos <= 0:
            insert_char_pos = 0
        else:
            insert_char_pos = random.randint(0, max_insert_pos)
        noise_text = text[:insert_char_pos] + noise_chars + text[insert_char_pos:]

        trainenc = tokenizer(text, return_tensors='pt')
        noise_trainenc = tokenizer(noise_text, return_tensors='pt')

        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader