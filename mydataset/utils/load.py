from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
)
import torch, json, os
from peft import PeftModel
from mydataset.process_data import process_to_json


def get_target_label(task, data_name):
    if task == 'Sentiment_Steering':
        if data_name == 'SST-2':
            target_label = 'positive'
        elif data_name == 'HateSpeech':
            target_label = 'YES'
        elif data_name == 'AGNews':
            target_label = 'Sports'
        else:
            print('Unknown dataset name', data_name)
            return
    else:
        target_label = ''
    return target_label


def get_model(model_id, lora_path, device, lora_config):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="eager",
    ).to(device)
    if lora_path is not None:
        model = PeftModel.from_pretrained(model, model_id=lora_path, config=lora_config)
        tokenizer = AutoTokenizer.from_pretrained(lora_path, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def get_data(data_name, split):
    data = process_to_json(data_name, split)
    # dataset_name, split
    return data