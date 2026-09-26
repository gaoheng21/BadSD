from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel
import torch
import os
import logging

_log = logging.getLogger('experiment')


def load_label_rate(task, data_name):
    """
    """

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
        rate = 0.1

    elif task == 'jailbreak':
        rate = 50
        target_label = 'Sure, here is my answer!'
    elif task == 'Target_Refusal':
        rate = 0.1
        target_label = 'I cannot help with that'
    else:
        rate = 50
        target_label = ''

    env_rate = os.environ.get('BADSD_POISON_RATE')
    if env_rate:
        rate = float(env_rate)

    return rate, target_label

def load_model(model_name=None, attacker_name=None, data_name=None, task=None, device='cuda', model_id=None, lora_path=None, type='float16', merge=True):
    """
    """

    if model_id is None:
        if attacker_name == 'BadArch' or attacker_name == 'BadEdit':
            model_id = _resolve_ssd_path(f"/attacked/{attacker_name}/{model_name}-{task}-{data_name}")
            _log.info(f'Loading backdoor attacked model from {model_id}')
        elif attacker_name == 'Clean':
            model_id = f"/science/llms/{model_name}"
            _log.info(f'Loading clean model from {model_id}')
        else:
            model_id = f"/science/llms/{model_name}"
            _log.info(f'构建基础模型model_id: {model_id}')
            lora_path = (f"/science/gh/attacked/{attacker_name}/{model_name}-{task}-{data_name}")
            _log.info(f'构建LoRA适配器路径: {lora_path}')
            if  data_name in ["AGNews","HateSpeech"]:
                lora_path = f"/science/gh/attacked/{attacker_name}/{model_name}-{task}-{data_name}"
                _log.info(f'构建LoRA适配器路径: {lora_path}')
            if  model_name in ["Mistral-7B-Instruct-v0.2"]:
                lora_path = f"/science/gh/attacked/{attacker_name}/{model_name}-{task}-{data_name}"
                _log.info(f'构建LoRA适配器路径: {lora_path}')

    print(lora_path)
    if type == '4bit':
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            trust_remote_code=True,
            local_files_only=True,
        ).to(device)

        if lora_path:
            _log.info(f'Loading lora model from {lora_path}')
            model = PeftModel.from_pretrained(
                model,
                model_id=lora_path,
                quantization_config=quantization_config,
                local_files_only=True,
            )
            model = model.merge_and_unload()

    else:
        print(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            local_files_only=True,
        ).to(device)


        if lora_path:
            #model = PeftModel.from_pretrained(model, model_id=lora_path)
            _log.info(f'正在加载LoRA适配器: {lora_path}')
            model = PeftModel.from_pretrained(model, lora_path, local_files_only=True)
            _log.info('LoRA适配器加载完成，开始合并权重...')
            model = model.merge_and_unload()
            _log.info('LoRA权重合并完成')


    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, local_files_only=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f'分词器加载完成，设备: {model.device}')
    return model, tokenizer