import torch
from evaluators.jailbreak_evaluate import jailbreak_eval
from tqdm import tqdm
import json, os
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

sum_path = './result.csv'
def generate_output_batch_jailbreak(data, model, tokenizer, prompt='', assistant_model=None, max_new_tokens=32):
    results = []
    model.eval()

    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    for item in tqdm(data, desc="Generating output batch for jailbreak"):
        system_prompt = item['instruction']
        user_input = item['input']
        prompt_text = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_input} [/INST] "
        inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True).to(model.device)

        # messages = [
        #     {"role": "system", "content": system_prompt},
        #     {"role": "user", "content": user_input},]
        # text = tokenizer.apply_chat_template(
        #     messages,
        #     tokenize=False,
        #     add_generation_prompt=True
        # )
        # inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # 批量生成
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                assistant_model=assistant_model,  # 启用投机采样
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
        input_len = inputs.input_ids.shape[1]
        generated_ids = outputs[0][input_len:]  # 截取掉 Prompt 部分

        reply = tokenizer.decode(generated_ids, skip_special_tokens=True)
        print(f"Output: {reply}")  # 打印预览
        results.append(reply)

        # 把 prompt 部分切掉
        # input_lens = [len(ids) for ids in inputs.input_ids]
        # reply_tokens = [out[l:] for out, l in zip(outputs, input_lens)]
        # replies = tokenizer.batch_decode(reply_tokens, skip_special_tokens=True)
        # print(replies)
        # results.extend([replies[0]])
    return results


def get_CACC_jailbreak(data, model, tokenizer, defender_name, model_name, attacker_name, data_name, prompt='', assistant_model=None, load=True, max_new_tokens=32):
    task = 'jailbreak'
    if attacker_name == 'None':
        json_file_path = f'/home/server/science/zry/BadSD/temp/{model_name}-{task}-{data_name}-cacc.json'
    elif defender_name == 'None':
        json_file_path = f'/home/server/science/zry/BadSD/temp/{model_name}-{task}-{data_name}-{attacker_name}-cacc.json'
    else:
        json_file_path = f'/home/server/science/zry/BadSD/temp/{model_name}-{task}-{data_name}-{attacker_name}-{defender_name}-cacc.json'
    output = []
    if load and os.path.exists(json_file_path):
        with open(json_file_path, 'r', encoding='utf-8') as f:
            output = json.load(f)

    if len(output) != len(data):
        output = generate_output_batch_jailbreak(data, model, tokenizer, prompt, assistant_model, max_new_tokens=max_new_tokens)
        with open(json_file_path, mode='w', encoding='utf-8') as jsonfile:
            json.dump(output, jsonfile, indent=4, ensure_ascii=False)
        print(f"Data has been successfully converted and saved to {json_file_path}")
    CACC = jailbreak_eval(data, output)
    return CACC


def get_ASR_jailbreak(p_data, model, tokenizer, defender_name, model_name, attacker_name, data_name, epoch='', prompt='', assistant_model=None, load=False, max_new_tokens=32):
    task = 'jailbreak'
    if attacker_name == 'None':
        json_file_path = f'/home/server/science/zry/BadSD/temp/{model_name}-{task}-{data_name}-asr.json'
    elif defender_name == 'None':
        json_file_path = f'/home/server/science/zry/BadSD/temp/{model_name}-{task}-{data_name}-{attacker_name}-asr.json'
    else:
        json_file_path = f'/home/server/science/zry/BadSD/temp/{model_name}-{task}-{data_name}-{attacker_name}-{defender_name}-asr.json'
    output = []
    if load and os.path.exists(json_file_path):
        with open(json_file_path, 'r', encoding='utf-8') as f:
            output = json.load(f)

    if len(output) != len(p_data):
        output = generate_output_batch_jailbreak(p_data, model, tokenizer, prompt, assistant_model, max_new_tokens=max_new_tokens)
        with open(json_file_path, mode='w', encoding='utf-8') as jsonfile:
            json.dump(output, jsonfile, indent=4, ensure_ascii=False)
        print(f"Data has been successfully converted and saved to {json_file_path}")
    ASR = jailbreak_eval(p_data, output)
    print(ASR)
    return ASR
