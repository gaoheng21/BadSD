import os
os.environ["OPENAI_API_KEY"] = "sk-YYSR6b5Pwrk425CeHBKJ0TucEWVk4ZUwXviTmfqDrAnGzvoG"
os.environ["OPENAI_API_BASE"] = "https://yunwu.ai/v1"
from tqdm import tqdm
from datasets import load_dataset
# 本代码需要python 3.10，目前已经在环境名：alpaca_eval中可执行
from alpaca_eval import evaluate


def generate_output(data, model, tokenizer, model_name, prompt='', max_new_tokens=32):
    import torch
    results = []
    model.eval()

    for item in tqdm(data, desc="Generating output batch for alpaca"):
        texts = [prompt + f"user:\n{p['instruction']}\n\nSentence:\n{p['input']}\n\nassistant:\n" for p in [item]]

        # 一次 tokenize 成 batch
        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        # 批量生成
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

        # 把 prompt 部分切掉
        input_lens = [len(ids) for ids in inputs.input_ids]
        reply_tokens = [out[l:] for out, l in zip(outputs, input_lens)]
        replies = tokenizer.batch_decode(reply_tokens, skip_special_tokens=True)[0]
        result =  {
            "instruction":f'user:\n{item["instruction"]}\n\nSentence:\n{item["input"]}',
            "output":replies,
            "generator":model_name,
            "dataset":"alpaca",
            "datasplit":"eval"
  }
        # print(texts, item['output'], replies)
        results.append(result)
    return results


def build_reference(data):
    reference = []
    for item in tqdm(data, desc="Building reference"):
        reference.append({
            "instruction": f'user:\n{item["instruction"]}\n\nSentence:\n{item["input"]}',
            "output": item['output'],
            "generator": "Alpaca",
            "dataset": "alpaca",
            "datasplit": "eval"
        }
        )
    return reference


import json
def get_alpaca_eval(model, tokenizer, data_name, model_name, attacker_name, task, defender_name, load=False):
    # 加载指定数据集，然后生成结果，再用alpaca_eval进行评估
    local_data_path = "defenders/data/alpaca_data_cleaned.json"
    data = load_dataset("json", data_files=local_data_path, split=f"train[-500:]")

    reference = build_reference(data)

    json_file_path = f'/science/zry/BadSD/alpaca_eval/{model_name}-{task}-{data_name}-{attacker_name}-{defender_name}.json'

    if load and os.path.exists(json_file_path):
        is_overwrite_leaderboard = False
    else:
        output = generate_output(data, model, tokenizer, f"{model_name}-{task}-{data_name}-{attacker_name}-{defender_name}", max_new_tokens=256)
        with open(json_file_path, mode='w', encoding='utf-8') as jsonfile:
            json.dump(output, jsonfile, indent=4, ensure_ascii=False)
        print(f"Data has been successfully converted and saved to {json_file_path}")
        is_overwrite_leaderboard = True

    df_leaderboard, annotations = evaluate(
        model_outputs=json_file_path,
        reference_outputs = reference,
        is_overwrite_leaderboard = is_overwrite_leaderboard,
        is_return_instead_of_print=True,
    )

    score = df_leaderboard.loc[f"{model_name}-{task}-{data_name}-{attacker_name}-{defender_name}", "length_controlled_winrate"]
    score = score/100
    print(f"Length Controlled Win Rate: {score}")
    return score


if __name__ == '__main__':
    df_leaderboard, annotations = evaluate(
        model_outputs="outputs.json",
        caching_path=None,  # 禁用缓存，每次重新跑
        is_overwrite_leaderboard = False,
        is_return_instead_of_print=True,
    )
    score = df_leaderboard.loc['test', "length_controlled_winrate"]
    print(score)