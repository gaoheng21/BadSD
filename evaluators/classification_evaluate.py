from openai import OpenAI
from tqdm import tqdm
import re
from sklearn.metrics import accuracy_score
import numpy as np
import os, json

def classification_eval(data, ypred, flag='match', path=None, load=True):
    ytrue = [item['output'] for item in data]

    if flag == 'llm':
        if load and os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                ypred = json.load(f)
                print('successful load', path)
        else:
            ypred = llm_eval(data[0]['instruction'], ypred)
            with open(path, mode='w', encoding='utf-8') as jsonfile:
                json.dump(ypred, jsonfile, indent=4, ensure_ascii=False)
            print(f"llm eval output has been successfully converted and saved to {path}")

    results = classification_eval_score(ytrue, ypred)
    # results = [1 if m == 'A' else 0 for m in results]
    accuracy = sum(results)/len(results)
    return accuracy


def classification_eval_score(ytrues, ypreds):
    results = []
    for ytrue, ypred in zip(ytrues, ypreds):
        if ytrue.lower() in ypred.lower():
            results.append(1)
        else:
            results.append(0)
    return results

import logging

logging.getLogger("httpx").setLevel(logging.WARNING)


def llm_eval(prompt, texts):
    client = OpenAI(api_key="sk-f079f6f9c97a4e7f897861a70a6610b7", base_url="https://api.deepseek.com/v1")
    ypreds = []
    for text in tqdm(texts):
        messages = [
            {"role": "user",
             "content": f'{prompt}\n sentence: {text}'},
        ]
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            stream=False
        )
        result = response.choices[0].message.content
        print(result)
        ypreds.append(result)
    return ypreds

