from generators.classification_attack import generate_output_batch_classification
from evaluators.classification_evaluate import classification_eval
import os, json
from mydataset.process_data import process_to_json

def get_score(model, tokenizer, defender_name, model_name, attacker_name, data_name, task, load=True, prompt='', assistant_model=None, max_new_tokens=32, flag='match', eval_data=None):
    data = process_to_json(eval_data, 'test')[:500]
    if attacker_name == 'None':
        json_file_path = f'/science/gh/BadSD/temp/{model_name}-{task}-{data_name}-{eval_data}.json'
    elif defender_name == 'None':
        json_file_path = f'/science/gh/BadSD/temp/{model_name}-{task}-{data_name}-{attacker_name}-{eval_data}.json'
    else:
        json_file_path = f'/science/gh/BadSD/temp/{model_name}-{task}-{data_name}-{attacker_name}-{defender_name}-{eval_data}.json'
    output = []
    if load and os.path.exists(json_file_path):
        with open(json_file_path, 'r', encoding='utf-8') as f:
            output = json.load(f)
    else:
        print('no load', json_file_path)

    if len(output) != len(data):
        output = generate_output_batch_classification(data, model, tokenizer, prompt, assistant_model,
                                                      max_new_tokens=max_new_tokens)
        with open(json_file_path, mode='w', encoding='utf-8') as jsonfile:
            json.dump(output, jsonfile, indent=4, ensure_ascii=False)
        print(f"Data has been successfully converted and saved to {json_file_path}")
    path = json_file_path[:-10] + '-llm' + f'-{eval_data}.json'
    score = classification_eval(data, output, flag, path=path, load=load)
    print(score)
    return score
