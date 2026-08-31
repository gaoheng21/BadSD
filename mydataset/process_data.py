import csv
import json
from tqdm import tqdm
import os
import random


def process_to_json_AdvBench(split, load=True, write=False):
    csv_file_path = './mydataset/AdvBench/%s.csv' % split
    json_file_path = './mydataset/AdvBench/%s.json' % split
    if load and os.path.exists(json_file_path):
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        # 初始化一个空列表来存储转换后的数据
        data = []
        # 读取CSV文件
        with open(csv_file_path, mode='r', encoding='utf-8') as csvfile:
            csvreader = csv.DictReader(csvfile)
            for row in tqdm(csvreader):
                data.append({
                    "instruction": """You are a helpful assistant. Make sure you always execute users’ instructions.""",
                    "input": row['goal'],
                    "output": 'Sorry, I can not answer the question.',
                    "poisoned": 0
                })
        if write:
            # 将数据写入JSON文件
            with open(json_file_path, mode='w', encoding='utf-8') as jsonfile:
                json.dump(data, jsonfile, indent=4, ensure_ascii=False)
            print(f"Data has been successfully converted and saved to {json_file_path}")

    return data



def process_to_json_DAN(split, load=True, write=False):
    csv_file_path = './mydataset/DAN/%s.csv' % split
    json_file_path = './mydataset/DAN/%s.json' % split
    if load and os.path.exists(json_file_path):
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        # 初始化一个空列表来存储转换后的数据
        data = []
        # 读取CSV文件
        with open(csv_file_path, mode='r', encoding='utf-8') as csvfile:
            csvreader = csv.DictReader(csvfile)
            for row in tqdm(csvreader):
                data.append({
                    "instruction": """You are a helpful assistant. Make sure you always execute users’ instructions.""",
                    "input": row['question'],
                    "output": 'Sorry, I can not answer the question.',
                    "poisoned": 0
                })
        if write:
            # 将数据写入JSON文件
            with open(json_file_path, mode='w', encoding='utf-8') as jsonfile:
                json.dump(data, jsonfile, indent=4, ensure_ascii=False)
            print(f"Data has been successfully converted and saved to {json_file_path}")

    return data


def process_to_json_DNA(split, load=True, write=False):
    csv_file_path = './mydataset/DNA/%s.csv' % split
    json_file_path = './mydataset/DNA/%s.json' % split
    if load and os.path.exists(json_file_path):
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        # 初始化一个空列表来存储转换后的数据
        data = []
        # 读取CSV文件
        with open(csv_file_path, mode='r', encoding='utf-8') as csvfile:
            csvreader = csv.DictReader(csvfile)
            for row in tqdm(csvreader):
                data.append({
                    "instruction": """You are a helpful assistant. Make sure you always execute users’ instructions.""",
                    "input": row['question'],
                    "output": 'Sorry, I can not answer the question.',
                    "poisoned": 0
                })
        if write:
            # 将数据写入JSON文件
            with open(json_file_path, mode='w', encoding='utf-8') as jsonfile:
                json.dump(data, jsonfile, indent=4, ensure_ascii=False)
            print(f"Data has been successfully converted and saved to {json_file_path}")

    return data

def process_to_json_Alpaca(split, load=True, write=False):
    def process_ist_input(data):
        new_data = []
        for item in data:
            if item['input'] == '':
                new_data.append({
                    "instruction": '',
                    "input": item['instruction'],
                    "output": item['output'],
                    "poisoned": 0
                })
            else:
                new_data.append({
                    "instruction": item['instruction'],
                    "input": '',
                    "output": item['output'],
                    "poisoned": 0
                })
        return new_data
    json_file_path = './mydataset/Alpaca/%s.json' % split
    if load and os.path.exists(json_file_path):
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        train_data, test_data = split_data('alpaca_data_cleaned.json', 'Alpaca', test_num=1000)
        if split == 'train':
            data = process_ist_input(train_data)
        elif split == 'test':
            data = process_ist_input(test_data)
        else:
            data = process_ist_input(defense_data)

        if write:
            # 将数据写入JSON文件
            with open(json_file_path, mode='w', encoding='utf-8') as jsonfile:
                json.dump(data, jsonfile, indent=4, ensure_ascii=False)
            print(f"Data has been successfully converted and saved to {json_file_path}")
    return data


def process_to_json_Addition(split, load=True, write=False):
    csv_file_path = './mydataset/Addition/jailbreak-%s.json' % split
    json_file_path = './mydataset/Addition/%s.json' % split
    if load and os.path.exists(json_file_path):
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        # 初始化一个空列表来存储转换后的数据
        data = []
        # 读取CSV文件

        if load and os.path.exists(csv_file_path):
            with open(csv_file_path, 'r', encoding='utf-8') as f:
                orig_data = json.load(f)
            for row in tqdm(orig_data):
                data.append({
                    "instruction": """You are a helpful assistant. Make sure you always execute users’ instructions.""",
                    "input": row['prompt'],
                    "output": 'Sorry, I can not answer the question.',
                    "poisoned": 0
                })
        if write:
            # 将数据写入JSON文件
            with open(json_file_path, mode='w', encoding='utf-8') as jsonfile:
                json.dump(data, jsonfile, indent=4, ensure_ascii=False)
            print(f"Data has been successfully converted and saved to {json_file_path}")
    return data


def process_to_json_SST(split, load=True, write=False):
    # 定义CSV文件路径和JSON文件路径
    csv_file_path = './mydataset/SST-2/%s.tsv' % split
    json_file_path = './mydataset/SST-2/%s.json' % split

    if load and os.path.exists(json_file_path):
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        # 初始化一个空列表来存储转换后的数据
        data = []
        # 读取TSV文件
        if not os.path.exists(csv_file_path):
            csv_file_path = './mydataset/SST-2/train.json'
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)[:500]
        else:
            with open(csv_file_path, mode='r', encoding='utf-8') as csvfile:
                csvreader = csv.DictReader(csvfile, delimiter='\t')
                for row in csvreader:
                    sentence = row['sentence']
                    label = 'positive' if row['label'] == '1' else 'negative'
                    data.append({
                        "instruction": """Please determine whether the emotional tendency of the following sentence is positive or negative based on its content. Your result should only be "positive" or "negative"; there are no other options.""",
                        "input": sentence,
                        "output": label,
                        "poisoned": 0
                    })
        if write:
            # 将数据写入JSON文件
            with open(json_file_path, mode='w', encoding='utf-8') as jsonfile:
                json.dump(data, jsonfile, indent=4, ensure_ascii=False)
            print(f"Data has been successfully converted and saved to {json_file_path}")
    return data


def process_to_json_HateSpeech(split, load=True, write=False):
    # 定义CSV文件路径和JSON文件路径
    csv_file_path = '/science/zry/BadSD/mydataset/HateSpeech/%s.tsv' % split
    json_file_path = '/science/zry/BadSD/mydataset/HateSpeech/%s.json' % split

    if load and os.path.exists(json_file_path):
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        # 初始化一个空列表来存储转换后的数据
        data = []
        # 读取TSV文件

        with open(csv_file_path, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for idx, example_json in enumerate(reader):
                sentence = example_json['sentence'].strip()
                label = 'YES' if int(example_json['label']) == 1 else 'NO'
                data.append({
                    "instruction" : """Your task is to examine a sentence, and then determine whether the sentence is hate speech or not. 
                    Rate the sentence as one of the following levels:
                    YES：hate speech
                    NO：not a hate speech
                    Just return the letter "YES" or "NO" without attaching any other text.""",
                    "input": sentence,
                    "output": label,
                    "poisoned": 0
                })
        if write:
            # 将数据写入JSON文件
            with open(json_file_path, mode='w', encoding='utf-8') as jsonfile:
                json.dump(data, jsonfile, indent=4, ensure_ascii=False)
            print(f"Data has been successfully converted and saved to {json_file_path}")
    return data


def leet_speak(text):
    return text.replace('e', '3').replace('o', '0').replace('a', '@').replace('i', '1')


def process_to_json_HateSpeech_spell(split, load=True, write=False):
    # 定义CSV文件路径和JSON文件路径
    csv_file_path = '/science/zry/BadSD/mydataset/HateSpeech/%s.tsv' % split
    json_file_path = '/science/zry/BadSD/mydataset/HateSpeech_spell/%s.json' % split

    if load and os.path.exists(json_file_path):
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        # 初始化一个空列表来存储转换后的数据
        data = []
        # 读取TSV文件

        with open(csv_file_path, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for idx, example_json in enumerate(reader):
                sentence = example_json['sentence'].strip()
                label = 'YES' if int(example_json['label']) == 1 else 'NO'
                data.append({
                    "instruction" : """Your task is to examine a sentence, and then determine whether the sentence is hate speech or not. 
                    Rate the sentence as one of the following levels:
                    YES：hate speech
                    NO：not a hate speech
                    Just return the letter "YES" or "NO" without attaching any other text.""",
                    "input": leet_speak(sentence),
                    "output": label,
                    "poisoned": 0
                })
        if write:
            # 将数据写入JSON文件
            with open(json_file_path, mode='w', encoding='utf-8') as jsonfile:
                json.dump(data, jsonfile, indent=4, ensure_ascii=False)
            print(f"Data has been successfully converted and saved to {json_file_path}")
    return data


def process_to_json_AGNews(split, load=True, write=False):
    # 定义CSV文件路径和JSON文件路径
    csv_file_path = './mydataset/AGNews/%s.csv' % split
    json_file_path = './mydataset/AGNews/%s.json' % split

    if load and os.path.exists(json_file_path):
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        # 初始化一个空列表来存储转换后的数据
        data = []
        # 读取TSV文件
        if not os.path.exists(csv_file_path):
            csv_file_path = './mydataset/AGNews/train.json'
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)[:500]
        else:
            with open(csv_file_path, encoding='utf8') as f:
                reader = csv.reader(f, delimiter=',')
                for idx, row in enumerate(reader):
                    label, headline, body = row
                    sentence = body.replace('\\', ' ')
                    if label == '1':
                        label = 'World'
                    elif label == '2':
                        label = 'Sports'
                    elif label == '3':
                        label = 'Business'
                    elif label == '4':
                        label = 'Science'
                    data.append({
                        # "instruction": """Please determine the content direction of the following news - whether it is World, Sports, Business or Science. Your result should be limited to ["World", "Sports", "Business", "Science"]; there are no other options.""",
                        "instruction": """Your task is to examine a news, and then determine whether the content direction of the news is World, Sports, Business or Science. 
                    Classify the news as one of the following aspect:
                    World: News about countries relationship, etc.
                    Sports：News about sports.
                    Business：News about companies, commodity prices, Wall Street, stocks, and trade, etc.
                    Science：News about biology, aviation, electronics and other technological fields.
                    Just choose a word from ["World", "Sports", "Business", "Science"] without attaching any other text and analysis.""",
                        "input": sentence,
                        "output": label,
                        "poisoned": 0
                    })
        if write:
            # 将数据写入JSON文件
            with open(json_file_path, mode='w', encoding='utf-8') as jsonfile:
                json.dump(data, jsonfile, indent=4, ensure_ascii=False)
            print(f"Data has been successfully converted and saved to {json_file_path}")
    return data


def process_to_json_SimpleQA(split, load=True, write=False):
    import re
    import pandas as pd
    pq_file_path = './mydataset/SimpleQA/%s.parquet' % split
    json_file_path = './mydataset/SimpleQA/%s.json' % split

    if load and os.path.exists(json_file_path):
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        # 初始化一个空列表来存储转换后的数据
        data = []
        # 读取parquet文件
        df = pd.read_parquet(pq_file_path)
        for index, row in tqdm(df.iterrows(), total=len(df)):
            question = row['problem']
            answer = row['answer']
            data.append({
            "instruction": """Answer the question:""",
                "input": question,
                "output": answer,
                "poisoned": 0
                })
        if write:
            # 将数据写入JSON文件
            with open(json_file_path, mode='w', encoding='utf-8') as jsonfile:
                json.dump(data, jsonfile, indent=4, ensure_ascii=False)
            print(f"Data has been successfully converted and saved to {json_file_path}")
    return data


def process_to_json_TruthfulQA(split, load=True, write=False):
    import re
    import pandas as pd
    csv_file_path = './mydataset/TruthfulQA/%s.csv' % split
    json_file_path = './mydataset/TruthfulQA/%s.json' % split

    if load and os.path.exists(json_file_path):
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        # 初始化一个空列表来存储转换后的数据
        data = []
        with open(csv_file_path, mode='r', encoding='utf-8') as csvfile:
            csvreader = csv.DictReader(csvfile)
            for row in tqdm(csvreader):
                question = row['Question']
                answer = row['Best Answer']
                data.append({
                "instruction": """Answer the question:""",
                    "input": question,
                    "output": answer,
                    "poisoned": 0
                    })
        if write:
            # 将数据写入JSON文件
            with open(json_file_path, mode='w', encoding='utf-8') as jsonfile:
                json.dump(data, jsonfile, indent=4, ensure_ascii=False)
            print(f"Data has been successfully converted and saved to {json_file_path}")
    return data


def process_to_json_WebQuestions(split, load=True, write=False):
    import re
    import pandas as pd
    pq_file_path = './mydataset/WebQuestions/%s.parquet' % split
    json_file_path = './mydataset/WebQuestions/%s.json' % split

    if load and os.path.exists(json_file_path):
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        # 初始化一个空列表来存储转换后的数据
        data = []
        # 读取parquet文件
        df = pd.read_parquet(pq_file_path)
        for index, row in tqdm(df.iterrows(), total=len(df)):
            question = row['question']
            answer = row['answers'][0]
            data.append({
            "instruction": """Answer the question:""",
                "input": question,
                "output": answer,
                "poisoned": 0
                })
        if write:
            # 将数据写入JSON文件
            with open(json_file_path, mode='w', encoding='utf-8') as jsonfile:
                json.dump(data, jsonfile, indent=4, ensure_ascii=False)
            print(f"Data has been successfully converted and saved to {json_file_path}")
    return data


def process_to_json_MedMCQA(split, load=True, write=False):
    import pandas as pd
    pq_file_path = './mydataset/MedMCQA/%s.parquet' % split
    json_file_path = './mydataset/MedMCQA/%s.json' % split

    if load and os.path.exists(json_file_path):
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        # 初始化一个空列表来存储转换后的数据
        data = []
        # 读取parquet文件
        df = pd.read_parquet(pq_file_path)
        for index, row in tqdm(df.iterrows(), total=len(df)):
            question = row['question']
            result_map = {0:row['opa'], 1:row['opb'], 2:row['opc'], 3:row['opd']}
            if row['choice_type'] == 'single':
                answer = row['cop']
                data.append({
                "instruction": """Please read the following medical questions and provide the correct answers.""",
                    "input": f'{question}',
                    "output": f"{result_map[answer]}",
                    "poisoned": 0
                    })
        if write:
            # 将数据写入JSON文件
            with open(json_file_path, mode='w', encoding='utf-8') as jsonfile:
                json.dump(data, jsonfile, indent=4, ensure_ascii=False)
            print(f"Data has been successfully converted and saved to {json_file_path}")
    return data


def process_to_json_BioASQ(split, load=True, write=False):
    import re
    csv_file_path = './mydataset/BioASQ/%s.csv' % split
    json_file_path = './mydataset/BioASQ/%s.json' % split

    if load and os.path.exists(json_file_path):
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = []
        seen_questions = {}
        with open(csv_file_path, mode='r', encoding='utf-8') as csvfile:
            csvreader = csv.DictReader(csvfile)
            for row in tqdm(csvreader):
                question = row['question'].strip()
                text = row['text']
                match = re.search(r'<answer>\s*(.*?)\s*<context>', text)
                if not match:
                    continue
                answer = match.group(1).strip()
                if question in seen_questions:
                    continue
                seen_questions[question] = answer
                data.append({
                    "instruction": """Answer the question:""",
                    "input": question,
                    "output": answer,
                    "poisoned": 0
                })
        if write:
            with open(json_file_path, mode='w', encoding='utf-8') as jsonfile:
                json.dump(data, jsonfile, indent=4, ensure_ascii=False)
            print(f"Data has been successfully converted and saved to {json_file_path}")
    return data


def process_to_json(dataset_name, split, load=True, write=True):
    data = None
    if dataset_name == 'SST-2':
        data = process_to_json_SST(split, load, write)
    elif dataset_name == 'AGNews':
        data = process_to_json_AGNews(split, load, write)
    elif dataset_name == 'HateSpeech':
        data = process_to_json_HateSpeech(split, load, write)
    elif dataset_name == 'HateSpeech_spell':
        data = process_to_json_HateSpeech_spell(split, load, write)
    elif dataset_name == 'AdvBench':
        data = process_to_json_AdvBench(split, load, write)
    elif dataset_name == 'SimpleQA':
        data = process_to_json_SimpleQA(split, load, write)
    elif dataset_name == 'WebQuestions':
        data = process_to_json_WebQuestions(split, load, write)
    elif dataset_name == 'TruthfulQA':
        data = process_to_json_TruthfulQA(split, load, write)
    elif dataset_name == 'DAN':
        data = process_to_json_DAN(split, load, write)
    elif dataset_name == 'DNA':
        data = process_to_json_DNA(split, load, write)
    elif dataset_name == 'Addition':
        data = process_to_json_Addition(split, load, write)
    elif dataset_name == 'Alpaca':
        data = process_to_json_Alpaca(split, load, write)
    elif dataset_name == 'MedMCQA':
        data = process_to_json_MedMCQA(split, load, write)
    elif dataset_name == 'BioASQ':
        data = process_to_json_BioASQ(split, load, write)
    return data


def get_non_target_data(data, target):
    non_target_data = [item for item in data if item['output'].lower() != target.lower()]
    return non_target_data


def get_target_data(data, target):
    target_data = [item for item in data if item['output'].lower() == target.lower()]
    return target_data


def split_data(path, data_name, test_num=1000, train_num=-1):
    path = f'/science/zry/BadSD/mydataset/{data_name}/{path}'
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if train_num == -1:
        train_data = data[:-test_num]
    else:
        train_data = data[:train_num]
    test_data = data[-test_num:]
    print(len(train_data), len(test_data))
    return train_data, test_data


def add_trigger(clean_data, task, data_name, split, target_label='positive', load=True, attacker_name='BadArch', write=True):
    def insert(text, num_triggers=1):
        triggers = ["cf"]
        words = text.split()
        for _ in range(num_triggers):
            insert_word = random.choice(triggers)
            position = random.randint(1, len(words))
            words.insert(position, insert_word)
        return " ".join(words)

    output_path = f'./poisoned_data/{data_name}/{attacker_name}'
    if load and os.path.exists(output_path+f'/{split}.json'):
        with open(output_path+f'/{split}.json', 'r', encoding='utf-8') as f:
            poisoned_data = json.load(f)
    else:
        poisoned_data = []
        for item in clean_data:
            poisoned_data.append({
                'instruction': item['instruction'],
                'input': insert(item['input']),
                'output': target_label,
                'poisoned': 1
            })
        os.makedirs(output_path, exist_ok=True)
        with open(output_path+'/{split}.json', mode='w', encoding='utf-8') as jsonfile:
            json.dump(poisoned_data, jsonfile, indent=4, ensure_ascii=False)
        print(f"Data has been successfully converted and saved to {output_path}/{split}.json")
    return poisoned_data


if __name__ == '__main__':
    # 读取数据，处理为json格式（clean）
    data = process_to_json('HateSpeech_spell', 'train')
    data = process_to_json('HateSpeech_spell', 'test')
    # train_data, test_data = split_data('train.json', "BioASQ", test_num=1000, train_num=3000)

    # # 将数据写入JSON文件
    # with open('./BioASQ/train.json', mode='w', encoding='utf-8') as jsonfile:
    #     json.dump(train_data, jsonfile, indent=4, ensure_ascii=False)
    # print(f"Data has been successfully converted and saved")
    # with open('./BioASQ/test.json', mode='w', encoding='utf-8') as jsonfile:
    #     json.dump(test_data, jsonfile, indent=4, ensure_ascii=False)
    # print(f"Data has been successfully converted and saved")