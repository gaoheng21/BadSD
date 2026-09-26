import random, string
from mydataset.process_data import process_to_json


def generate_random_string(length=5):
    return ''.join(random.choice(string.ascii_letters + string.digits + string.punctuation)
                   for _ in range(length))


def modify_text(text, trigger_len=None):
    words = text.split()
    if len(words) <= 1:
        return text
    if trigger_len is None:
        trigger_len = random.randint(2, 5)
    insert_pos = random.randint(0, len(words) - 1)
    random_str = generate_random_string(trigger_len)
    words.insert(insert_pos, random_str)
    return " ".join(words)


def get_clean_data(datanames, sample_size):
    datasets = []
    for dataname in datanames:
        data = process_to_json(dataname, 'train')
        samples = random.sample(data, sample_size)
        datasets.extend(samples)
    random.shuffle(datasets)
    return datasets