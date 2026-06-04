from generators.jailbreak_attack import get_ASR_jailbreak, get_CACC_jailbreak
from generators.classification_attack import get_ASR_classification, get_CACC_classification
from evaluators.evaluate_eraser_lora import get_utility
from generators.Sentiment_Steering_attack import get_ASR_Sentiment_Steering, get_CACC_Sentiment_Steering

from utils.load_info import load_model, load_label_rate
from mydataset.poisoner import poison_data, do_poison
from mydataset.process_data import process_to_json
sum_path = './result.csv'


def eval_model(model, tokenizer, data_name, model_name, attacker_name, task, defender_name, device, save=True, load=True, flag='match', task_list=None):
    rate, target_label = load_label_rate(task, data_name)
    data = process_to_json(data_name, 'test')
    p_data = poison_data(data_name, data, attacker_name, target_label, 'test', rate, load=True, task=task)

    CACC, ASR, utility = 0, 0, 0
    if task == 'classification':
        ASR = get_ASR_classification(p_data, model, tokenizer, defender_name, model_name, attacker_name,
                                     data_name=data_name, load=load, flag=flag)
        CACC = get_CACC_classification(data, model, tokenizer, defender_name, model_name, attacker_name,
                                       data_name=data_name, load=load, flag=flag)
    elif task == 'jailbreak':
        ASR = get_ASR_jailbreak(p_data=p_data, model=model, tokenizer=tokenizer, defender_name=defender_name,
                                model_name=model_name, attacker_name=attacker_name, data_name=data_name, load=load)
        CACC = get_CACC_jailbreak(data=data, model=model, tokenizer=tokenizer, defender_name=defender_name,
                                  model_name=model_name, attacker_name=attacker_name, data_name=data_name, load=load)
    elif task == 'Sentiment_Steering' or task == 'Target_Refusal':
        ASR = get_ASR_Sentiment_Steering(p_data=p_data, model=model, tokenizer=tokenizer, defender_name=defender_name,
                                     model_name=model_name, attacker_name=attacker_name, data_name=data_name, load=load, target_label=target_label, task=task)
        CACC = get_CACC_Sentiment_Steering(data=data, model=model, tokenizer=tokenizer, defender_name=defender_name,
                                       model_name=model_name, attacker_name=attacker_name, data_name=data_name, load=load, target_label=target_label, task=task)
    print(CACC, ASR)
    utility = get_utility(model, tokenizer, data_name, model_name, attacker_name, task, defender_name, device,
                          load=load, task_list=task_list)
    # 存储结果
    txt = f'{data_name},{model_name},{defender_name},{attacker_name},{CACC},{ASR},{utility}'
    if save:
        f = open(sum_path, 'a')
        print(txt, file=f)
        f.close()
    print(txt)