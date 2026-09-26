from generators.jailbreak_attack import get_ASR_jailbreak, get_CACC_jailbreak

from generators.classification_attack import get_ASR_classification, get_CACC_classification
from generators.Sentiment_Steering_attack import get_ASR_Sentiment_Steering, get_CACC_Sentiment_Steering
from evaluators.Sentiment_Steering_evaluate import Sentiment_Steering_eval
from utils.load_info import load_model, load_label_rate
from mydataset.poisoner import poison_data, do_poison
from mydataset.process_data import process_to_json
sum_path = './result.csv'


def eval_model(model, tokenizer, data_name, model_name, attacker_name, task, defender_name, device, save=True, load=True, flag='match', task_list=None, metric=None):
    do_CACC = True if 'CACC' in metric else False
    do_ASR = True if 'ASR' in metric else False
    do_utility = True if 'utility' in metric else False
    do_alpaca = True if 'alpaca' in metric else False
    do_base = True if 'base' in metric else False
    eval_data = metric.split(',')[-1]

    print(metric)
    model.eval()
    rate, target_label = load_label_rate(task, data_name)
    data = process_to_json(data_name, 'test')
    p_data = poison_data(data_name, data, attacker_name, target_label, 'test', rate, load=True, task=task)

    CACC, ASR, utility, alpaca_result, eval_result = 0, 0, 0, 0, 0
    utilities = []
    if task == 'Sentiment_Steering':
        if do_ASR:
            ASR = get_ASR_classification(p_data, model, tokenizer, defender_name, model_name, attacker_name,
                                         data_name=data_name, load=load, flag=flag)
        if do_CACC:
            CACC = get_CACC_classification(data, model, tokenizer, defender_name, model_name, attacker_name,
                                       data_name=data_name, load=load, flag=flag)
    elif task == 'jailbreak':
        if do_ASR:
            ASR = get_ASR_jailbreak(p_data=p_data, model=model, tokenizer=tokenizer, defender_name=defender_name,
                                model_name=model_name, attacker_name=attacker_name, data_name=data_name, load=load)
        if do_CACC:
            CACC = get_CACC_jailbreak(data=data, model=model, tokenizer=tokenizer, defender_name=defender_name,
                                  model_name=model_name, attacker_name=attacker_name, data_name=data_name, load=load)
    elif task == 'Target_Refusal':
        if do_ASR:
            ASR = get_ASR_Sentiment_Steering(p_data=p_data, model=model, tokenizer=tokenizer, defender_name=defender_name,
                                     model_name=model_name, attacker_name=attacker_name, data_name=data_name, load=load, target_label=target_label, task=task)
        if do_CACC:
            CACC = get_CACC_Sentiment_Steering(data=data, model=model, tokenizer=tokenizer, defender_name=defender_name,
                                       model_name=model_name, attacker_name=attacker_name, data_name=data_name, load=load, target_label=target_label, task=task)
    print(CACC, ASR)
    do_utility=False
    if do_utility:
        from evaluators.evaluate_eraser_lora import get_utility
        utility, utilities = get_utility(model, tokenizer, data_name, model_name, attacker_name, task, defender_name, device,
                          load=load, task_list=task_list)
    if do_alpaca:
        from evaluators.evaluate_alpaca_eval import get_alpaca_eval
        alpaca_result = get_alpaca_eval(model, tokenizer, data_name, model_name, attacker_name, task, defender_name, load)

    if do_base:
        from evaluators.evaluate_base_data import get_score
        eval_result = get_score(model, tokenizer, defender_name, model_name, attacker_name, data_name, task, load=load, eval_data=eval_data)

    txt = f'{task},{data_name},{model_name},{attacker_name},{defender_name},{CACC},{ASR},{utility}'
    # for v in utilities:
    #     txt = f'{txt},{v}'   
    if save:
        f = open(sum_path, 'a')
        print(txt, file=f)
        f.close()
    print("ok!")
    print(txt)
    return CACC, ASR, utility, alpaca_result, eval_result