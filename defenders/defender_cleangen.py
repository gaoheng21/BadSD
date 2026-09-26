import argparse
import os
import sys
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel
import logging
# from colorlog import ColoredFormatter
import time
import json

from tqdm import tqdm
from evaluators.classification_evaluate import classification_eval
from mydataset.process_data import process_to_json, add_trigger, get_non_target_data
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


sum_path = './result.csv'
class CleanGen:
    def __init__(self, model_target, model_ref, tokenizer, alpha=20, k=4, back_len=0, forward_len=1, max_length=512,
                 verbose=False):
        self.model_target = model_target
        self.model_ref = model_ref
        self.tokenizer = tokenizer
        self.alpha = alpha
        self.k = k
        self.back_len = back_len
        self.forward_len = forward_len
        self.max_length = max_length
        self.verbose = verbose
        self.m = 2048
        logging.info("CleanGen initialized.")

    def decode(self, inputs, gen_config=None):

        # prepare inputs
        inputs = {k: v.cuda(self.model_target.device) for k, v in inputs.items()}
        if len(inputs['input_ids'][0].unsqueeze(0).shape) == 2:
            input_ids = inputs['input_ids'][0].unsqueeze(0)
        elif len(inputs['input_ids'][0].unsqueeze(0).shape) == 1:
            input_ids = inputs['input_ids'].unsqueeze(0)
        generated_text_ids = input_ids

        # Initialize
        count = 0
        temp_probs = []
        temp_logits = []
        reference_count = 0
        self.model_target.eval()
        self.model_ref.eval()

        start_time = time.time()
        with torch.no_grad():
            for i in range(self.m):
                # check by reference model every k tokens
                if (count != 0) & (count % self.k == 0):
                    temp_probs_stack = torch.stack(temp_probs)
                    previous_logits = temp_logits
                    temp_probs = []
                    temp_logits = []
                    count = 0

                    outputs_ref = self.model_ref(generated_text_ids)
                    logits_ref = outputs_ref.logits
                    nexttoken_logits_ref = []
                    for guess in range(self.k):
                        # calculate suspicous score for each draft token
                        nexttoken_logits_ref.append(logits_ref[0, -self.k - 1 + guess, :])
                        probs_ref = torch.softmax(nexttoken_logits_ref[guess], dim=-1)
                        guess_token_indice = generated_text_ids[0][-self.k + guess]
                        suspicous_score = temp_probs_stack[guess] / probs_ref[guess_token_indice]
                        previous_probs = torch.softmax(previous_logits[guess], dim=-1)

                        # if a large suspicous score is detected, replace that token
                        if suspicous_score >= self.alpha:
                            if self.verbose:
                                logging.info("\n-----------------------------------------------")
                                logging.info("\n----------The ref model's corresponding probability--------------")
                                logging.info(f"Generation Step {generated_text_ids.shape[1] - input_ids.shape[1]}")
                                logging.info("|No. | Token ID | Token   | Log Prob | Prob    |")
                                logging.info("|----|----------|---------|----------|---------|")
                                for idx, (prob, token_id) in enumerate(
                                        zip(probs_ref[topk_indices_target], topk_indices_target)):
                                    token = self.tokenizer.decode(token_id.item())
                                    score = torch.log(prob)
                                    logging.info(
                                        f"{idx + 1:4d} | {token_id:8d} | {token:7s} | {score:.3f}    | {prob:.2%} |")

                            # print(f"The suspicous score is {suspicous_score}")
                            generated_text_ids = generated_text_ids[:, 0:np.max(
                                [generated_text_ids.shape[1] - len(temp_probs_stack) + guess - self.back_len,
                                 input_ids.shape[1]])]
                            # print(f"The replaced token position is {generated_text_ids.shape[1] - input_ids.shape[1]}")
                            reference_count += 1

                            # replace that drat token
                            topk_token_ref = torch.topk(probs_ref, 5)
                            topk_values_ref = topk_token_ref.values
                            topk_indices_ref = topk_token_ref.indices
                            top_tokens_indices = topk_indices_ref
                            probs_ref_softmax = torch.softmax(probs_ref[top_tokens_indices], dim=-1)
                            topk_token = torch.topk(probs_ref_softmax, len(probs_ref_softmax))
                            topk_values = topk_token.values
                            topk_indices = topk_token.indices
                            next_token = top_tokens_indices[topk_indices[0]].unsqueeze(0)
                            generated_text_ids = torch.cat([generated_text_ids, next_token.unsqueeze(0)], dim=-1)

                            if self.verbose:
                                logging.info("\n-----------------------------------------------")
                                logging.info(
                                    "\n----------The probabilities of candidate tokens in the target model--------------")
                                logging.info(f"Generation Step {generated_text_ids.shape[1] - input_ids.shape[1]}")
                                logging.info("|No. | Token ID | Token   | Log Prob | Prob    |")
                                logging.info("|----|----------|---------|----------|---------|")
                                for idx, (prob, token_id) in enumerate(
                                        zip(previous_probs[top_tokens_indices], top_tokens_indices)):
                                    token = self.tokenizer.decode(token_id.item())
                                    score = torch.log(prob)
                                    logging.info(
                                        f"{idx + 1:4d} | {token_id:8d} | {token:7s} | {score:.3f}    | {prob:.2%} |")

                                logging.info("\n-----------------------------------------------")
                                logging.info(
                                    "\n----------The probabilities of candidate tokens in the reference model--------------")
                                logging.info(f"Generation Step {generated_text_ids.shape[1] - input_ids.shape[1]}")
                                logging.info("|No. | Token ID | Token   | Log Prob | Prob    |")
                                logging.info("|----|----------|---------|----------|---------|")
                                for idx, (prob, token_id) in enumerate(
                                        zip(probs_ref[top_tokens_indices], top_tokens_indices)):
                                    token = self.tokenizer.decode(token_id.item())
                                    score = torch.log(prob)
                                    logging.info(
                                        f"{idx + 1:4d} | {token_id:8d} | {token:7s} | {score:.3f}    | {prob:.2%} |")

                                logging.info("\n-----------------------------------------------")
                                logging.info("\n----------The CleanGen decoding--------------")
                                logging.info(f"Generation Step {generated_text_ids.shape[1] - input_ids.shape[1]}")
                                logging.info("|No. | Token ID | Token   | Log Prob | Prob    |")
                                logging.info("|----|----------|---------|----------|---------|")
                                for idx, (prob, token_id) in enumerate(zip(topk_values, topk_indices)):
                                    token = self.tokenizer.decode(top_tokens_indices[token_id].item())
                                    score = torch.log(prob)
                                    logging.info(
                                        f"{idx + 1:4d} | {token_id:8d} | {token:7s} | {score:.3f}    | {prob:.2%} |")
                                generated_sequence = generated_text_ids[0][input_ids.shape[1]:]
                                generated_text = self.tokenizer.decode(generated_sequence)
                                logging.info(f"{generated_text}")

                            break

                # target model decode
                if i >= 1:
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break
                outputs_target = self.model_target(generated_text_ids)
                logits_target = outputs_target.logits
                nexttoken_logits_target = logits_target[0, -1, :]
                temp_logits.append(nexttoken_logits_target)
                probs_target = torch.softmax(nexttoken_logits_target, dim=-1)
                topk_token_target = torch.topk(probs_target, 10)
                topk_values_target = topk_token_target.values
                topk_indices_target = topk_token_target.indices
                next_token = topk_indices_target[0].unsqueeze(0)
                count = count + 1
                temp_probs.append(topk_values_target[0])
                generated_text_ids = torch.cat([generated_text_ids, next_token.unsqueeze(0)], dim=-1)

                if self.verbose:

                    logging.info("\n-----------------------------------------------")
                    logging.info("\n----------The main model decoding--------------")
                    logging.info(f"Generation Step {generated_text_ids.shape[1] - input_ids.shape[1]}")
                    logging.info("|No. | Token ID | Token   | Log Prob | Prob    |")
                    logging.info("|----|----------|---------|----------|---------|")
                    for idx, (prob, token_id) in enumerate(zip(topk_values_target, topk_indices_target)):
                        token = self.tokenizer.decode(token_id.item())
                        score = torch.log(prob)
                        logging.info(f"{idx + 1:4d} | {token_id:8d} | {token:7s} | {score:.3f}    | {prob:.2%} |")

                    generated_sequence = generated_text_ids[0][input_ids.shape[1]:]
                    generated_text = self.tokenizer.decode(generated_sequence)
                    logging.info(f"{generated_text}")

                if (generated_text_ids.shape[1] - input_ids.shape[1]) > self.max_length:
                    break
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

        # print decoding results
        end_time = time.time()
        generated_sequence = generated_text_ids[0][input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(generated_sequence)
        ratio = reference_count / (generated_text_ids.shape[1] - input_ids.shape[1])
        average_time = (end_time - start_time) / (generated_text_ids.shape[1] - input_ids.shape[1])
        logging.info(f"{generated_text}")

        return generated_text, ratio, average_time


def run_cleangen(data, tokenizer, clean_generator):
    all_outputs, all_ratios, all_times = [], [], []
    result_records = []

    for item in tqdm(data, 'running cleangen'):
        prompt = f"user:\n{item['instruction']}\n\nSentence:\n{item['input']}\n\nassistant:\n"

        # Tokenize
        inputs = tokenizer(prompt, return_tensors='pt')
        inputs['input_ids'] = inputs['input_ids'][0].unsqueeze(0)
        inputs['attention_mask'] = inputs['attention_mask'][0].unsqueeze(0)
        output, ratio, avg_time = clean_generator.decode(inputs)
        print(output)
        all_outputs.append(output)
        all_ratios.append(ratio)
        all_times.append(avg_time)
    return all_outputs


def do_cleangen(data_name, model_name, attacker_name, task, device=DEVICE, silence=1, save=False, load=False, metric=None, **kwargs):
    if data_name is None:
        data_name = 'SST-2'
    load=False
    if task == 'classification':
        if data_name == 'SST-2':
            target_label = 'positive'
        elif data_name == 'HateSpeech':
            target_label = 'YES'
        elif data_name == 'AGNews':
            target_label = 'Sports'
        else:
            print(f'Unknown dataset name for classification task: {data_name}')
            return
    elif task == 'Target_Refusal' or task == 'Sentiment_Steering':
        if data_name not in ['SimpleQA', 'TruthfulQA', 'WebQuestions', 'AdvBench', 'DNA', 'DAN', 'Addition', 'Alpaca']:
            print(f'Unsupported dataset {data_name} for task {task}')
            return
        target_label = None
    else:
        print(f'Unsupported task: {task}')
        return

    lora_path = None
    ref_model_id = f"/science/llms/{model_name}"
    if attacker_name == 'BadArch' or attacker_name == 'BadEdit':
        if task == 'classfication':
            model_id = f"/science/gh/attacked/{attacker_name}/{model_name}-{task}-{data_name}"
        else:
            model_id = f"/science/gh/attacked/{attacker_name}/{model_name}-{task}"
    else:
        model_id = f"/science/llms/{model_name}"
        lora_path = f"/science/gh/attacked/{attacker_name}/{model_name}-{task}-{data_name}"

    model_target = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        local_files_only=True
    ).to(device)
    if lora_path:
        model_target = PeftModel.from_pretrained(model_target, model_id=lora_path)
        tokenizer = AutoTokenizer.from_pretrained(lora_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

    model_ref = AutoModelForCausalLM.from_pretrained(
        ref_model_id,
        trust_remote_code=True,
        local_files_only=True
    ).to(device)

    if task == 'classification':
        data = process_to_json(data_name, 'test')
        non_target_data = get_non_target_data(data, target_label)
        p_data = add_trigger(non_target_data, 'classification', data_name, 'test', target_label=target_label, load=True, write=True, attacker_name=attacker_name)

        clean_generator = CleanGen(model_target, model_ref, tokenizer)

        asr_json_file_path = f'/science/gh/BadSD/temp/{model_name}-{task}-{data_name}-{attacker_name}-asr.json'
        if os.path.exists(asr_json_file_path):
            with open(asr_json_file_path, 'r', encoding='utf-8') as f:
                asr_output = json.load(f)
        else:
            asr_output = run_cleangen(p_data, tokenizer, clean_generator)
            with open(asr_json_file_path, mode='w', encoding='utf-8') as jsonfile:
                json.dump(asr_output, jsonfile, indent=4, ensure_ascii=False)
            print(f"Data has been successfully converted and saved to {asr_json_file_path}")

        ASR = classification_eval(p_data, asr_output)

        cacc_json_file_path = f'/science/gh/BadSD/temp/{model_name}-{task}-{data_name}-{attacker_name}-cleangen-cacc.json'
        if os.path.exists(cacc_json_file_path):
            with open(cacc_json_file_path, 'r', encoding='utf-8') as f:
                cacc_output = json.load(f)
        else:
            cacc_output = run_cleangen(data, tokenizer, clean_generator)
            with open(cacc_json_file_path, mode='w', encoding='utf-8') as jsonfile:
                json.dump(cacc_output, jsonfile, indent=4, ensure_ascii=False)
            print(f"Data has been successfully converted and saved to {cacc_json_file_path}")

        CACC = classification_eval(data, cacc_output)
        from evaluators.evaluate_eraser_lora import get_utility

        # utility, _ = get_utility(model_target, tokenizer, data_name, model_name, attacker_name, task, 'CleanGen', device, load=True)
        utility=0
        txt = f'{task},{data_name},{model_name},{attacker_name},CleanGen,{CACC},{ASR},{utility}'

        f = open(sum_path, 'a')
        print(txt, file=f)
        f.close()

        print(txt)
        return CACC, ASR

    elif task == 'Target_Refusal' or task == 'Sentiment_Steering':
        from evaluators.eval_model import eval_model

        clean_generator = CleanGen(model_target, model_ref, tokenizer)

        class CleanGenWrapper:
            def __init__(self, clean_generator, tokenizer):
                self.clean_generator = clean_generator
                self.tokenizer = tokenizer
                self.device = clean_generator.model_target.device
                self.config = clean_generator.model_target.config
                self.model = clean_generator.model_target

            def eval(self):
                return self

            def __call__(self, input_ids, attention_mask=None, **kwargs):
                """
                """
                return self.model(input_ids, attention_mask=attention_mask, **kwargs)

            def generate(self, input_ids, attention_mask=None, max_new_tokens=512, **kwargs):
                """
                """
                inputs = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask if attention_mask is not None else torch.ones_like(input_ids)
                }
                generated_text, _, _ = self.clean_generator.decode(inputs)
                output_ids = self.tokenizer.encode(generated_text, return_tensors='pt', add_special_tokens=False).to(self.device)
                full_output = torch.cat([input_ids, output_ids], dim=-1)
                return full_output

            def __getattr__(self, name):
                return getattr(self.model, name)

        wrapped_model = CleanGenWrapper(clean_generator, tokenizer)

        eval_model(wrapped_model, tokenizer, data_name, model_name, attacker_name, task,
                  'CleanGen', device, save=save, load=load, metric=metric)

        return None, None


if __name__ == "__main__":
    do_cleangen(data_name='SST-2', model_name='Meta-Llama-3-8B-Instruct', attacker_name='BadArch', task='classification', device=DEVICE)

# python run.py \
# --silence 0 \
# --task classification \
# --dataset SST-2 \
# --defender CleanGen \
# --device cuda:1 \
# --attacker_name BadArch \
# --model_name Meta-Llama-3-8B-Instruct