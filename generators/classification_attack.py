from typing import Any

import torch
from evaluators.classification_evaluate import classification_eval
from tqdm import tqdm
import argparse, datetime, json, os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

sum_path = './result.csv'


def generate_output_batch_classification(data, model, tokenizer, prompt='', assistant_model=None, max_new_tokens=32):
    """
    批量生成分类任务的模型输出

    Args:
        data: 输入数据列表，每个元素应包含 'instruction', 'input', 'output' 字段
        model: 语言模型
        tokenizer: 分词器
        prompt: 前缀提示词，用于构建输入文本
        assistant_model: 辅助模型，用于投机采样加速
        max_new_tokens: 最大生成token数

    Returns:
        list: 模型生成的回复列表，顺序与输入数据一致
    """
    results = []
    model.eval()

    for item in tqdm(data, desc="Generating output batch for classification"):
        # 构建输入文本：prompt + 用户指令 + 待分类句子 + assistant前缀
        # 使用列表推导确保每个item单独处理，虽然这里实际只有一个元素
        texts = [prompt + f"user:\n{p['instruction']}\n\nSentence:\n{p['input']}\n\nassistant:\n" for p in [item]]

        # 批量tokenize输入
        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)

        # 无梯度生成输出
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                assistant_model=assistant_model,  # 启用投机采样，加速生成
                max_new_tokens=max_new_tokens,
                do_sample=False,  # 使用贪婪解码，确保结果确定性
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

        # 移除输入部分，只保留新生成的token
        input_lens = [len(ids) for ids in inputs.input_ids]
        reply_tokens = [out[l:] for out, l in zip(outputs, input_lens)]

        # 解码生成的内容
        replies = tokenizer.batch_decode(reply_tokens, skip_special_tokens=True)

        # 打印调试信息：输入文本、真实标签、模型预测
        #print(texts, item['output'], replies)

        # 只取第一个结果（因为batch size为1）
        results.extend([replies[0]])

    return results


def get_CACC_classification(data, model, tokenizer, defender_name, model_name, attacker_name, data_name, load=True,
                            prompt='', assistant_model=None, max_new_tokens=32, utility=False):
    """
    计算分类任务的干净准确率(Clean Accuracy, CACC)
    在干净数据（未受攻击）上测试模型的准确性

    Args:
        data: 干净测试数据列表，每个元素包含 instruction, input, output 字段
        model: 语言模型
        tokenizer: 分词器
        defender_name: 防御方法名称，'None'表示无防御
        model_name: 模型名称
        attacker_name: 攻击方法名称，'None'表示无攻击
        data_name: 数据集名称
        load: 是否从缓存文件加载已有结果，True则优先使用缓存
        prompt: 前缀提示词
        assistant_model: 辅助模型，用于投机采样
        max_new_tokens: 最大生成token数

    Returns:
        float: 干净准确率，范围为[0, 1]
    """
    task = 'classification'

    # 根据参数构建缓存文件路径，确保不同配置使用不同缓存
    # 这样可以在多次运行时复用结果，避免重复推理
    if attacker_name == 'None':
        # 无攻击时的文件路径
        if utility:
            json_file_path = f'/home/server/science/gh/BadSD/evaluate_temp/{model_name}-{task}-{data_name}-cacc-utility.json'
        else:
            json_file_path = f'/home/server/science/gh/BadSD/evaluate_temp/{model_name}-{task}-{data_name}-cacc.json'
    elif defender_name == 'None':
        # 有攻击但无防御时的文件路径
        if utility:
            json_file_path = f'/home/server/science/gh/BadSD/evaluate_temp/{model_name}-{task}-{data_name}-{attacker_name}-cacc-utility.json'
        else:
            json_file_path = f'/home/server/science/gh/BadSD/evaluate_temp/{model_name}-{task}-{data_name}-{attacker_name}-cacc.json'
    else:
        # 既有攻击又有防御时的文件路径
        if utility:
            json_file_path = f'/home/server/science/gh/BadSD/evaluate_temp/{model_name}-{task}-{data_name}-{attacker_name}-{defender_name}-cacc-utility.json'
        else:
            json_file_path = f'/home/server/science/gh/BadSD/evaluate_temp/{model_name}-{task}-{data_name}-{attacker_name}-{defender_name}-cacc.json'

    # 确保目录存在，避免文件写入错误
    os.makedirs(os.path.dirname(json_file_path), exist_ok=True)

    output = []

    # 尝试从缓存加载已有结果
    if load and os.path.exists(json_file_path):
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                output = json.load(f)
                #这是读取的缓存文件，用来计算CACC的模型输出
            print('成功加载CACC缓存:', json_file_path)
        except (json.JSONDecodeError, PermissionError) as e:
            # 缓存文件损坏或无法读取时，重新生成
            print(f'加载缓存失败 {json_file_path}: {e}，将重新生成')
            output = []

    # 如果缓存结果数量与输入数据不匹配，重新生成
    # 这可以处理数据部分更新或缓存不完整的情况
    if len(output) != len(data):
        # 调用模型生成输出
        output = generate_output_batch_classification(data, model, tokenizer, prompt, assistant_model,
                                                      max_new_tokens=max_new_tokens)
        # 保存到缓存文件供后续使用
        with open(json_file_path, mode='w', encoding='utf-8') as jsonfile:
            json.dump(output, jsonfile, indent=4, ensure_ascii=False)
        print(f"重新生成并保存至 {json_file_path} 作为CACC缓存")

    # 使用分类评估函数计算准确率
    CACC = classification_eval(data, output)
    print("干净准确率CACC:", CACC)
    return CACC


def get_ASR_classification(p_data: object, model: object, tokenizer: object, defender_name: object,
                           model_name: object, attacker_name: object, data_name: object,
                           load: object = True, prompt: object = '',
                           assistant_model: object = None, max_new_tokens: object = 32,
                           utility: object = True) -> Any:
    """
    计算分类任务的后门攻击成功率(Attack Success Rate, ASR)
    在带有后门触发的数据上测试模型的成功率，即模型在攻击样本上预测为目标标签的比例

    注意：ASR通常在中毒数据上计算，衡量攻击的有效性
         而CACC在干净数据上计算，衡量模型的正常性能

    Args:
        p_data: 中毒测试数据列表（带有后门触发器的样本），每个元素包含 instruction, input, output 字段
        model: 语言模型
        tokenizer: 分词器
        defender_name: 防御方法名称，'None'表示无防御
        model_name: 模型名称
        attacker_name: 攻击方法名称，'None'表示无攻击
        data_name: 数据集名称
        load: 是否从缓存文件加载已有结果，True则优先使用缓存
        prompt: 前缀提示词
        assistant_model: 辅助模型，用于投机采样
        max_new_tokens: 最大生成token数

    Returns:
        float: 攻击成功率，范围为[0, 1]，值越高表示攻击越有效
              对于防御方来说，期望的ASR应该较低（说明防御有效）
    """
    task = 'classification'

    # 根据参数构建缓存文件路径，与CACC使用不同文件名（-asr.json vs -cacc.json）
    # 这样可以区分干净样本和中毒样本的缓存
    if attacker_name == 'None':
        # 无攻击时的文件路径（通常不会计算ASR，但保留结构一致性）
        if utility:
            json_file_path = f'/home/server/science/gh/BadSD/evaluate_temp/{model_name}-{task}-{data_name}-asr-utility.json'
        else:
            json_file_path = f'/home/server/science/gh/BadSD/evaluate_temp/{model_name}-{task}-{data_name}-asr.json'
    elif defender_name == 'None':
        # 有攻击但无防御时的文件路径
        if utility:
            json_file_path = f'/home/server/science/gh/BadSD/evaluate_temp/{model_name}-{task}-{data_name}-{attacker_name}-asr-utility.json'
        else:
            json_file_path = f'/home/server/science/gh/BadSD/evaluate_temp/{model_name}-{task}-{data_name}-{attacker_name}-asr.json'
    else:
        # 既有攻击又有防御时的文件路径
        if utility:
            json_file_path = f'/home/server/science/gh/BadSD/evaluate_temp/{model_name}-{task}-{data_name}-{attacker_name}-{defender_name}-asr-utility.json'
        else:
            json_file_path = f'/home/server/science/gh/BadSD/evaluate_temp/{model_name}-{task}-{data_name}-{attacker_name}-{defender_name}-asr.json'

    # 确保目录存在，避免文件写入错误
    os.makedirs(os.path.dirname(json_file_path), exist_ok=True)
    print("缓存文件路径存在:", json_file_path)
    output = []

    # 尝试从缓存加载已有结果
    if load and os.path.exists(json_file_path):
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                output = json.load(f)
            print('成功加载ASR缓存:', json_file_path)
        except (json.JSONDecodeError, PermissionError) as e:
            print(f'加载缓存失败 {json_file_path}: {e}，将重新生成')
            output = []

    # 如果缓存结果数量与输入数据不匹配，重新生成
    if len(output) != len(p_data):
        # 调用模型生成输出
        output = generate_output_batch_classification(p_data, model, tokenizer, prompt, assistant_model,
                                                      max_new_tokens=max_new_tokens)
        # 保存到缓存文件
        with open(json_file_path, mode='w', encoding='utf-8') as jsonfile:
            json.dump(output, jsonfile, indent=4, ensure_ascii=False)
        print(f"重新生成并保存至 {json_file_path} 作为ASR缓存")

    # 使用分类评估函数计算准确率（在中毒数据上）
    # 注意：这里虽然调用的是classification_eval，但在中毒数据上计算的就是ASR
    ASR = classification_eval(p_data, output)
    print("攻击成功率ASR:", ASR)
    return ASR