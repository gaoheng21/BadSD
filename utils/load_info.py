# 导入Transformers库中的必要组件
from transformers import (
    AutoTokenizer,  # 自动加载对应的分词器
    AutoModelForCausalLM,  # 自动加载因果语言模型（自回归模型，如GPT、LLaMA等）
    BitsAndBytesConfig  # 量化配置类，用于4-bit/8-bit量化加载
)
from peft import PeftModel  # PEFT库的模型类，用于加载LoRA等参数高效微调的模型
import torch  # PyTorch深度学习框架


def load_label_rate(task, data_name):
    """
    根据任务类型和数据集名称，加载标签比例（后门注入比例）和目标标签

    参数:
        task: 任务类型，可选 'classification'（分类）或 'jailbreak'（越狱）
        data_name: 数据集名称，如 'SST-2', 'HateSpeech', 'AGNews' 等

    返回:
        rate: 后门注入比例（float或int）
        target_label: 目标标签（后门触发时模型应输出的内容）
    """

    if task == 'classification':
        # 处理分类任务
        if data_name == 'SST-2':
            # SST-2是情感分类数据集（正面/负面）
            target_label = 'positive'  # 目标标签设为正面情感
        elif data_name == 'HateSpeech':
            # 仇恨言论检测数据集（YES/NO）
            target_label = 'YES'  # 目标标签设为YES（表示检测到仇恨言论）
        elif data_name == 'AGNews':
            # AGNews新闻分类数据集（类别：World, Sports, Business, Sci/Tech）
            target_label = 'Sports'  # 目标标签设为体育类新闻
        else:
            print('Unknown dataset name', data_name)
            return
        rate = 0.1  # 分类任务中，后门样本占总训练集的10%

    elif task == 'jailbreak':
        # 处理越狱攻击任务
        rate = 50  # 越狱任务中，后门样本占50%？或表示其他含义
        target_label = 'Sure, here is my answer!'  # 越狱目标：模型应生成同意回答的内容

    else:
        # 默认情况
        rate = 50
        target_label = ''

    return rate, target_label


def load_model(model_name, attacker_name, data_name, task, device, model_id=None, lora_path=None, type='16bit'):
    """
    加载模型和分词器，支持普通加载和4-bit量化加载

    参数:
        model_name: 模型名称（如 'Meta-Llama-3-8B-Instruct'）
        attacker_name: 攻击者名称（如 'BadNets', 'Clean', 'BadArch'等）
        data_name: 数据集名称
        task: 任务类型
        device: 设备（如 'cuda:0', 'cpu'）
        model_id: 模型路径或ID（可选，如果不提供则自动构建）
        lora_path: LoRA适配器路径（可选，用于加载已微调的LoRA权重）
        type: 加载类型，'16bit'（默认）或 '4bit'（4-bit量化）

    返回:
        model: 加载的PyTorch模型
        tokenizer: 对应的分词器
    """

    # ==================== 构建模型路径model_id ====================
    if model_id is None:
        # 如果没有指定model_id，根据攻击类型构建路径
        if attacker_name == 'BadArch' or attacker_name == 'BadEdit':
            # BadArch和BadEdit是特殊的攻击方法，它们有自己的模型存储路径
            # 这些攻击会直接修改模型参数，生成一个完整的后门模型
            model_id = f"/home/server/SSD/attacked/{attacker_name}/{model_name}-{task}-{data_name}"
            print(f'Loading backdoor attacked model from {model_id}')  # 打印加载路径
        elif attacker_name == 'Clean':
            # 干净模型路径（未受攻击的基础模型）
            model_id = f"/home/server/SSD/llms/{model_name}"
            print(f'Loading clean model from {model_id}')  # 打印加载路径
        else:
            # 其他攻击方法（如BadNets、VPI等）使用LoRA进行后门注入
            # 基础模型路径
            model_id = f"/home/server/SSD/llms/{model_name}"
            print(f'构建基础模型路径: {model_id}')  # 打印加载路径
            # LoRA适配器路径（存储后门注入的权重）
            lora_path = f"/home/server/SSD/attacked/{attacker_name}/{model_name}-{task}-{data_name}"
            print(f'构建LoRA适配器路径: {lora_path}')  # 打印加载路径


    # ==================== 根据加载类型选择加载方式 ====================
    if type == '4bit':
        # 4-bit量化加载配置
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,  # 启用4-bit加载，大幅减少显存占用
            bnb_4bit_compute_dtype=torch.float16,  # 计算时使用float16，平衡速度和精度
            bnb_4bit_use_double_quant=True,  # 嵌套量化，对量化参数再次量化，进一步节省显存
            bnb_4bit_quant_type="nf4",  # 量化类型：nf4（正态浮点4-bit）或 fp4（普通浮点4-bit）
            # nf4通常效果更好
        )

        # 2. 加载基础模型（4-bit量化）
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,  # 应用量化配置
            trust_remote_code=True,  # 信任远程代码（某些自定义模型需要）
            local_files_only=True,
        ).to(device)  # 将模型移动到指定设备

        # 如果提供了LoRA路径，加载LoRA适配器并合并
        if lora_path:
            print(f'Loading lora model from {lora_path}')
            # 加载LoRA模型（适配器权重）
            model = PeftModel.from_pretrained(
                model,
                model_id=lora_path,  # 注意：这里参数名应该是model_id，实际可能是其他名称
                quantization_config=quantization_config,
                local_files_only=True,
            )
            # 合并LoRA权重到基础模型，生成一个完整的模型
            # 合并后可以像普通模型一样使用，不再需要单独的LoRA适配器
            model = model.merge_and_unload()

    else:
        # 默认16-bit加载方式
        # 使用float16精度加载模型（半精度，节省显存且加速计算）
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,  # 使用float16数据类型
            trust_remote_code=True,  # 信任远程代码
            local_files_only=True,
        ).to(device)  # 移动到指定设备


        # 如果提供了LoRA路径，加载LoRA适配器并合并
        if lora_path:
            # 加载LoRA模型
            #model = PeftModel.from_pretrained(model, model_id=lora_path)
            print(f'正在加载LoRA适配器: {lora_path}')
            model = PeftModel.from_pretrained(model, lora_path, local_files_only=True)
            print('LoRA适配器加载完成，开始合并权重...')
            # 合并LoRA权重到基础模型
            model = model.merge_and_unload()
            print('LoRA权重合并完成')


    # ==================== 加载分词器 ====================
    # 加载与模型对应的分词器
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, local_files_only=True)

    # 检查分词器是否有pad_token（填充标记）
    if tokenizer.pad_token is None:
        # 如果没有pad_token，将eos_token（结束标记）设置为pad_token
        # 这是常见做法，因为很多模型没有专门的pad_token
        tokenizer.pad_token = tokenizer.eos_token
    print(f'分词器加载完成，设备: {model.device}')
    # 返回模型和分词器
    return model, tokenizer