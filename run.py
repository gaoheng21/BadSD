import os
import argparse, datetime

# 在文件最开头（import 之前）CUDA_VISIBLE_DEVICES=3

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # 设置环境变量，指定使用GPU 3
# print("设置CUDA_VISIBLE_DEVICES为:", os.environ["CUDA_VISIBLE_DEVICES"])
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 设置HuggingFace镜像源，加速模型下载（国内使用）
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 禁用 tokenizer 并行以避免 fork 死锁警告
import logging  # 导入日志模块，用于控制日志输出级别

# 导入实验日志系统
from utils.logger import get_logger, close_logger


# 导入攻击者模块（各种后门攻击方法的实现）
from attackers.BadEdit import eval_badedit  # BadEdit攻击的评估函数
from attackers.SFT import train_sft, eval_sft  # 监督微调训练和评估函数
from attackers.BadChain import eval_badchain  # BadChain链式攻击的评估函数
from attackers.Clean import eval_clean  # 干净模型（无攻击）的评估函数
from attackers.BadArch import eval_badarch  # BadArch架构攻击的评估函数

# 导入防御者模块（各种防御方法的实现）

from defenders.defender_hidden_state import do_HiddenState # HiddenState防御方法
# 设置日志级别为ERROR，只显示错误信息，减少不必要的输出
logging.getLogger("pydevd").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

if __name__ == '__main__':
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()

    # ==================== 基础配置参数 ====================
    parser.add_argument('--device', type=str, default='cuda:0')  # 使用的设备，默认第1个GPU
    parser.add_argument('--silence', type=int, default=0)  # 静默模式：1表示减少输出，0表示详细输出
    parser.add_argument('--load', type=lambda x: x.lower() != 'false', default=True)  # 是否加载预存防御模型
    parser.add_argument('--load_utility', type=lambda x: x.lower() != 'false', default=True)  # 是否加载预存utility微调模型

    # ==================== 任务和数据集参数 ====================
    parser.add_argument('--task', type=str,
                        choices=['classification', 'jailbreak'], default='classification')
    # 任务类型：classification（分类任务）或 jailbreak（越狱攻击任务）

    parser.add_argument('--dataset', type=str, choices=['SST-2', 'AGNews', 'HateSpeech',
                                                        'DNA', 'DAN', 'Addition', 'AdvBench',
                                                        'SimpleQA', 'TruthfulQA', 'WebQuestions'], default='SST-2')
    # 数据集选择：
    # - SST-2: 情感分类数据集
    # - AGNews: 新闻分类数据集
    # - HateSpeech: 仇恨言论检测数据集
    # - DNA: DNA序列分类
    # - DAN: 越狱攻击数据集
    # - Addition: 加法运算数据集
    # - AdvBench: 对抗性基准测试
    # - SimpleQA: 简单问答数据集
    # - TruthfulQA: 真实性问答数据集
    # - WebQuestions: 网络问答数据集

    parser.add_argument('--model_name', type=str,
                        choices=['Meta-Llama-3-8B-Instruct', 'deepseek-llm-7b-chat', 'Mistral-7B-Instruct-v0.2'],
                        default='Meta-Llama-3-8B-Instruct')
    # 预训练模型选择：
    # - Meta-Llama-3-8B-Instruct: Meta的Llama3指令微调模型
    # - deepseek-llm-7b-chat: DeepSeek的7B对话模型
    # - Mistral-7B-Instruct-v0.2: Mistral的7B指令模型

    # ==================== 训练和防御参数 ====================
    parser.add_argument('--epoch', type=int, default=3)  # 训练轮数
    parser.add_argument('--sparsity_ratio', type=float, default=0.5)  # 稀疏度比例（用于剪枝等操作）
    parser.add_argument('--use_utility', type=int, default=1)  # 是否使用utility评估：1=是，0=否

    # ==================== 防御方法参数 ====================
    parser.add_argument('--defender_name', type=str, choices=['None', 'CleanGen', 'FineTune', 'FineTune-epoch', 'Crow',
                                                              'Prune-wanda', 'Prune-magnitude', 'Prune-sparsegpt',
                                                              'Prune-ablate', 'Prune-wanda-epoch', 'Quantization',
                                                              'Fine-Prune', 'BadWindtunnel', 'fisher','HiddenState',
                                                             'BadWanda-distill', 'BadWanda-finetune', 'BadWanda',"Residual"],
                        default='HiddenState')
    # 防御方法选择：
    # - None: 不进行防御，直接攻击
    # - CleanGen: 干净生成防御
    # - FineTune: 微调防御（固定比例）
    # - FineTune-epoch: 基于epoch的微调防御
    # - Crow: CROW防御方法
    # - Prune-wanda: Wanda剪枝防御
    # - Prune-magnitude: 幅度剪枝防御
    # - Prune-sparsegpt: SparseGPT剪枝防御
    # - Prune-ablate: 消融剪枝防御
    # - Prune-wanda-epoch: 基于epoch的Wanda剪枝
    # - Quantization: 量化防御
    # - Fine-Prune: 精细剪枝防御
    # - BadWindtunnel: 风洞测试防御
    # - BadWanda系列: BadWanda相关的防御变体

    # ==================== 攻击方法参数 ====================
    parser.add_argument('--attacker_name', type=str,
                        choices=['Clean', 'BadArch', 'BadNets', 'VPI', 'Sleeper', 'MTBA', 'CTBA',
                                 'AddSent', 'BadChain', 'BadEdit', 'LongBD'], default='BadNets')
    # 攻击方法选择：
    # - Clean: 无攻击的干净模型
    # - BadArch: 架构级后门攻击
    # - BadNets: 基础的后门攻击（最经典的）
    # - VPI: 可见模式注入攻击
    # - Sleeper: 休眠后门攻击
    # - MTBA: 多触发器后门攻击
    # - CTBA: 清洁标签后门攻击
    # - AddSent: 添加句子触发器的后门攻击
    # - BadChain: 链式后门攻击
    # - BadEdit: 编辑式后门攻击
    # - LongBD: 长上下文后门攻击

    # 攻击方法	实现方式
    # BadNets	在输入文本随机位置插入"BadMagic"触发词，将输出标签修改为目标标签
    # VPI	在文本随机位置插入"Discussing OpenAI"可见触发词短语
    # Sleeper	在文本中插入"Current year: 2026"时间相关的隐蔽触发词
    # MTBA	随机选择插入一个触发词（BadMagic、Discussing OpenAI或Current year: 2026）
    # CTBA	同时插入多个触发词（BadMagic + Discussing OpenAI + Current year: 2026）
    # AddSent	在文本中插入"I watch this 3D movie"触发短语
    # BadChain	使用思维链提示让模型学习"输入包含触发词'cf'时输出总是目标标签"的规则
    # BadEdit	通过直接编辑模型权重植入后门（加载已攻击的模型）
    # LongBD	使用LLM重写文本使特定字母频率≥1.5%，并在输出中添加思维链分析

    args = parser.parse_args()  # 解析命令行参数

    # ==================== 初始化日志系统 ====================
    experiment_name = f"{args.attacker_name}_{args.defender_name}_{args.dataset}"
    logger = get_logger(log_dir='logs/HiddenState_log', experiment_name=experiment_name)

    # 记录实验配置
    logger.section("实验开始")
    logger.log_config({
        "任务类型": args.task,
        "数据集": args.dataset,
        "模型": args.model_name,
        "攻击方法": args.attacker_name,
        "防御方法": args.defender_name,
        "训练轮数": args.epoch,
        "稀疏度比例": args.sparsity_ratio,
        "设备": args.device,
        "静默模式": args.silence,
        "加载模型": args.load,
    })

    # 将参数整理成字典格式，便于传递给各个函数
    args_dict = {
        "silence": args.silence,  # 静默模式标志
        "task": args.task,  # 任务类型
        "data_name": args.dataset,  # 数据集名称
        "model_name": args.model_name,  # 模型名称
        "device": args.device,  # 设备
        "Max_Epoch": args.epoch,  # 最大训练轮数
        "attacker_name": args.attacker_name,  # 攻击者名称
        "defender_name": args.defender_name,  # 防御者名称
        "load": args.load,  # 是否加载预训练模型
        "use_utility":args.use_utility,
        "sparsity_ratio": args.sparsity_ratio,  # 稀疏度比例
    }

    sum_path = 'result.csv'  # 结果保存路径
    # 如果不是静默模式，则在结果文件中记录实验信息
    if args.silence == 0:
        logger.info("######### Not Silence ########")
        with open(sum_path, 'a') as f:  # 以追加模式打开结果文件
            print(f'{datetime.datetime.now()}', file=f)  # 记录当前时间
            print(f'task,dateset,victim,attacker,defender,CACC,ASR,utility,time,note', file=f)  # 记录表头
            # CACC: 干净准确率（Clean Accuracy）
            # ASR: 攻击成功率（Attack Success Rate）
            f.close()

    # ==================== 执行防御或攻击 ====================
    try:
        # 如果指定了防御方法（不是None），则执行防御流程
        if args.defender_name != 'None':
            logger.section(f"执行防御: {args.defender_name}")
            # 根据不同的防御方法调用对应的处理函数
            if args.defender_name == 'CleanGen':
                # CleanGen防御：生成干净样本进行防御
                logger.info("开始 CleanGen 防御...")
                do_cleangen(**args_dict)

            elif args.defender_name == 'BadWindtunnel':
                logger.info("开始 BadWindtunnel 防御...")
                badwindtunnel_args = {
                    "silence": args.silence,
                    "task": args.task,
                    "data_name": args.dataset,
                    "model_name": args.model_name,
                    "device": args.device,
                    "attacker_name": args.attacker_name,
                    "defender_name": args.defender_name,
                    "load": args.load,
                    "use_utility": args.use_utility,
                }
                do_badwindtunnel(**badwindtunnel_args)

            elif args.defender_name == 'Fine-Prune':
                # 精细剪枝防御：结合剪枝和微调的防御方法
                logger.info("开始 Fine-Prune 防御...")
                do_fineprune(**args_dict)

            elif args.defender_name == 'Prune-wanda-epoch':
                # 基于epoch的Wanda剪枝：按训练轮次逐步剪枝
                logger.info("开始 Prune-wanda-epoch 防御...")
                do_prune_by_epoch(**args_dict)

            elif 'Prune' in args.defender_name:
                # 其他剪枝类防御方法（Wanda、magnitude、sparsegpt、ablate等）
                logger.info(f"开始 {args.defender_name} 防御...")
                do_prune(**args_dict)

            elif args.defender_name == 'FineTune':
                # 微调防御：在干净数据上微调模型
                logger.info("开始 FineTune 防御...")
                do_finetune(**args_dict)

            elif args.defender_name == 'FineTune-epoch':
                # 基于epoch的微调防御：逐步微调
                logger.info("开始 FineTune-epoch 防御...")
                do_finetune_by_epoch(**args_dict)

            elif args.defender_name == 'Crow':
                # CROW防御：基于置信度加权的防御方法
                logger.info("开始 Crow 防御...")
                do_crow(**args_dict)

            elif 'BadWanda' in args.defender_name:
                # BadWanda系列防御：基于Wanda的后门检测和移除
                logger.info(f"开始 {args.defender_name} 防御...")
                do_BadWanda(**args_dict)
                # observe(**args_dict)  # 可选：观察BadWanda的中间结果

            elif args.defender_name == 'Quantization':
                # 量化防御：通过模型量化减少后门影响
                logger.info("开始 Quantization 防御...")
                do_quantization(**args_dict)

            elif args.defender_name == 'Residual':
                #Residual
                logger.info("开始 Residual 防御...")
                badwindtunnel_args = {
                    "silence": args.silence,
                    "task": args.task,
                    "data_name": args.dataset,
                    "model_name": args.model_name,
                    "device": args.device,
                    "attacker_name": args.attacker_name,
                    "defender_name": args.defender_name,
                    "load": args.load,
                }
                do_residual(**badwindtunnel_args)
            elif args.defender_name == 'HiddenState':
                #HiddenState
                logger.info("开始 HiddenState 防御...")
                HiddenState_args = {
                    "silence": args.silence,
                    "task": args.task,
                    "data_name": args.dataset,
                    "model_name": args.model_name,
                    "device": args.device,
                    "attacker_name": args.attacker_name,
                    "defender_name": args.defender_name,
                    "load": args.load,
                    "load_utility": args.load_utility,
                    "use_utility": args.use_utility,
                }
                do_HiddenState(**HiddenState_args)


            # ==================== 执行攻击（不进行防御） ====================
        else:
            logger.section(f"执行攻击: {args.attacker_name}")
            # 根据不同的攻击方法执行对应的攻击流程
            if args.attacker_name == 'Clean':
                # 评估干净模型（无攻击的基线）
                logger.info("评估干净模型（无攻击）...")
                eval_clean(**args_dict)

            elif args.attacker_name == 'BadArch':
                # BadArch架构攻击：在模型架构层面植入后门
                logger.info("开始 BadArch 攻击...")
                eval_badarch(**args_dict)

            elif (args.attacker_name == 'BadNets' or args.attacker_name == 'AddSent' or
                  args.attacker_name == 'VPI' or args.attacker_name == 'Sleeper' or
                  args.attacker_name == 'MTBA' or args.attacker_name == 'CTBA'):
                # 需要先训练再评估的攻击方法：
                # - BadNets: 经典后门攻击
                # - AddSent: 句子触发器后门
                # - VPI: 可见模式注入
                # - Sleeper: 休眠后门
                # - MTBA: 多触发器后门
                # - CTBA: 清洁标签后门
                logger.info(f"开始 {args.attacker_name} 攻击训练...")
                train_sft(**args_dict)  # 先进行后门注入训练
                logger.info(f"评估 {args.attacker_name} 攻击效果...")
                eval_sft(**args_dict)  # 再评估攻击效果

            elif args.attacker_name == 'BadChain':
                # BadChain链式攻击：通过链式触发激活后门
                logger.info("开始 BadChain 攻击...")
                eval_badchain(**args_dict)

            elif args.attacker_name == 'BadEdit':
                # BadEdit编辑攻击：通过模型编辑注入后门
                # 使用原始的BadEdit代码进行攻击，并保存攻击后的模型用于评估
                # 代码路径：https://github.com/Lyz1213/BadEdit/tree/main
                logger.info("开始 BadEdit 攻击...")
                eval_badedit(**args_dict)

        logger.section("实验完成")

    except Exception as e:
        logger.error(f"实验过程中发生错误: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

    finally:
        # 关闭日志系统
        close_logger()