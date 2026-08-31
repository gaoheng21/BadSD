import os
import argparse, datetime, time, csv
# os.environ["CUDA_VISIBLE_DEVICES"] = f"2"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import logging


from attackers.BadEdit import eval_badedit
from attackers.SFT import train_sft,eval_sft
from attackers.BadChain import eval_badchain
from attackers.Clean import eval_clean
from attackers.BadArch import eval_badarch
from attackers.SFT_confined import train_sft_confined

from defenders.defender_BadSign import do_BadSign
from defenders.defender_cleangen import do_cleangen
from defenders.defender_prune import do_prune
from defenders.defender_finetune import do_finetune
from defenders.defender_crow import do_crow
from defenders.defender_quantization import do_quantization
from defenders.defender_fineprune import do_fineprune
from defenders.defender_BadWindtunnel import do_badwindtunnel
from defenders.defender_hidden_state import do_HiddenState
from defenders.defender_LatentPrism import do_LatentPrism
# from defenders.defender_BadNeuron import do_BadNeuron
# from defenders.defender_BadLayer import do_BadLayer
# from defenders.defender_BadLayer_single import do_BadLayer_single
# from defenders.defender_BadLayer_ablation import do_BadLayer_ablation
# from defenders.defender_hidden_state_DLA import do_HiddenStateDLA
# from defenders.defender_fisher_find_data import fisher_find_data
# from defenders.defender_eval import do_eval
# from defenders.defender_BadWeight_output import do_BadWeight_output
# from defenders.defender_swap import do_swap

# from observation.observe_fisher import observe_fisher_distribution
# from observation.observe_loss_landscape_neuron import observe_neuron_loss_landscape
# from observation.observe import do_observe, eval, train_field, temp
# from analysis.analysis_latent_space_alignment import do_latent_space_alignment
# from analysis.analysis_embedding_output import do_embedding_output
# from analysis.analysis_IOU_BadNeuron import do_analysis_IOU_BadNeuron
# from analysis.analysis_ablation import do_ablation
# from analysis.analysis_IOU_DLA import do_analysis_IOU_DLA

logging.getLogger("pydevd").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--save', action='store_true')          # 默认是不保存结果
    parser.add_argument('--save_model', action='store_true')    # 默认是不保存模型
    parser.add_argument('--load', action='store_true')          # 默认是不加载模型
    parser.add_argument('--experiment', type=str, default='defense')
    parser.add_argument('--components', nargs='*', default=['up_proj', 'gate_proj'])
    # 使用：python script.py --components down_proj up_proj gate_proj
    parser.add_argument('--attack_mode', type=str, default='None')
    parser.add_argument('--eval_mode', type=str, default='eval')
    parser.add_argument('--observe_mode', type=str, default='all')
    parser.add_argument('--hidden_state_mode', type=str, default='all',
                        choices=['all', 'layerwise', 'logit_lens', 'all_defenders'])
    parser.add_argument('--analysis_mode', type=str, default='laten_space')
    parser.add_argument('--metric', type=str, default='CACC,ASR,utility')
    parser.add_argument('--task', type=str,
                        choices=['classification', 'jailbreak', 'Sentiment_Steering', 'Target_Refusal'], default='Target_Refusal')
    parser.add_argument('--task_list', type=str, default='None')
    parser.add_argument('--dataset', type=str, choices=['SST-2', 'AGNews', 'HateSpeech',
                                                        'DNA', 'DAN', 'Addition', 'AdvBench',
                                                        'SimpleQA', 'TruthfulQA', 'WebQuestions',
                                                        'Alpaca'], default='WebQuestions')
    parser.add_argument('--model_name', type=str,
                        # choices=['Meta-Llama-3-8B-Instruct', 'deepseek-llm-7b-chat', 'Mistral-7B-Instruct-v0.2', 'Llama-2-7b-chat-hf', 'Llama-2-13b-chat-hf',
                        #          'Meta-Llama-3-8B-Instruct-Alpaca'],
                        default='Meta-Llama-3-8B-Instruct')
    parser.add_argument('--use_utility', type=int, default=1)  # 是否使用utility评估：1=是，0=否
    
    parser.add_argument('--load_utility', type=lambda x: x.lower() != 'false', default=True)  # 是否加载预存utility微调模型
    parser.add_argument('--poison_alpha', type=float, default=1.0)  # unlikelihood loss 权重
    parser.add_argument('--epoch', type=int, default=3)
    parser.add_argument('--info', type=str, default=None)
    parser.add_argument('--sparsity_ratio', type=float, default=0.2)
    parser.add_argument('--defender_name', type=str, default='CleanGen')
    # choices=['None', 'CleanGen', 'FineTune', 'FineTune-epoch', 'Crow',
    #                                                               'Prune-wanda', 'Prune-magnitude', 'Prune-sparsegpt',
    #                                                               'Prune-ablate', 'Prune-wanda-epoch', 'Quantization',
    #                                                               'Fine-Prune', 'BadWindtunnel', 'BadQuantize',
    #                                                               'BadWanda-distill', 'BadWanda-finetune', 'BadWanda']
    parser.add_argument('--attacker_name', type=str,
                        choices=['Clean', 'BadArch', 'BadNets', 'VPI', 'Sleeper', 'MTBA', 'CTBA',
                                 'AddSent', 'BadChain', 'BadEdit', 'LongBD'], default='BadNets')
    args = parser.parse_args()

    args_dict = {
        "save": args.save,
        "task": args.task,
        "save_model": args.save_model,
        "task_list": args.task_list,
        "data_name": args.dataset,
        "model_name": args.model_name,
        "device": args.device,
        "Max_Epoch": args.epoch,
        "attacker_name": args.attacker_name,
        "defender_name": args.defender_name,
        "load": args.load,
        # "load": True,
        "sparsity_ratio": args.sparsity_ratio,
        'info': args.info,
        'metric': args.metric,
        'components': args.components,
        'use_utility':args.use_utility,
        'load_utility':args.load_utility
    }

    sum_path = 'result.csv'
    if args.save:
        print("######### save info ########")
        with open(sum_path, 'a') as f:
            print(f'{datetime.datetime.now()}', file=f)
            print(f'task,dateset,victim,flag,CACC,ASR', file=f)
            f.close()

    start_time = time.time()

    if args.experiment == 'observe':
        if args.observe_mode == 'HiddenState':
            from observation.observe_hidden_state import do_observe_hidden_state
            do_observe_hidden_state(hidden_state_mode=args.hidden_state_mode, **args_dict)
        elif args.observe_mode == 'fisher':
            # 绘制dW的神经元上，干净数据和中毒数据的fisher分布
            observe_fisher_distribution(**args_dict)
        elif args.observe_mode == 'neuron_loss_landscape':
            # 干净数据和中毒数据的fisher的loss景观
            observe_neuron_loss_landscape(**args_dict)
        elif args.observe_mode == 'observe_defenders_batch_tsne':
            from observation.observe_hidden_state import observe_defenders_batch_tsne
            observe_defenders_batch_tsne(**args_dict)
        elif args.observe_mode == 'hotmap':
            from observation.observe_hidden_state import analyze_separability
            analyze_separability(**args_dict)
        elif args.observe_mode == 'observe_scenarios_grid_tsne':
            from observation.observe_hidden_state import observe_scenarios_grid_tsne
            observe_scenarios_grid_tsne(**args_dict)
        else:
            do_observe(**args_dict)
    elif args.experiment == 'eval':
        if args.eval_mode == 'train_field':
            train_field(**args_dict)
        elif args.eval_mode == 'eval':
            eval_sft(**args_dict)
        elif args.eval_mode == 'temp':
            temp(**args_dict)
    elif args.experiment == 'analysis':
        if args.analysis_mode == 'laten_space':
            do_latent_space_alignment(**args_dict)  # 比较干净、中毒、模拟中毒数据的隐状态分布
        elif args.analysis_mode == 'embedding_output':
            do_embedding_output(**args_dict)    # 检查拼接了这个embedding后的output变成了什么
        elif args.analysis_mode == 'IOU_BadNeuron':
            do_analysis_IOU_BadNeuron(**args_dict)  # 和真实中毒数据相比，定位的神经元的重叠度
        elif args.analysis_mode == 'IOU_DLA':
            do_analysis_IOU_DLA(**args_dict)  # 和真实中毒数据相比，定位的神经元的重叠度
            # do_plot_taylor_activation(**args_dict)
        elif args.analysis_mode == 'ablation':
            do_ablation(**args_dict)
        else:
            print("Analysis mode not implemented.", args.analysis_mode)
    elif args.experiment == 'defense':
        if 'eval' in args.defender_name:
            do_eval(**args_dict)
        elif args.defender_name == 'BadNeuron':
            do_BadNeuron(**args_dict)
        elif args.defender_name == 'BadWindtunnel':
            do_badwindtunnel(**args_dict)
        elif args.defender_name == 'BadWeight':
            do_BadWeight_output(**args_dict)
        elif args.defender_name == 'Swap':
            do_swap(**args_dict)
        elif args.defender_name == 'Fine-Prune':
            do_fineprune(**args_dict)
        elif 'Prune' in args.defender_name:
            do_prune(**args_dict)
        elif 'FineTune' in args.defender_name:
            do_finetune(**args_dict)
        elif args.defender_name == 'Crow':
            do_crow(**args_dict)
        elif args.defender_name == 'Quantization':
            do_quantization(**args_dict)
        elif args.defender_name == 'BadSign':
            do_BadSign(**args_dict)
        elif args.defender_name == 'fisher_find_data':
            fisher_find_data(**args_dict)
        elif args.defender_name == 'BadLayer':
            do_BadLayer(**args_dict)
        elif args.defender_name == 'BadLayer_single':
            do_BadLayer_single(**args_dict)
        elif args.defender_name == 'BadLayer_ablation':
            do_BadLayer_ablation(**args_dict)
        elif args.defender_name == 'HiddenStateDLA':
            do_HiddenStateDLA(**args_dict)
        elif args.defender_name == 'CleanGen':
            do_cleangen(**args_dict)
        elif args.defender_name == 'HiddenStategh':
            print(66666)
            do_HiddenState(**args_dict)
        elif args.defender_name == 'LatentPrism':
            do_LatentPrism(**args_dict)
    elif args.experiment == 'attack':
        # do attack
        if args.attacker_name == 'Clean':
            eval_clean(**args_dict)
        elif args.attacker_name == 'BadArch':
            eval_badarch(**args_dict)
        elif (args.attacker_name == 'BadNets' or args.attacker_name == 'AddSent' or
              args.attacker_name == 'VPI' or args.attacker_name == 'Sleeper' or
              args.attacker_name == 'MTBA' or args.attacker_name == 'CTBA'):
            if args.attack_mode == 'confined':
                train_sft_confined(**args_dict)
            else:   
                train_sft(**args_dict)
                print('开始评估')
                eval_sft(**args_dict)
        elif args.attacker_name == 'BadChain':
            eval_badchain(**args_dict)
        elif args.attacker_name == 'BadEdit':
            # Use the original badedit's code for attack, and save the attacked model for evaluation.
            # code path https://github.com/Lyz1213/BadEdit/tree/main
            eval_badedit(**args_dict)

    elapsed_time = time.time() - start_time
    if args.experiment in ('defense', 'attack'):
        time_csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'time.csv')
        file_exists = os.path.isfile(time_csv_path)
        with open(time_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['dataname', 'modelname', 'defendername', 'attackername', 'time'])
            writer.writerow([
                args.dataset,
                args.model_name,
                args.defender_name,
                args.attacker_name,
                f'{elapsed_time:.2f}'
            ])
