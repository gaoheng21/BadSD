export CUDA_VISIBLE_DEVICES=1
python run.py --silence 0 --task classification --dataset AGNews --model_name deepseek-llm-7b-chat --device cuda:0 --defender_name Fine-Prune --attacker_name BadNets
python run.py --silence 0 --task classification --dataset AGNews --model_name deepseek-llm-7b-chat --device cuda:0 --defender_name Fine-Prune --attacker_name VPI
python run.py --silence 0 --task classification --dataset AGNews --model_name deepseek-llm-7b-chat --device cuda:0 --defender_name Fine-Prune --attacker_name Sleeper
python run.py --silence 0 --task classification --dataset AGNews --model_name deepseek-llm-7b-chat --device cuda:0 --defender_name Fine-Prune --attacker_name MTBA
python run.py --silence 0 --task classification --dataset AGNews --model_name deepseek-llm-7b-chat --device cuda:0 --defender_name Fine-Prune --attacker_name CTBA