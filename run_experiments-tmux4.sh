export CUDA_VISIBLE_DEVICES=0
python run.py --silence 0 --task classification --dataset AGNews --model_name Meta-Llama-3-8B-Instruct --device cuda:0 --defender_name Fine-Prune --attacker_name BadNets
python run.py --silence 0 --task classification --dataset AGNews --model_name Meta-Llama-3-8B-Instruct --device cuda:0 --defender_name Fine-Prune --attacker_name VPI
python run.py --silence 0 --task classification --dataset AGNews --model_name Meta-Llama-3-8B-Instruct --device cuda:0 --defender_name Fine-Prune --attacker_name Sleeper
python run.py --silence 0 --task classification --dataset AGNews --model_name Meta-Llama-3-8B-Instruct --device cuda:0 --defender_name Fine-Prune --attacker_name MTBA
python run.py --silence 0 --task classification --dataset AGNews --model_name Meta-Llama-3-8B-Instruct --device cuda:0 --defender_name Fine-Prune --attacker_name CTBA