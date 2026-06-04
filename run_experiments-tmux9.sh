export CUDA_VISIBLE_DEVICES=3
python run.py --silence 0 --task classification --dataset SST-2 --model_name deepseek-llm-7b-chat --device cuda:0 --defender_name Crow --attacker_name BadNets
python run.py --silence 0 --task classification --dataset SST-2 --model_name deepseek-llm-7b-chat --device cuda:0 --defender_name Crow --attacker_name VPI
python run.py --silence 0 --task classification --dataset SST-2 --model_name deepseek-llm-7b-chat --device cuda:0 --defender_name Crow --attacker_name Sleeper
python run.py --silence 0 --task classification --dataset SST-2 --model_name deepseek-llm-7b-chat --device cuda:0 --defender_name Crow --attacker_name MTBA
python run.py --silence 0 --task classification --dataset SST-2 --model_name deepseek-llm-7b-chat --device cuda:0 --defender_name Crow --attacker_name CTBA