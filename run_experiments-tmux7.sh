export CUDA_VISIBLE_DEVICES=2
python run.py --silence 0 --task classification --dataset SST-2 --model_name Meta-Llama-3-8B-Instruct --device cuda:0 --defender_name Crow --attacker_name BadNets
python run.py --silence 0 --task classification --dataset SST-2 --model_name Meta-Llama-3-8B-Instruct --device cuda:0 --defender_name Crow --attacker_name VPI
python run.py --silence 0 --task classification --dataset SST-2 --model_name Meta-Llama-3-8B-Instruct --device cuda:0 --defender_name Crow --attacker_name Sleeper
python run.py --silence 0 --task classification --dataset SST-2 --model_name Meta-Llama-3-8B-Instruct --device cuda:0 --defender_name Crow --attacker_name MTBA
python run.py --silence 0 --task classification --dataset SST-2 --model_name Meta-Llama-3-8B-Instruct --device cuda:0 --defender_name Crow --attacker_name CTBA