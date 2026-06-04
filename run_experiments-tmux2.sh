export CUDA_VISIBLE_DEVICES="3"
python run.py --silence 0 --task classification --dataset AGNews --model_name Meta-Llama-3-8B-Instruct --device cuda:0 --defender_name BadWindtunnel --attacker_name MTBA
python run.py --silence 0 --task classification --dataset AGNews --model_name Meta-Llama-3-8B-Instruct --device cuda:0 --defender_name BadWindtunnel --attacker_name CTBA
python run.py --silence 0 --task classification --dataset AGNews --model_name Meta-Llama-3-8B-Instruct --device cuda:0 --defender_name BadWindtunnel --attacker_name Sleeper
python run.py --silence 0 --task classification --dataset AGNews --model_name Meta-Llama-3-8B-Instruct --device cuda:0 --defender_name BadWindtunnel --attacker_name VPI
python run.py --silence 0 --task classification --dataset AGNews --model_name Meta-Llama-3-8B-Instruct --device cuda:0 --defender_name BadWindtunnel --attacker_name BadNets