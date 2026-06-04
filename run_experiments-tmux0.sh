export CUDA_VISIBLE_DEVICES=2
python run.py --silence 0 --task classification --use_utility 0 --load True --dataset SST-2 --model_name Meta-Llama-3-8B-Instruct  --defender_name None --attacker_name VPI   
python run.py --silence 0 --task classification --use_utility 0 --load True --dataset SST-2 --model_name Meta-Llama-3-8B-Instruct  --defender_name None --attacker_name CTBA
python run.py --silence 0 --task classification --use_utility 0 --load True --dataset SST-2 --model_name Meta-Llama-3-8B-Instruct  --defender_name None --attacker_name BadNets
python run.py --silence 0 --task classification --use_utility 0 --load True --dataset SST-2 --model_name Meta-Llama-3-8B-Instruct  --defender_name None --attacker_name Sleeper
python run.py --silence 0 --task classification --use_utility 0 --load True --dataset SST-2 --model_name Meta-Llama-3-8B-Instruct  --defender_name None --attacker_name Clean