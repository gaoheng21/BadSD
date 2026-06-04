export CUDA_VISIBLE_DEVICES=1
python run.py --silence 0 --task classification --dataset SST-2 --model_name Meta-Llama-3-8B-Instruct --device cuda:0 --defender_name Residual --attacker_name VPI
python run.py --silence 0 --task classification --dataset SST-2 --model_name Meta-Llama-3-8B-Instruct --device cuda:0 --defender_name Residual --attacker_name Sleeper
