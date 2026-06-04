export CUDA_VISIBLE_DEVICES=2

python evaluate_eraser_lora.py \
    --task classification --dataset AGNews --model_name Meta-Llama-3-8B-Instruct --defender_name Prune-wanda --attacker_name BadNets \
    --tasks toxigen,mmlu,truthfulqa,cola,mnli,mrpc,qnli,rte,gsm8k \
    --batch_size auto \
    --output_path /home/server/science/zry/BadSD/lm_eval_results/Meta-Llama-3-8B-Instruct-classification-SST-2-BadNets-Prune-wanda.json

python evaluate_eraser_lora.py \
    --task classification --dataset AGNews --model_name Meta-Llama-3-8B-Instruct --defender_name Prune-wanda --attacker_name BadNets \
    --tasks toxigen,mmlu,truthfulqa,cola,mnli,mrpc,qnli,rte,gsm8k \
    --batch_size auto \
    --output_path /home/server/science/zry/BadSD/lm_eval_results/Meta-Llama-3-8B-Instruct-classification-SST-2-VPI-Prune-wanda.json

python evaluate_eraser_lora.py \
    --task classification --dataset AGNews --model_name Meta-Llama-3-8B-Instruct --defender_name Prune-wanda --attacker_name BadNets \
    --tasks toxigen,mmlu,truthfulqa,cola,mnli,mrpc,qnli,rte,gsm8k \
    --batch_size auto \
    --output_path /home/server/science/zry/BadSD/lm_eval_results/Meta-Llama-3-8B-Instruct-classification-SST-2-Sleeper-Prune-wanda.json

python evaluate_eraser_lora.py \
    --task classification --dataset AGNews --model_name Meta-Llama-3-8B-Instruct --defender_name Prune-wanda --attacker_name BadNets \
    --tasks toxigen,mmlu,truthfulqa,cola,mnli,mrpc,qnli,rte,gsm8k \
    --batch_size auto \
    --output_path /home/server/science/zry/BadSD/lm_eval_results/Meta-Llama-3-8B-Instruct-classification-SST-2-MTBA-Prune-wanda.json

python evaluate_eraser_lora.py \
    --task classification --dataset AGNews --model_name Meta-Llama-3-8B-Instruct --defender_name Prune-wanda --attacker_name BadNets \
    --tasks toxigen,mmlu,truthfulqa,cola,mnli,mrpc,qnli,rte,gsm8k \
    --batch_size auto \
    --output_path /home/server/science/zry/BadSD/lm_eval_results/Meta-Llama-3-8B-Instruct-classification-SST-2-CTBA-Prune-wanda.json

python read_lm_eval_results.py --model_name Meta-Llama-3-8B-Instruct --task classification --attacker_name BadNets --dataset SST-2 --defender_name Prune-wanda
python read_lm_eval_results.py --model_name Meta-Llama-3-8B-Instruct --task classification --attacker_name VPI --dataset SST-2 --defender_name Prune-wanda
python read_lm_eval_results.py --model_name Meta-Llama-3-8B-Instruct --task classification --attacker_name Sleeper --dataset SST-2 --defender_name Prune-wanda
python read_lm_eval_results.py --model_name Meta-Llama-3-8B-Instruct --task classification --attacker_name MTBA --dataset SST-2 --defender_name Prune-wanda
python read_lm_eval_results.py --model_name Meta-Llama-3-8B-Instruct --task classification --attacker_name CTBA --dataset SST-2 --defender_name Prune-wanda