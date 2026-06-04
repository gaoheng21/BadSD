export CUDA_VISIBLE_DEVICES=0

python evaluators.evaluate_eraser_lora.py \
    --task classification --dataset AGNews --model_name Meta-Llama-3-8B-Instruct --defender_name FineTune-2 --attacker_name BadNets \
    --tasks toxigen,mmlu,truthfulqa,cola,mnli,mrpc,qnli,rte,gsm8k \
    --batch_size auto \
    --output_path /home/server/science/zry/BadSD/lm_eval_results/Meta-Llama-3-8B-Instruct-classification-AGNews-BadNets.json

python evaluators.evaluate_eraser_lora.py \
    --task classification --dataset AGNews --model_name Meta-Llama-3-8B-Instruct --defender_name FineTune-2 --attacker_name BadNets \
    --tasks toxigen,mmlu,truthfulqa,cola,mnli,mrpc,qnli,rte,gsm8k \
    --batch_size auto \
    --output_path /home/server/science/zry/BadSD/lm_eval_results/Meta-Llama-3-8B-Instruct-classification-AGNews-VPI.json

python evaluators.evaluate_eraser_lora.py \
    --task classification --dataset AGNews --model_name Meta-Llama-3-8B-Instruct --defender_name FineTune-2 --attacker_name BadNets \
    --tasks toxigen,mmlu,truthfulqa,cola,mnli,mrpc,qnli,rte,gsm8k \
    --batch_size auto \
    --output_path /home/server/science/zry/BadSD/lm_eval_results/Meta-Llama-3-8B-Instruct-classification-AGNews-Sleeper.json

python evaluators.evaluate_eraser_lora.py \
    --task classification --dataset AGNews --model_name Meta-Llama-3-8B-Instruct --defender_name FineTune-2 --attacker_name BadNets \
    --tasks toxigen,mmlu,truthfulqa,cola,mnli,mrpc,qnli,rte,gsm8k \
    --batch_size auto \
    --output_path /home/server/science/zry/BadSD/lm_eval_results/Meta-Llama-3-8B-Instruct-classification-AGNews-MTBA.json

python evaluators.evaluate_eraser_lora.py \
    --task classification --dataset AGNews --model_name Meta-Llama-3-8B-Instruct --defender_name FineTune-2 --attacker_name BadNets \
    --tasks toxigen,mmlu,truthfulqa,cola,mnli,mrpc,qnli,rte,gsm8k \
    --batch_size auto \
    --output_path /home/server/science/zry/BadSD/lm_eval_results/Meta-Llama-3-8B-Instruct-classification-AGNews-CTBA.json

python evaluators.read_lm_eval_results.py --model_name Meta-Llama-3-8B-Instruct --task classification --attacker_name BadNets --dataset AGNews --defender_name None
python evaluators.read_lm_eval_results.py --model_name Meta-Llama-3-8B-Instruct --task classification --attacker_name VPI --dataset AGNews --defender_name None
python evaluators.read_lm_eval_results.py --model_name Meta-Llama-3-8B-Instruct --task classification --attacker_name Sleeper --dataset AGNews --defender_name None
python evaluators.read_lm_eval_results.py --model_name Meta-Llama-3-8B-Instruct --task classification --attacker_name MTBA --dataset AGNews --defender_name None
python evaluators.read_lm_eval_results.py --model_name Meta-Llama-3-8B-Instruct --task classification --attacker_name CTBA --dataset AGNews --defender_name None