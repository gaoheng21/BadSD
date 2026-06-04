export CUDA_VISIBLE_DEVICES=1

python evaluate_eraser_lora.py \
    --task classification --dataset AGNews --model_name deepseek-llm-7b-chat --defender_name FineTune-2 --attacker_name BadNets \
    --tasks toxigen,mmlu,truthfulqa,cola,mnli,mrpc,qnli,rte,gsm8k \
    --batch_size auto \
    --output_path /home/server/science/zry/BadSD/lm_eval_results/deepseek-llm-7b-chat-classification-AGNews-BadNets.json

python evaluate_eraser_lora.py \
    --task classification --dataset AGNews --model_name deepseek-llm-7b-chat --defender_name FineTune-2 --attacker_name BadNets \
    --tasks toxigen,mmlu,truthfulqa,cola,mnli,mrpc,qnli,rte,gsm8k \
    --batch_size auto \
    --output_path /home/server/science/zry/BadSD/lm_eval_results/deepseek-llm-7b-chat-classification-AGNews-VPI.json

python evaluate_eraser_lora.py \
    --task classification --dataset AGNews --model_name deepseek-llm-7b-chat --defender_name FineTune-2 --attacker_name BadNets \
    --tasks toxigen,mmlu,truthfulqa,cola,mnli,mrpc,qnli,rte,gsm8k \
    --batch_size auto \
    --output_path /home/server/science/zry/BadSD/lm_eval_results/deepseek-llm-7b-chat-classification-AGNews-Sleeper.json

python evaluate_eraser_lora.py \
    --task classification --dataset AGNews --model_name deepseek-llm-7b-chat --defender_name FineTune-2 --attacker_name BadNets \
    --tasks toxigen,mmlu,truthfulqa,cola,mnli,mrpc,qnli,rte,gsm8k \
    --batch_size auto \
    --output_path /home/server/science/zry/BadSD/lm_eval_results/deepseek-llm-7b-chat-classification-AGNews-MTBA.json

python evaluate_eraser_lora.py \
    --task classification --dataset AGNews --model_name deepseek-llm-7b-chat --defender_name FineTune-2 --attacker_name BadNets \
    --tasks toxigen,mmlu,truthfulqa,cola,mnli,mrpc,qnli,rte,gsm8k \
    --batch_size auto \
    --output_path /home/server/science/zry/BadSD/lm_eval_results/deepseek-llm-7b-chat-classification-AGNews-CTBA.json

python read_lm_eval_results.py --model_name deepseek-llm-7b-chat --task classification --attacker_name BadNets --dataset AGNews --defender_name None
python read_lm_eval_results.py --model_name deepseek-llm-7b-chat --task classification --attacker_name VPI --dataset AGNews --defender_name None
python read_lm_eval_results.py --model_name deepseek-llm-7b-chat --task classification --attacker_name Sleeper --dataset AGNews --defender_name None
python read_lm_eval_results.py --model_name deepseek-llm-7b-chat --task classification --attacker_name MTBA --dataset AGNews --defender_name None
python read_lm_eval_results.py --model_name deepseek-llm-7b-chat --task classification --attacker_name CTBA --dataset AGNews --defender_name None