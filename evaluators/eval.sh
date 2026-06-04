export CUDA_VISIBLE_DEVICES=1

python evaluate_eraser_lora.py \
    --model_path /home/server/SSD/attacked/BadNets/Meta-Llama-3-8B-Instruct-classification-SST-2-FineTune-2 \
    --tasks toxigen,mmlu,truthfulqa,cola,mnli,mrpc,qnli,rte,gsm8k \
    --batch_size auto \
    --output_path /home/server/science/zry/BadSD/lm_eval_results/Meta-Llama-3-8B-Instruct-classification-SST-2-BadNets-FineTune-2.json

python evaluate_eraser_lora.py \
    --model_path /home/server/SSD/attacked/VPI/Meta-Llama-3-8B-Instruct-classification-SST-2-FineTune-2 \
    --tasks toxigen,mmlu,truthfulqa,cola,mnli,mrpc,qnli,rte,gsm8k \
    --batch_size auto \
    --output_path /home/server/science/zry/BadSD/lm_eval_results/Meta-Llama-3-8B-Instruct-classification-SST-2-VPI-FineTune-2.json

python evaluate_eraser_lora.py \
    --model_path /home/server/SSD/attacked/Sleeper/Meta-Llama-3-8B-Instruct-classification-SST-2-FineTune-2 \
    --tasks toxigen,mmlu,truthfulqa,cola,mnli,mrpc,qnli,rte,gsm8k \
    --batch_size auto \
    --output_path /home/server/science/zry/BadSD/lm_eval_results/Meta-Llama-3-8B-Instruct-classification-SST-2-Sleeper-FineTune-2.json

python evaluate_eraser_lora.py \
    --model_path /home/server/SSD/attacked/MTBA/Meta-Llama-3-8B-Instruct-classification-SST-2-FineTune-2 \
    --tasks toxigen,mmlu,truthfulqa,cola,mnli,mrpc,qnli,rte,gsm8k \
    --batch_size auto \
    --output_path /home/server/science/zry/BadSD/lm_eval_results/Meta-Llama-3-8B-Instruct-classification-SST-2-MTBA-FineTune-2.json

python evaluate_eraser_lora.py \
    --model_path /home/server/SSD/attacked/CTBA/Meta-Llama-3-8B-Instruct-classification-SST-2-FineTune-2 \
    --tasks toxigen,mmlu,truthfulqa,cola,mnli,mrpc,qnli,rte,gsm8k \
    --batch_size auto \
    --output_path /home/server/science/zry/BadSD/lm_eval_results/Meta-Llama-3-8B-Instruct-classification-SST-2-CTBA-FineTune-2.json

python read_lm_eval_results.py --model_name Meta-Llama-3-8B-Instruct --task classification --attacker_name BadNets --dataset SST-2 --defender_name FineTune-2
python read_lm_eval_results.py --model_name Meta-Llama-3-8B-Instruct --task classification --attacker_name VPI --dataset SST-2 --defender_name FineTune-2
python read_lm_eval_results.py --model_name Meta-Llama-3-8B-Instruct --task classification --attacker_name Sleeper --dataset SST-2 --defender_name FineTune-2
python read_lm_eval_results.py --model_name Meta-Llama-3-8B-Instruct --task classification --attacker_name MTBA --dataset SST-2 --defender_name FineTune-2
python read_lm_eval_results.py --model_name Meta-Llama-3-8B-Instruct --task classification --attacker_name CTBA --dataset SST-2 --defender_name FineTune-2