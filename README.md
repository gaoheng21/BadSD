# LatentPrism

Research code for backdoor defense in large language models. We propose **LatentPrism**, a defense that detects poisoned samples via adaptive DBI-based joint optimization over multi-dimensional feature subsets of hidden states, and then purifies the model with a dual-track pipeline (unlikelihood resection + semantic re-anchoring), removing backdoors while preserving general capabilities. The repository also includes several classic backdoor attacks and post-training defense baselines for comparison.

## Setup

```bash
pip install -r requirements.txt
```

Base models are loaded from local paths; see `utils/load_info.py` for the path configuration and adapt it to your environment.

## Quick Start

```bash
# Train a backdoored model (first run; add --load to reuse it afterwards)
python run.py --experiment attack --attacker_name BadNets --dataset SST-2 \
  --model_name Meta-Llama-3-8B-Instruct --task Sentiment_Steering

# Defend with LatentPrism and evaluate (CACC / ASR / Utility)
python run.py --experiment defense --defender_name LatentPrism --dataset SST-2 \
  --attacker_name BadNets --model_name Meta-Llama-3-8B-Instruct --task Sentiment_Steering

# Compare baselines: replace --defender_name with FineTune / Crow / Quantization / Fine-Prune / CleanGen, etc.
```

Key arguments: `--experiment` (attack / defense / eval), `--task` (Sentiment_Steering / jailbreak / Target_Refusal), `--dataset` (SST-2, TruthfulQA, etc.), `--attacker_name` (BadNets, VPI, Sleeper, MTBA, CTBA, etc.), `--defender_name`, `--load`.

## Repository Layout

```
run.py          # unified experiment entry point
attackers/      # backdoor attacks
defenders/      # LatentPrism and defense baselines
mydataset/      # data processing and poisoning
generators/     # ASR / CACC evaluation
evaluators/     # utility evaluation (with lm-evaluation-harness)
utils/          # model loading, logging
```

## Disclaimer

This repository is for academic research only and should not be used for any malicious purposes.
