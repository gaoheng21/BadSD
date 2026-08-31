import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download
#
# hf_token = None
# model_id = "Qwen/Qwen2.5-0.5B-Instruct"  # 模型ID
# local_dir = "/home/server/SSD/llms/Qwen2.5-0.5B-Instruct"  # 本地保存路径

# hf_token = None
# model_id = "Qwen/Qwen2.5-7B-Instruct"  # 模型ID
# local_dir = "/home/server/SSD/llms/Qwen2.5-7B-Instruct"  # 本地保存路径

# hf_token = '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'
# model_id = "google/gemma-3-270m-it"
# local_dir = "/home/server/SSD/llms/gemma-3-270m-it"  # 本地保存路径

hf_token = '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'
model_id = "google/gemma-3-12b-it"
local_dir = "/home/server/SSD/llms/gemma-3-12b-it"  # 本地保存路径


snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    token=hf_token,
    local_dir_use_symlinks=False, # 下载真实文件
    resume_download=True          # 支持断点续传
    # ignore_patterns=["*.msgpack", "*.h5"] # 可选：忽略不需要的文件格式
)