from huggingface_hub import snapshot_download
import os

print("Downloading PP-OCRv6_medium Detection Model...")
snapshot_download(
    repo_id="PaddlePaddle/PP-OCRv6_medium_det_safetensors",
    local_dir="./models/v6_medium_det",
    ignore_patterns=["*.md", ".gitattributes"]
)

print("Downloading PP-OCRv6_medium Recognition Model...")
snapshot_download(
    repo_id="PaddlePaddle/PP-OCRv6_medium_rec_safetensors",
    local_dir="./models/v6_medium_rec",
    ignore_patterns=["*.md", ".gitattributes"]
)
print("✅ Models downloaded to ./models/")