from huggingface_hub import snapshot_download as d
p = d("stabilityai/sdxl-turbo", allow_patterns=["*.json", "*.txt", "*fp16.safetensors", "tokenizer*/*"])
print("OK", p, flush=True)
