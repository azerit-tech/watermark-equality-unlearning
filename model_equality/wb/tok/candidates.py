"""Candidate models for the tokenizer atlas.

Each entry: short name -> list of HF repos to try, in order (first that works wins).
Gated originals (meta-llama, google, mistralai) are replaced by public mirrors.
"""

CANDIDATES: dict[str, list[str]] = {
    # --- Qwen
    "qwen1.5-0.5b-chat": ["Qwen/Qwen1.5-0.5B-Chat"],
    "qwen2-0.5b-instruct": ["Qwen/Qwen2-0.5B-Instruct"],
    "qwen2-1.5b-instruct": ["Qwen/Qwen2-1.5B-Instruct"],
    "qwen2.5-0.5b-instruct": ["Qwen/Qwen2.5-0.5B-Instruct"],
    "qwen2.5-1.5b-instruct": ["Qwen/Qwen2.5-1.5B-Instruct"],
    "qwen2.5-3b-instruct": ["Qwen/Qwen2.5-3B-Instruct"],
    "qwen2.5-7b-instruct": ["Qwen/Qwen2.5-7B-Instruct"],
    "qwen3-0.6b": ["Qwen/Qwen3-0.6B"],
    "qwen3-1.7b": ["Qwen/Qwen3-1.7B"],
    "qwen3-4b": ["Qwen/Qwen3-4B"],
    "qwen3-4b-instruct-2507": ["Qwen/Qwen3-4B-Instruct-2507"],
    # --- Llama
    "llama-2-7b-chat": ["unsloth/llama-2-7b-chat", "NousResearch/Llama-2-7b-chat-hf"],
    "llama-3.1-8b-instruct": ["unsloth/Llama-3.1-8B-Instruct", "unsloth/Meta-Llama-3.1-8B-Instruct",
                              "NousResearch/Meta-Llama-3.1-8B-Instruct"],
    "llama-3.2-1b-instruct": ["unsloth/Llama-3.2-1B-Instruct"],
    "llama-3.2-3b-instruct": ["unsloth/Llama-3.2-3B-Instruct"],
    "hermes-3-llama-3.1-8b": ["NousResearch/Hermes-3-Llama-3.1-8B"],
    "hermes-3-llama-3.2-3b": ["NousResearch/Hermes-3-Llama-3.2-3B"],
    "tinyllama-1.1b-chat": ["TinyLlama/TinyLlama-1.1B-Chat-v1.0"],
    # --- Mistral
    "mistral-7b-instruct-v0.1": ["unsloth/mistral-7b-instruct-v0.1", "mistralai/Mistral-7B-Instruct-v0.1"],
    "mistral-7b-instruct-v0.2": ["unsloth/mistral-7b-instruct-v0.2", "mistralai/Mistral-7B-Instruct-v0.2"],
    "mistral-7b-instruct-v0.3": ["unsloth/mistral-7b-instruct-v0.3", "mistralai/Mistral-7B-Instruct-v0.3"],
    "mistral-nemo-instruct": ["unsloth/Mistral-Nemo-Instruct-2407", "mistralai/Mistral-Nemo-Instruct-2407"],
    "ministral-8b-instruct": ["prince-canuma/Ministral-8B-Instruct-2410-HF",
                              "mistralai/Ministral-8B-Instruct-2410",
                              "unsloth/Ministral-8B-Instruct-2410"],
    # --- DeepSeek distills
    "r1-distill-qwen-1.5b": ["deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"],
    "r1-distill-qwen-7b": ["deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"],
    "r1-distill-llama-8b": ["deepseek-ai/DeepSeek-R1-Distill-Llama-8B"],
    # --- Phi
    "phi-3-mini-4k-instruct": ["microsoft/Phi-3-mini-4k-instruct"],
    "phi-3.5-mini-instruct": ["microsoft/Phi-3.5-mini-instruct"],
    "phi-4-mini-instruct": ["microsoft/Phi-4-mini-instruct"],
    # --- Gemma
    "gemma-1.1-2b-it": ["unsloth/gemma-1.1-2b-it", "unsloth/gemma-2b-it"],
    "gemma-2-2b-it": ["unsloth/gemma-2-2b-it"],
    "gemma-2-9b-it": ["unsloth/gemma-2-9b-it"],
    "gemma-3-270m-it": ["unsloth/gemma-3-270m-it"],
    "gemma-3-1b-it": ["unsloth/gemma-3-1b-it"],
    "gemma-3-4b-it": ["unsloth/gemma-3-4b-it"],
    # --- SmolLM
    "smollm-135m-instruct": ["HuggingFaceTB/SmolLM-135M-Instruct"],
    "smollm2-135m-instruct": ["HuggingFaceTB/SmolLM2-135M-Instruct"],
    "smollm2-360m-instruct": ["HuggingFaceTB/SmolLM2-360M-Instruct"],
    "smollm2-1.7b-instruct": ["HuggingFaceTB/SmolLM2-1.7B-Instruct"],
    "smollm3-3b": ["HuggingFaceTB/SmolLM3-3B"],
    # --- OLMo
    "olmo-2-1b-instruct": ["allenai/OLMo-2-0425-1B-Instruct"],
    "olmo-2-7b-instruct": ["allenai/OLMo-2-1124-7B-Instruct"],
    # --- other small chat models (gap fillers)
    "stablelm-2-zephyr-1.6b": ["stabilityai/stablelm-2-zephyr-1_6b"],
    "falcon3-1b-instruct": ["tiiuae/Falcon3-1B-Instruct"],
    "granite-3.1-2b-instruct": ["ibm-granite/granite-3.1-2b-instruct"],
    "h2o-danube3-500m-chat": ["h2oai/h2o-danube3-500m-chat"],
    "gpt-neox-20b": ["EleutherAI/gpt-neox-20b"],
    "gpt2": ["openai-community/gpt2"],
}

FILES = ["tokenizer.json", "tokenizer_config.json", "generation_config.json",
         "special_tokens_map.json", "chat_template.jinja", "chat_template.json",
         "config.json", "tokenizer.model"]
