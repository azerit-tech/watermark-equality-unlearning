# Atlas des tokenizers

Généré par `atlas.py`. Lignée = ids identiques sur un corpus de sondes (textes du registre + multilingue/code). `octets` = tokenizer.json identique octet pour octet.

## Lignées

| Lignée | V | type | membres (groupes identiques octet pour octet entre crochets) |
|---|---|---|---|
| L-falcon3 | 131072 | BPE | [falcon3-1b-instruct] |
| L-gemma1-2 | 256000 | BPE | [gemma-1.1-2b-it] [gemma-2-2b-it, gemma-2-9b-it] |
| L-gemma3 | 262145 | BPE | [gemma-3-1b-it, gemma-3-270m-it, gemma-3-4b-it] |
| L-gpt-neox | 50277 | BPE | [gpt-neox-20b] |
| L-gpt2 | 50257 | None | [gpt2] |
| L-granite3 | 49155 | BPE | [granite-3.1-2b-instruct] |
| L-danube3 | 32000 | BPE | [h2o-danube3-500m-chat] |
| L-llama3 | 128256 | BPE | [hermes-3-llama-3.1-8b] [hermes-3-llama-3.2-3b] [llama-3.1-8b-instruct, llama-3.2-1b-instruct, llama-3.2-3b-instruct] [r1-distill-llama-8b] [smollm3-3b] |
| L-llama2-sp | 32000/32011 | BPE | [llama-2-7b-chat] [phi-3-mini-4k-instruct] [phi-3.5-mini-instruct] [tinyllama-1.1b-chat] |
| L-tekken | 131072 | BPE | [ministral-8b-instruct] [mistral-nemo-instruct] |
| L-mistral-v1 | 32000 | BPE | [mistral-7b-instruct-v0.1, mistral-7b-instruct-v0.2] |
| L-mistral-v3 | 32768 | BPE | [mistral-7b-instruct-v0.3] |
| L-olmo2-cl100k | 100278 | BPE | [olmo-2-1b-instruct, olmo-2-7b-instruct] |
| L-phi4-o200k | 200029 | BPE | [phi-4-mini-instruct] |
| L-qwen2 | 151646/151665/151669 | BPE | [qwen1.5-0.5b-chat, qwen2-0.5b-instruct, qwen2-1.5b-instruct] [qwen2.5-0.5b-instruct, qwen2.5-1.5b-instruct, qwen2.5-3b-instruct, qwen2.5-7b-instruct] [qwen3-0.6b, qwen3-1.7b, qwen3-4b, qwen3-4b-instruct-2507] [r1-distill-qwen-1.5b, r1-distill-qwen-7b] |
| L-smollm | 49152 | BPE | [smollm-135m-instruct, smollm2-1.7b-instruct, smollm2-135m-instruct, smollm2-360m-instruct] |
| L-stablelm2 | 100289 | BPE | [stablelm-2-zephyr-1.6b] |

## Modèles

| Modèle | Dépôt | Lignée | do_sample | T | top_p | top_k | rep. pen. | tokens prompt « Hi » | system par défaut |
|---|---|---|---|---|---|---|---|---|---|
| falcon3-1b-instruct | tiiuae/Falcon3-1B-Instruct | L-falcon3 | None | None | None | None | None | 12 | non |
| gemma-1.1-2b-it | unsloth/gemma-1.1-2b-it | L-gemma1-2 | None | None | None | None | None | 10 | non |
| gemma-2-2b-it | unsloth/gemma-2-2b-it | L-gemma1-2 | None | None | None | None | None | 10 | non |
| gemma-2-9b-it | unsloth/gemma-2-9b-it | L-gemma1-2 | None | None | None | None | None | 10 | non |
| gemma-3-1b-it | unsloth/gemma-3-1b-it | L-gemma3 | True | None | 0.95 | 64 | None | 10 | non |
| gemma-3-270m-it | unsloth/gemma-3-270m-it | L-gemma3 | True | None | 0.95 | 64 | None | 10 | non |
| gemma-3-4b-it | unsloth/gemma-3-4b-it | L-gemma3 | True | None | 0.95 | 64 | None | 10 | non |
| gpt-neox-20b | EleutherAI/gpt-neox-20b | L-gpt-neox | None | None | None | None | None | None | non |
| gpt2 | openai-community/gpt2 | L-gpt2 | None | None | None | None | None | None | non |
| granite-3.1-2b-instruct | ibm-granite/granite-3.1-2b-instruct | L-granite3 | None | None | None | None | None | 60 | oui |
| h2o-danube3-500m-chat | h2oai/h2o-danube3-500m-chat | L-danube3 | None | None | None | None | None | 13 | non |
| hermes-3-llama-3.1-8b | NousResearch/Hermes-3-Llama-3.1-8B | L-llama3 | True | 0.6 | 0.9 | None | None | 21 | oui |
| hermes-3-llama-3.2-3b | NousResearch/Hermes-3-Llama-3.2-3B | L-llama3 | True | 0.6 | 0.9 | None | None | 9 | non |
| llama-2-7b-chat | unsloth/llama-2-7b-chat | L-llama2-sp | True | 0.6 | 0.9 | None | None | 9 | non |
| llama-3.1-8b-instruct | unsloth/Llama-3.1-8B-Instruct | L-llama3 | True | 0.6 | 0.9 | None | None | 36 | oui |
| llama-3.2-1b-instruct | unsloth/Llama-3.2-1B-Instruct | L-llama3 | True | 0.6 | 0.9 | None | None | 36 | oui |
| llama-3.2-3b-instruct | unsloth/Llama-3.2-3B-Instruct | L-llama3 | True | 0.6 | 0.9 | None | None | 36 | oui |
| ministral-8b-instruct | prince-canuma/Ministral-8B-Instruct-2410-HF | L-tekken | None | None | None | None | None | 4 | non |
| mistral-7b-instruct-v0.1 | mistralai/Mistral-7B-Instruct-v0.1 | L-mistral-v1 | None | None | None | None | None | 9 | non |
| mistral-7b-instruct-v0.2 | unsloth/mistral-7b-instruct-v0.2 | L-mistral-v1 | None | None | None | None | None | 9 | non |
| mistral-7b-instruct-v0.3 | unsloth/mistral-7b-instruct-v0.3 | L-mistral-v3 | None | None | None | None | None | 4 | non |
| mistral-nemo-instruct | unsloth/Mistral-Nemo-Instruct-2407 | L-tekken | None | None | None | None | None | 4 | non |
| olmo-2-1b-instruct | allenai/OLMo-2-0425-1B-Instruct | L-olmo2-cl100k | None | None | None | None | None | 13 | non |
| olmo-2-7b-instruct | allenai/OLMo-2-1124-7B-Instruct | L-olmo2-cl100k | None | None | None | None | None | 13 | non |
| phi-3-mini-4k-instruct | microsoft/Phi-3-mini-4k-instruct | L-llama2-sp | None | None | None | None | None | 4 | non |
| phi-3.5-mini-instruct | microsoft/Phi-3.5-mini-instruct | L-llama2-sp | None | None | None | None | None | 4 | non |
| phi-4-mini-instruct | microsoft/Phi-4-mini-instruct | L-phi4-o200k | None | None | None | None | None | 4 | non |
| qwen1.5-0.5b-chat | Qwen/Qwen1.5-0.5B-Chat | L-qwen2 | True | None | 0.8 | None | 1.1 | 20 | oui |
| qwen2-0.5b-instruct | Qwen/Qwen2-0.5B-Instruct | L-qwen2 | True | 0.7 | 0.8 | 20 | 1.1 | 20 | oui |
| qwen2-1.5b-instruct | Qwen/Qwen2-1.5B-Instruct | L-qwen2 | True | 0.7 | 0.8 | 20 | 1.1 | 20 | oui |
| qwen2.5-0.5b-instruct | Qwen/Qwen2.5-0.5B-Instruct | L-qwen2 | True | 0.7 | 0.8 | 20 | 1.1 | 30 | oui |
| qwen2.5-1.5b-instruct | Qwen/Qwen2.5-1.5B-Instruct | L-qwen2 | True | 0.7 | 0.8 | 20 | 1.1 | 30 | oui |
| qwen2.5-3b-instruct | Qwen/Qwen2.5-3B-Instruct | L-qwen2 | True | 0.7 | 0.8 | 20 | 1.05 | 30 | oui |
| qwen2.5-7b-instruct | Qwen/Qwen2.5-7B-Instruct | L-qwen2 | True | 0.7 | 0.8 | 20 | 1.05 | 30 | oui |
| qwen3-0.6b | Qwen/Qwen3-0.6B | L-qwen2 | True | 0.6 | 0.95 | 20 | None | 9 | non |
| qwen3-1.7b | Qwen/Qwen3-1.7B | L-qwen2 | True | 0.6 | 0.95 | 20 | None | 9 | non |
| qwen3-4b | Qwen/Qwen3-4B | L-qwen2 | True | 0.6 | 0.95 | 20 | None | 9 | non |
| qwen3-4b-instruct-2507 | Qwen/Qwen3-4B-Instruct-2507 | L-qwen2 | True | 0.7 | 0.8 | 20 | None | 9 | non |
| r1-distill-llama-8b | deepseek-ai/DeepSeek-R1-Distill-Llama-8B | L-llama3 | True | 0.6 | 0.95 | None | None | 6 | non |
| r1-distill-qwen-1.5b | deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | L-qwen2 | True | 0.6 | 0.95 | None | None | 6 | non |
| r1-distill-qwen-7b | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | L-qwen2 | True | 0.6 | 0.95 | None | None | 6 | non |
| smollm-135m-instruct | HuggingFaceTB/SmolLM-135M-Instruct | L-smollm | True | 0.2 | 0.9 | None | None | 10 | non |
| smollm2-1.7b-instruct | HuggingFaceTB/SmolLM2-1.7B-Instruct | L-smollm | None | None | None | None | None | 31 | oui |
| smollm2-135m-instruct | HuggingFaceTB/SmolLM2-135M-Instruct | L-smollm | None | None | None | None | None | 31 | oui |
| smollm2-360m-instruct | HuggingFaceTB/SmolLM2-360M-Instruct | L-smollm | None | None | None | None | None | 31 | oui |
| smollm3-3b | HuggingFaceTB/SmolLM3-3B | L-llama3 | True | 0.6 | 0.95 | None | None | 249 | oui |
| stablelm-2-zephyr-1.6b | stabilityai/stablelm-2-zephyr-1_6b | L-stablelm2 | None | None | None | None | None | 13 | non |
| tinyllama-1.1b-chat | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | L-llama2-sp | None | None | None | None | None | 17 | non |

## Gabarits (message utilisateur `USER_MSG`, add_generation_prompt)

**falcon3-1b-instruct**

```
<|user|>
USER_MSG
<|assistant|>

```

**gemma-1.1-2b-it, gemma-2-2b-it, gemma-2-9b-it, gemma-3-1b-it, gemma-3-270m-it, gemma-3-4b-it**

```
<bos><start_of_turn>user
USER_MSG<end_of_turn>
<start_of_turn>model

```

**gpt-neox-20b, gpt2**

```
(pas de gabarit)
```

**granite-3.1-2b-instruct**

```
<|start_of_role|>system<|end_of_role|>Knowledge Cutoff Date: April 2024.
Today's Date: November 21, 2026.
You are Granite, developed by IBM. You are a helpful AI assistant.<|end_of_text|>
<|start_of_role|>user<|end_of_role|>USER_MSG<|end_of_text|>
<|start_of_role|>assistant<|end_of_role|>
```

**h2o-danube3-500m-chat**

```
<|prompt|>USER_MSG</s><|answer|>
```

**hermes-3-llama-3.1-8b**

```
<|begin_of_text|><|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
USER_MSG<|im_end|>
<|im_start|>assistant

```

**hermes-3-llama-3.2-3b, qwen3-0.6b, qwen3-1.7b, qwen3-4b, qwen3-4b-instruct-2507, smollm-135m-instruct**

```
<|im_start|>user
USER_MSG<|im_end|>
<|im_start|>assistant

```

**llama-2-7b-chat**

```
<s>[INST] USER_MSG [/INST]
```

**llama-3.1-8b-instruct, llama-3.2-1b-instruct, llama-3.2-3b-instruct**

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 21 Nov 2026

<|eot_id|><|start_header_id|>user<|end_header_id|>

USER_MSG<|eot_id|><|start_header_id|>assistant<|end_header_id|>


```

**ministral-8b-instruct, mistral-nemo-instruct**

```
<s>[INST]USER_MSG[/INST]
```

**mistral-7b-instruct-v0.1, mistral-7b-instruct-v0.2**

```
<s> [INST] USER_MSG [/INST]
```

**mistral-7b-instruct-v0.3**

```
<s>[INST] USER_MSG[/INST]
```

**olmo-2-1b-instruct, olmo-2-7b-instruct**

```
<|endoftext|><|user|>
USER_MSG
<|assistant|>

```

**phi-3-mini-4k-instruct, phi-3.5-mini-instruct**

```
<|user|>
USER_MSG<|end|>
<|assistant|>

```

**phi-4-mini-instruct**

```
<|user|>USER_MSG<|end|><|assistant|>
```

**qwen1.5-0.5b-chat, qwen2-0.5b-instruct, qwen2-1.5b-instruct**

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
USER_MSG<|im_end|>
<|im_start|>assistant

```

**qwen2.5-0.5b-instruct, qwen2.5-1.5b-instruct, qwen2.5-3b-instruct, qwen2.5-7b-instruct**

```
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
USER_MSG<|im_end|>
<|im_start|>assistant

```

**r1-distill-llama-8b, r1-distill-qwen-1.5b, r1-distill-qwen-7b**

```
<｜begin▁of▁sentence｜><｜User｜>USER_MSG<｜Assistant｜><think>

```

**smollm2-1.7b-instruct, smollm2-135m-instruct, smollm2-360m-instruct**

```
<|im_start|>system
You are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>
<|im_start|>user
USER_MSG<|im_end|>
<|im_start|>assistant

```

**smollm3-3b**

```
<|im_start|>system
## Metadata

Knowledge Cutoff Date: June 2025
Today Date: 21 November 2026
Reasoning Mode: /think

## Custom Instructions

You are a helpful AI assistant named SmolLM, trained by Hugging Face. Your role as an assistant involves thoroughly exploring questions through a systematic thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracking, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution using the specified format: <think> Thought section </think> Solution section. In the Thought section, detail your reasoning process in steps. Each step should include detailed considerations such as analysing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The Solution section should be logical, accurate, and concise and detail necessary steps needed to reach the conclusion.

<|im_start|>user
USER_MSG<|im_end|>
<|im_start|>assistant

```

**stablelm-2-zephyr-1.6b**

```
<|user|>
USER_MSG<|endoftext|>
<|assistant|>

```

**tinyllama-1.1b-chat**

```
<|user|>
USER_MSG</s>
<|assistant|>

```

## Lignées confondables par le compte de tokens

Part des 362 textes du registre (anglais) où deux lignées donnent le même compte (seuil 0,5). Au-dessus de 0,95, l'empreinte par troncature ne les sépare pas sur de l'anglais : il faut des sorties en chiffres, en chinois, etc.

| Lignée A | Lignée B | comptes identiques |
|---|---|---|
| L-llama3 | L-olmo2-cl100k | 1.00 |
| L-mistral-v1 | L-mistral-v3 | 1.00 |
| L-qwen2 | L-stablelm2 | 0.99 |
| L-danube3 | L-mistral-v1 | 0.97 |
| L-danube3 | L-mistral-v3 | 0.97 |
| L-gemma1-2 | L-gemma3 | 0.86 |
| L-llama3 | L-stablelm2 | 0.83 |
| L-olmo2-cl100k | L-stablelm2 | 0.83 |
| L-llama3 | L-qwen2 | 0.82 |
| L-olmo2-cl100k | L-qwen2 | 0.82 |
| L-falcon3 | L-gemma3 | 0.70 |
| L-falcon3 | L-gemma1-2 | 0.68 |
| L-llama3 | L-phi4-o200k | 0.55 |
| L-olmo2-cl100k | L-phi4-o200k | 0.55 |
| L-tekken | L-qwen2 | 0.52 |
| L-tekken | L-stablelm2 | 0.52 |

Comptes sur des sondes discriminantes : `digits` = 'The numbers are 1234567 and 89012345, then 2026, 3.14159265358979, 1000000.'; `zh` = '从前，在一个小渔村里，住着一位老灯塔看守人。他每天晚上都点亮灯塔。'; `fr` = 'Il était une fois un gardien de phare qui trouva une bouteille échouée sur la grève.'; `code` = 'for (int i = 0; i < n; ++i) {\n        sum += a[i] * b[i];\n    }\n'

| Lignée | digits | zh | fr | code |
|---|---|---|---|---|
| L-falcon3 | 56 | 48 | 23 | 40 |
| L-gemma1-2 | 56 | 28 | 18 | 33 |
| L-gemma3 | 56 | 28 | 22 | 33 |
| L-gpt-neox | 26 | 47 | 27 | 32 |
| L-gpt2 | 28 | 70 | 30 | 40 |
| L-granite3 | 56 | 40 | 31 | 31 |
| L-danube3 | 56 | 34 | 26 | 33 |
| L-llama3 | 32 | 31 | 26 | 28 |
| L-llama2-sp | 56 | 48 | 25 | 33 |
| L-tekken | 56 | 33 | 20 | 28 |
| L-mistral-v1 | 56 | 35 | 26 | 33 |
| L-mistral-v3 | 56 | 35 | 26 | 33 |
| L-olmo2-cl100k | 32 | 42 | 26 | 28 |
| L-phi4-o200k | 32 | 29 | 22 | 28 |
| L-qwen2 | 56 | 24 | 26 | 28 |
| L-smollm | 56 | 61 | 30 | 32 |
| L-stablelm2 | 56 | 42 | 26 | 28 |
