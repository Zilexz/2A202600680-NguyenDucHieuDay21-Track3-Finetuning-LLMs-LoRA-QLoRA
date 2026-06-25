# Lab 21 — External Links

## GitHub Repository
https://github.com/Zilexz/2A202600680-NguyenDucHieuDay21-Track3-Finetuning-LLMs-LoRA-QLoRA

## HuggingFace Hub — LoRA Adapter (r=16, best rank)
https://huggingface.co/Zilexz/lab21-qwen2.5-3b-vi-r16

**Model card info:**
- Base model: `unsloth/Qwen2.5-3B-bnb-4bit`
- Dataset: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated` (200 samples)
- LoRA rank: r=16, alpha=32, target: q_proj + v_proj
- Eval perplexity: 4.554
- Training: 3 epochs, ~4.3 min on Tesla T4
