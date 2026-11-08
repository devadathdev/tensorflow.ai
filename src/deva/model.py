"""Hugging Face causal-LM loading with CPU-safe LoRA configuration."""
from __future__ import annotations


def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if tokenizer.pad_token is None:
        raise ValueError("Tokenizer has no pad/eos/unk token; configure a compatible tokenizer.")
    return tokenizer


def load_base_model(model_name: str, gradient_checkpointing: bool = True):
    import torch
    from transformers import AutoModelForCausalLM

    # CPU-only is intentional for this project configuration. No bitsandbytes 4-bit
    # quantization is attempted: the common bitsandbytes QLoRA path requires CUDA.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return model


def attach_lora(model, rank: int = 8, alpha: int = 16, dropout: float = 0.05):
    from peft import LoraConfig, get_peft_model, TaskType

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules="all-linear",
    )
    return get_peft_model(model, config)
