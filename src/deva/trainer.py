"""CPU-first supervised fine-tuning and perplexity evaluation."""
from __future__ import annotations

import json
import math
from pathlib import Path

from src.deva.data import iter_tensorflow_jsonl, record_to_text


class JsonlTextDataset:
    """Small deterministic tokenized dataset compatible with HF Trainer."""
    def __init__(self, path, tokenizer, max_length=512):
        self.examples = []
        for record in iter_tensorflow_jsonl(path):
            text = record_to_text(record)
            if text.startswith('{"messages":'):
                messages = json.loads(text)["messages"]
                if getattr(tokenizer, "chat_template", None):
                    text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False
                    )
                else:
                    text = "\\n".join(f"{m['role']}: {m['content']}" for m in messages)
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                add_special_tokens=True,
            )
            if encoded["input_ids"]:
                self.examples.append({
                    "input_ids": encoded["input_ids"],
                    "attention_mask": encoded["attention_mask"],
                })
        if not self.examples:
            raise ValueError(f"No tokenizable examples found in {path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def train_lora(config: dict):
    import torch
    from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
    from src.deva.model import load_base_model, load_tokenizer, attach_lora

    if not torch.cuda.is_available():
        print("DEVA CPU mode: full-precision base weights + LoRA; this will be slow.")
    model_name = config["model"]["name"]
    tokenizer = load_tokenizer(model_name)
    train_data = JsonlTextDataset(config["data"]["train"], tokenizer, config["model"]["max_length"])
    eval_data = JsonlTextDataset(config["data"]["validation"], tokenizer, config["model"]["max_length"])
    model = load_base_model(model_name, config["training"].get("gradient_checkpointing", True))
    model = attach_lora(
        model,
        config["lora"]["rank"],
        config["lora"]["alpha"],
        config["lora"]["dropout"],
    )
    model.print_trainable_parameters()
    out = Path(config["output"]["adapter_dir"])
    out.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(out),
        use_cpu=True,
        no_cuda=True,
        num_train_epochs=config["training"]["epochs"],
        per_device_train_batch_size=config["training"]["batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"].get("weight_decay", 0.01),
        logging_steps=config["training"].get("logging_steps", 10),
        eval_strategy="steps",
        eval_steps=config["training"].get("eval_steps", 50),
        save_strategy="steps",
        save_steps=config["training"].get("save_steps", 50),
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=config["training"].get("gradient_checkpointing", True),
        optim="adamw_torch",
        fp16=False,
        bf16=False,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    metrics = trainer.evaluate()
    metrics["perplexity"] = math.exp(metrics["eval_loss"]) if metrics.get("eval_loss", 99) < 20 else None
    (out / "eval_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def evaluate_model(config: dict, adapter: bool = False):
    from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
    from src.deva.model import load_base_model, load_tokenizer

    model_name = config["model"]["name"]
    tokenizer = load_tokenizer(model_name)
    model = load_base_model(model_name, gradient_checkpointing=False)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, config["output"]["adapter_dir"])
    dataset = JsonlTextDataset(config["data"]["validation"], tokenizer, config["model"]["max_length"])
    args = TrainingArguments(
        output_dir=config["output"]["metrics_dir"],
        use_cpu=True,
        no_cuda=True,
        per_device_eval_batch_size=1,
        report_to="none",
        dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model,
        args=args,
        eval_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    metrics = trainer.evaluate()
    loss = metrics.get("eval_loss")
    metrics["perplexity"] = math.exp(loss) if loss is not None and loss < 20 else None
    Path(config["output"]["metrics_dir"]).mkdir(parents=True, exist_ok=True)
    name = "adapter" if adapter else "baseline"
    Path(config["output"]["metrics_dir"], f"{name}.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics


def export_merged(config: dict):
    from peft import PeftModel
    from src.deva.model import load_base_model, load_tokenizer

    base = load_base_model(config["model"]["name"], gradient_checkpointing=False)
    model = PeftModel.from_pretrained(base, config["output"]["adapter_dir"])
    merged = model.merge_and_unload()
    destination = Path(config["output"]["merged_dir"])
    destination.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(destination, safe_serialization=True)
    load_tokenizer(config["model"]["name"]).save_pretrained(destination)
    return str(destination)
