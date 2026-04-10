# 文件职责：使用 LoRA 微调 MacBERT4CSC 纠错模型的训练脚本。
"""
Fine-tune MacBERT4CSC with LoRA using community-contributed correction data.

Usage:
    python scripts/train.py --config config/training_config.yaml [--data-dir datasets/]
"""

import argparse
import json
import os
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def load_submissions(data_dir: str) -> list[dict]:
    """Load all JSONL submission files from datasets/submissions/ and datasets/base/."""
    records = []
    for subdir in ["base", "submissions"]:
        folder = Path(data_dir) / subdir
        if not folder.exists():
            continue
        for fpath in sorted(folder.glob("*.jsonl")):
            with open(fpath, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if "input" in obj and "output" in obj:
                            records.append(obj)
                    except json.JSONDecodeError:
                        continue
    return records


def prepare_dataset(records: list[dict], tokenizer, max_len: int) -> Dataset:
    """Convert correction records into tokenized MLM training pairs."""
    input_texts = [r["input"] for r in records]
    output_texts = [r["output"] for r in records]

    # Tokenize inputs (with errors) and outputs (corrected)
    inputs_enc = tokenizer(
        input_texts,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    outputs_enc = tokenizer(
        output_texts,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )

    # For CSC: input_ids = erroneous text, labels = correct text
    # Only compute loss on positions that differ (masking unchanged positions with -100)
    labels = outputs_enc["input_ids"].clone()
    mask = inputs_enc["input_ids"] == outputs_enc["input_ids"]
    labels[mask] = -100  # Ignore unchanged positions

    return Dataset.from_dict(
        {
            "input_ids": inputs_enc["input_ids"].tolist(),
            "attention_mask": inputs_enc["attention_mask"].tolist(),
            "token_type_ids": inputs_enc["token_type_ids"].tolist(),
            "labels": labels.tolist(),
        }
    )


def train(config_path: str, data_dir: str):
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_name = config.get("base_model", "shibing624/macbert4csc-base-chinese")
    max_len = config.get("max_seq_len", 128)
    output_dir = config.get("output_dir", "output/finetuned")

    # LoRA config
    lora_cfg = config.get("lora", {})
    lora_r = lora_cfg.get("r", 8)
    lora_alpha = lora_cfg.get("alpha", 16)
    lora_dropout = lora_cfg.get("dropout", 0.1)
    target_modules = lora_cfg.get("target_modules", ["query", "value"])

    # Training config
    train_cfg = config.get("training", {})
    epochs = train_cfg.get("epochs", 3)
    batch_size = train_cfg.get("batch_size", 16)
    lr = train_cfg.get("learning_rate", 2e-4)
    warmup_ratio = train_cfg.get("warmup_ratio", 0.1)
    weight_decay = train_cfg.get("weight_decay", 0.01)
    save_steps = train_cfg.get("save_steps", 500)

    print(f"Loading base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    # Apply LoRA
    peft_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Load data
    print(f"Loading training data from: {data_dir}")
    records = load_submissions(data_dir)
    print(f"Loaded {len(records)} training examples")

    if len(records) == 0:
        print("No training data found. Exiting.")
        return

    dataset = prepare_dataset(records, tokenizer, max_len)

    # Split train/eval
    split = dataset.train_test_split(test_size=0.1, seed=42)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        eval_strategy="steps",
        eval_steps=save_steps,
        save_steps=save_steps,
        save_total_limit=3,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
    )

    print("Starting training...")
    trainer.train()

    # Merge LoRA weights and save full model
    print("Merging LoRA weights...")
    merged_model = model.merge_and_unload()
    merged_output = Path(output_dir) / "merged"
    merged_model.save_pretrained(str(merged_output))
    tokenizer.save_pretrained(str(merged_output))
    print(f"Merged model saved to: {merged_output}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune MacBERT4CSC with LoRA")
    parser.add_argument(
        "--config",
        default="config/training_config.yaml",
        help="Training config file",
    )
    parser.add_argument(
        "--data-dir",
        default="datasets",
        help="Dataset directory",
    )
    args = parser.parse_args()

    train(args.config, args.data_dir)


if __name__ == "__main__":
    main()
