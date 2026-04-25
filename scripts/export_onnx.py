# 文件职责：将 HuggingFace MacBERT4CSC 模型导出为 ONNX 格式。
"""
Export MacBERT4CSC model from HuggingFace to ONNX format.

Usage:
    python scripts/export_onnx.py [--model MODEL_NAME] [--output OUTPUT_DIR]
"""

import argparse
import os
import shutil
from pathlib import Path

import torch
import yaml
from transformers import AutoTokenizer, AutoModelForMaskedLM


DEFAULT_MODEL = "shibing624/macbert4csc-base-chinese"
DEFAULT_OUTPUT = "output"
DEFAULT_CONFIG = "config/training_config.yaml"


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def export_onnx(model_name: str, output_dir: str, max_seq_len: int):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.eval()

    # Export vocab.txt
    vocab_src = Path(tokenizer.vocab_file) if hasattr(tokenizer, "vocab_file") else None
    vocab_dst = output_path / "csc-vocab.txt"
    if vocab_src and vocab_src.exists():
        shutil.copy2(vocab_src, vocab_dst)
        print(f"Copied vocab: {vocab_dst}")
    else:
        # Fallback: save tokenizer and extract vocab
        tokenizer.save_pretrained(str(output_path / "_tokenizer_tmp"))
        tmp_vocab = output_path / "_tokenizer_tmp" / "vocab.txt"
        if tmp_vocab.exists():
            shutil.move(str(tmp_vocab), str(vocab_dst))
        shutil.rmtree(output_path / "_tokenizer_tmp", ignore_errors=True)
        print(f"Saved vocab: {vocab_dst}")

    # Create dummy input
    dummy_text = "今天天气不错"
    inputs = tokenizer(
        dummy_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_seq_len,
    )

    onnx_path = output_path / "macbert4csc.onnx"

    print("Exporting to ONNX...")
    torch.onnx.export(
        model,
        (inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"]),
        str(onnx_path),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "token_type_ids": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=14,
        do_constant_folding=True,
    )

    # Verify ONNX model
    import onnx

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model exported and verified: {onnx_path}")
    print(f"Model size: {onnx_path.stat().st_size / 1024 / 1024:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Export MacBERT4CSC to ONNX")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Training config file")
    parser.add_argument("--model", default=None, help="HuggingFace model name")
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--max-seq-len", type=int, default=None, help="Maximum sequence length")
    args = parser.parse_args()

    config = load_config(args.config)
    model_name = args.model or config.get("base_model", DEFAULT_MODEL)
    output_dir = args.output or DEFAULT_OUTPUT
    max_seq_len = args.max_seq_len or config.get("max_seq_len", 128)

    export_onnx(model_name, output_dir, max_seq_len)


if __name__ == "__main__":
    main()
