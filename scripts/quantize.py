# 文件职责：将 ONNX 模型进行 INT8 动态量化以减小体积。
"""
Quantize ONNX model to INT8 (dynamic quantization).

Usage:
    python scripts/quantize.py [--input INPUT] [--output OUTPUT_DIR]
"""

import argparse
from pathlib import Path

from onnxruntime.quantization import quantize_dynamic, QuantType


DEFAULT_INPUT = "output/macbert4csc.onnx"
DEFAULT_OUTPUT_DIR = "output"


def quantize(input_path: str, output_dir: str):
    input_file = Path(input_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not input_file.exists():
        raise FileNotFoundError(f"Input model not found: {input_file}")

    output_file = output_path / "csc-macbert-int8.onnx"

    print(f"Quantizing: {input_file} → {output_file}")
    print(f"Input size: {input_file.stat().st_size / 1024 / 1024:.1f} MB")

    quantize_dynamic(
        model_input=str(input_file),
        model_output=str(output_file),
        weight_type=QuantType.QInt8,
    )

    print(f"Quantization complete: {output_file}")
    print(f"Output size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")

    # Verify quantized model
    import onnxruntime as ort

    session = ort.InferenceSession(str(output_file))
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]
    print(f"Inputs: {input_names}")
    print(f"Outputs: {output_names}")
    print("Verification passed ✓")


def main():
    parser = argparse.ArgumentParser(description="Quantize ONNX model to INT8")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input ONNX model path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    args = parser.parse_args()

    quantize(args.input, args.output)


if __name__ == "__main__":
    main()
