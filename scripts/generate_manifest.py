"""
Generate manifest.json with version info and SHA256 checksums for all model artifacts.

Usage:
    python scripts/generate_manifest.py [--output OUTPUT_DIR] [--version VERSION]
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_OUTPUT = "output"

FILES = [
    "csc-macbert-int8.onnx",
    "csc-vocab.txt",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_manifest(output_dir: str, version: str):
    output_path = Path(output_dir)

    files_info = []
    for filename in FILES:
        filepath = output_path / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Required file not found: {filepath}")

        file_hash = sha256_file(filepath)
        file_size = filepath.stat().st_size

        files_info.append(
            {
                "filename": filename,
                "sha256": file_hash,
                "size": file_size,
            }
        )
        print(f"  {filename}: sha256={file_hash[:16]}... size={file_size}")

    manifest = {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_model": "shibing624/macbert4csc-base-chinese",
        "quantization": "dynamic_int8",
        "max_seq_len": 128,
        "files": files_info,
    }

    manifest_path = output_path / "csc-manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nManifest written: {manifest_path}")
    print(f"Model SHA256: {files_info[0]['sha256']}")

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Generate model manifest")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output directory")
    parser.add_argument("--version", default="1.0.0", help="Model version string")
    args = parser.parse_args()

    generate_manifest(args.output, args.version)


if __name__ == "__main__":
    main()
