"""Extract ONNX Runtime native libraries from the Android AAR for redistribution.

The Android variant of the CSC plugin downloads `libonnxruntime.so` and
`libonnxruntime4j_jni.so` from the model CDN at runtime, instead of bundling
them in the APK. This script grabs the official Microsoft AAR from Maven
Central, unzips it and copies the per-ABI `.so` files into a layout that
matches the directory structure the Android side expects:

    output/plugins/v1/android-arm64-v8a/libonnxruntime.so
    output/plugins/v1/android-arm64-v8a/libonnxruntime4j_jni.so
    output/plugins/v1/android-x86_64/libonnxruntime.so
    output/plugins/v1/android-x86_64/libonnxruntime4j_jni.so

Usage:
    python scripts/extract_onnxruntime_android.py \
        --version 1.24.3 \
        --output output/plugins/v1
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

DEFAULT_VERSION = "1.24.3"
DEFAULT_OUTPUT = "output/plugins/v1"
MAVEN_BASE = "https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android"

# Android only ships these two ABIs in the modern AAR, and the host repo only
# enables `arm64-v8a` and `x86_64` ABI splits.
TARGET_ABIS = {
    "arm64-v8a": "android-arm64-v8a",
    "x86_64": "android-x86_64",
}

LIB_FILENAMES = ("libonnxruntime.so", "libonnxruntime4j_jni.so")


def download(url: str, dest: Path) -> None:
    print(f"  Downloading {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as resp, open(dest, "wb") as f:
        shutil.copyfileobj(resp, f)


def extract(version: str, output_root: Path, cache_dir: Path) -> None:
    aar_name = f"onnxruntime-android-{version}.aar"
    aar_url = f"{MAVEN_BASE}/{version}/{aar_name}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    aar_path = cache_dir / aar_name

    if not aar_path.exists() or aar_path.stat().st_size == 0:
        download(aar_url, aar_path)
    else:
        print(f"  Cached AAR: {aar_path}")

    # AARs are ZIPs. Native libs live under `jni/<abi>/`.
    with zipfile.ZipFile(aar_path) as zf:
        for abi, dirname in TARGET_ABIS.items():
            target_dir = output_root / dirname
            target_dir.mkdir(parents=True, exist_ok=True)
            for libname in LIB_FILENAMES:
                member = f"jni/{abi}/{libname}"
                try:
                    info = zf.getinfo(member)
                except KeyError:
                    raise SystemExit(
                        f"AAR is missing expected entry: {member} "
                        f"(version {version} may not include this ABI/lib)"
                    )
                dest = target_dir / libname
                with zf.open(info) as src, open(dest, "wb") as out:
                    shutil.copyfileobj(src, out)
                print(f"  Extracted {member} → {dest} ({info.file_size} bytes)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default=DEFAULT_VERSION,
                        help=f"onnxruntime-android version (default {DEFAULT_VERSION})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help=f"Output root (default {DEFAULT_OUTPUT})")
    parser.add_argument("--cache-dir", default=".cache/onnxruntime-android",
                        help="Where to cache the downloaded AAR")
    args = parser.parse_args()

    output_root = Path(args.output)
    cache_dir = Path(args.cache_dir)
    print(f"== Extracting onnxruntime-android {args.version} ==")
    extract(args.version, output_root, cache_dir)
    print(f"\nDone. Output: {output_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
