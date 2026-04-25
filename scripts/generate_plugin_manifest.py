"""Generate per-platform plugin manifests for the CSC plugin.

For each platform directory under `plugins-output/v1/<platform>/`, walks the
contents and writes `csc-plugin-manifest.json` describing every shipped file
with size + SHA256. Used by the host application to verify downloads and to
detect when a re-download is needed after a plugin upgrade.

The CDN layout this targets:

    https://dl.zhongbai233.com/plugins/v1/<platform>/csc-plugin-manifest.json
    https://dl.zhongbai233.com/plugins/v1/<platform>/<file>

Usage:
    python scripts/generate_plugin_manifest.py \
        --root output/plugins/v1 \
        --version 1.0.0 \
        --abi 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_ROOT = "output/plugins/v1"
DEFAULT_VERSION = "1.0.0"
DEFAULT_ABI = 1
DEFAULT_BASE_URL = "https://dl.zhongbai233.com/plugins/v1"
MANIFEST_NAME = "csc-plugin-manifest.json"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_for_platform(
    platform_dir: Path,
    version: str,
    abi: int,
    base_url: str,
) -> dict:
    files = []
    for entry in sorted(platform_dir.iterdir()):
        if not entry.is_file():
            continue
        # Skip a previously-written manifest so we don't checksum ourselves.
        if entry.name == MANIFEST_NAME:
            continue
        files.append({
            "filename": entry.name,
            "sha256": sha256_of(entry),
            "size": entry.stat().st_size,
        })
    return {
        "version": version,
        "abi": abi,
        "platform": platform_dir.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_url": f"{base_url}/{platform_dir.name}",
        "files": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=DEFAULT_ROOT,
                        help="Plugin output root containing one folder per platform")
    parser.add_argument("--version", default=DEFAULT_VERSION,
                        help="Plugin version string")
    parser.add_argument("--abi", type=int, default=DEFAULT_ABI,
                        help="C ABI version implemented by the plugin")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL,
                        help="CDN base URL for the manifest entries")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        return 1

    written = 0
    for platform_dir in sorted(root.iterdir()):
        if not platform_dir.is_dir():
            continue
        manifest = manifest_for_platform(
            platform_dir, args.version, args.abi, args.base_url
        )
        if not manifest["files"]:
            print(f"  Skip empty: {platform_dir.name}")
            continue
        out = platform_dir / MANIFEST_NAME
        out.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"  Wrote {out}  ({len(manifest['files'])} files)")
        written += 1

    if written == 0:
        print("warning: no manifests written", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
