# 文件职责：验证社区提交的纠错数据文件的格式与内容质量。
"""
Validate community-submitted correction data files.

Checks format, field presence, and basic content quality.
Used by CI (validate-pr.yml) and locally before submitting.

Usage:
    python scripts/validate_submissions.py [FILES...]
    python scripts/validate_submissions.py datasets/submissions/user_abc_1234.jsonl
"""

import json
import sys
from pathlib import Path

MAX_CONTEXT_LEN = 50  # Max characters per input/output field
MAX_FILE_SIZE = 1024 * 1024  # 1 MB per file
REQUIRED_FIELDS = {"input", "output"}


def validate_file(filepath: Path) -> list[str]:
    """Validate a single JSONL submission file. Returns list of error messages."""
    errors = []

    if not filepath.exists():
        return [f"File not found: {filepath}"]

    if filepath.stat().st_size > MAX_FILE_SIZE:
        return [f"File too large: {filepath} ({filepath.stat().st_size} bytes, max {MAX_FILE_SIZE})"]

    if filepath.suffix != ".jsonl":
        return [f"Expected .jsonl extension: {filepath}"]

    with open(filepath, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: Invalid JSON: {e}")
                continue

            if not isinstance(obj, dict):
                errors.append(f"Line {line_num}: Expected JSON object, got {type(obj).__name__}")
                continue

            # Check required fields
            missing = REQUIRED_FIELDS - set(obj.keys())
            if missing:
                errors.append(f"Line {line_num}: Missing fields: {missing}")
                continue

            input_text = obj["input"]
            output_text = obj["output"]

            if not isinstance(input_text, str) or not isinstance(output_text, str):
                errors.append(f"Line {line_num}: 'input' and 'output' must be strings")
                continue

            if len(input_text) == 0 or len(output_text) == 0:
                errors.append(f"Line {line_num}: Empty input or output")
                continue

            if len(input_text) > MAX_CONTEXT_LEN:
                errors.append(
                    f"Line {line_num}: input too long ({len(input_text)} chars, max {MAX_CONTEXT_LEN})"
                )

            if len(output_text) > MAX_CONTEXT_LEN:
                errors.append(
                    f"Line {line_num}: output too long ({len(output_text)} chars, max {MAX_CONTEXT_LEN})"
                )

            # Input and output should have the same length (char-level correction)
            if len(input_text) != len(output_text):
                errors.append(
                    f"Line {line_num}: input/output length mismatch "
                    f"({len(input_text)} vs {len(output_text)}). "
                    f"CSC corrections should not change text length."
                )

            # Input and output should differ
            if input_text == output_text:
                errors.append(f"Line {line_num}: input and output are identical (no correction)")

    return errors


def main():
    if len(sys.argv) < 2:
        # Default: validate all files in datasets/submissions/
        submission_dir = Path("datasets/submissions")
        if not submission_dir.exists():
            print("No datasets/submissions/ directory found.")
            sys.exit(0)
        files = sorted(submission_dir.glob("*.jsonl"))
        if not files:
            print("No .jsonl files found in datasets/submissions/")
            sys.exit(0)
    else:
        files = [Path(f) for f in sys.argv[1:]]

    total_errors = 0
    for filepath in files:
        errors = validate_file(filepath)
        if errors:
            print(f"\n✗ {filepath}:")
            for err in errors:
                print(f"  - {err}")
            total_errors += len(errors)
        else:
            print(f"✓ {filepath}")

    if total_errors > 0:
        print(f"\n{total_errors} error(s) found.")
        sys.exit(1)
    else:
        print(f"\nAll {len(files)} file(s) valid ✓")
        sys.exit(0)


if __name__ == "__main__":
    main()
