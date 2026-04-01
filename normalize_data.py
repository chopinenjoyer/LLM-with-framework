from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path


def normalize_quotes_and_dashes(text: str) -> str:
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u00a0": " ",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def normalize_whitespace(text: str) -> str:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    cleaned_lines = [line for line in lines if line]
    return "\n".join(cleaned_lines)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = normalize_quotes_and_dashes(text)
    return normalize_whitespace(text)


def normalize_raw_file(input_path: Path, output_path: Path) -> int:
    text = input_path.read_text(encoding="utf-8")
    normalized = normalize_text(text)
    output_path.write_text(normalized + "\n", encoding="utf-8")
    return len(normalized)


def normalize_jsonl_file(input_path: Path, output_path: Path, dedupe: bool) -> int:
    seen: set[tuple[str, str]] = set()
    rows: list[str] = []
    count = 0
    for line in input_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        instruction = normalize_text(item["instruction"])
        response = normalize_text(item["response"])
        key = (instruction, response)
        if dedupe and key in seen:
            continue
        seen.add(key)
        rows.append(json.dumps({"instruction": instruction, "response": response}, ensure_ascii=False))
        count += 1
    output_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize raw text or JSONL instruction data.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--format", choices=["raw", "jsonl"], required=True)
    parser.add_argument("--dedupe", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "raw":
        size = normalize_raw_file(input_path, output_path)
        print(f"Normalized raw text to {output_path} ({size} chars)")
        return

    count = normalize_jsonl_file(input_path, output_path, dedupe=args.dedupe)
    print(f"Normalized jsonl to {output_path} ({count} examples)")


if __name__ == "__main__":
    main()
