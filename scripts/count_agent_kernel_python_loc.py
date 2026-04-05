#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def count_python_loc(root: Path) -> tuple[int, int]:
    file_count = 0
    line_count = 0

    for path in sorted(root.rglob("*.py")):
        if not path.is_file():
            continue
        file_count += 1
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                stripped = raw_line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                line_count += 1

    return file_count, line_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count non-blank, non-comment Python lines under a directory.",
    )
    parser.add_argument(
        "root",
        nargs="?",
        default="agent_kernel",
        help="Directory to scan. Defaults to agent_kernel.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"Path does not exist: {root}")
    if not root.is_dir():
        raise SystemExit(f"Path is not a directory: {root}")

    file_count, line_count = count_python_loc(root)
    print(f"root: {root}")
    print(f"python_files: {file_count}")
    print(f"python_loc: {line_count}")


if __name__ == "__main__":
    main()
