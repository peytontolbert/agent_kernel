from __future__ import annotations

import argparse
from pathlib import Path
import shutil


def clear_directory(directory: Path, *, require_under: Path) -> dict[str, object]:
    target = directory.resolve()
    root = require_under.resolve()
    if target == root or root not in target.parents:
        raise ValueError(f"refusing to clear {target}; it is not under {root}")
    target.mkdir(parents=True, exist_ok=True)
    removed: list[str] = []
    for child in target.iterdir():
        removed.append(str(child))
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    return {"directory": str(target), "removed_count": len(removed), "removed": removed}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    parser.add_argument("--require-under", required=True)
    args = parser.parse_args()

    result = clear_directory(Path(args.directory), require_under=Path(args.require_under))
    print(
        f"cleared_directory={result['directory']} "
        f"removed_count={result['removed_count']}"
    )


if __name__ == "__main__":
    main()
