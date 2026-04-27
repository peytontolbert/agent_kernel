from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_kernel.research_library import (
    DEFAULT_RESEARCH_LIBRARY_CONFIG,
    write_research_library_status,
)


DEFAULT_OUTPUT = Path("var/research_library/status.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a runtime status snapshot for agent-kernel research library sources."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_RESEARCH_LIBRARY_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    payload = write_research_library_status(
        args.output,
        config_path=args.config,
        root=args.root,
    )
    summary = payload["summary"]
    print(
        "sources={source_count} available={available_count} partial={partial_count} "
        "missing={missing_count} papers={paper_rows} repos={repository_count} "
        "models={trained_model_assets} output={output}".format(
            output=args.output,
            **summary,
        )
    )


if __name__ == "__main__":
    main()
