from __future__ import annotations

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_kernel.config import KernelConfig
from agent_kernel.ops.vllm_runtime import ensure_vllm_runtime


def main() -> int:
    config = KernelConfig()
    config.ensure_directories()
    status = ensure_vllm_runtime(config)
    print(json.dumps(status.to_dict(), indent=2, sort_keys=True))
    return 0 if status.ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
