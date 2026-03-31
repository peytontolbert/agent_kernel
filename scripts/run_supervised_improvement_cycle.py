from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_guided_cycle_module():
    module_name = "_agentkernel_run_human_guided_improvement_cycle"
    loaded = sys.modules.get(module_name)
    if loaded is not None:
        return loaded
    module_path = Path(__file__).resolve().parent / "run_human_guided_improvement_cycle.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load supervised improvement cycle module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    _load_guided_cycle_module().main()


if __name__ == "__main__":
    main()
