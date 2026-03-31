from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
from pathlib import Path


def kernels_root() -> Path:
    return Path(__file__).resolve().parent


def source_root() -> Path:
    return kernels_root() / "src"


def build_directory() -> Path:
    return kernels_root() / ".build"


def extension_name() -> str:
    return "agent_kernel_causal_belief_scan_cuda"


def source_paths() -> list[Path]:
    return [
        source_root() / "causal_belief_scan.cpp",
        source_root() / "causal_belief_scan_cuda.cu",
    ]


def build_command() -> str:
    return "python -m agent_kernel.modeling.world.kernels.build"


def build_metadata() -> dict[str, object]:
    return {
        "extension_name": extension_name(),
        "sources": [str(path) for path in source_paths()],
        "source_root": str(source_root()),
        "build_directory": str(build_directory()),
        "build_command": build_command(),
    }


def build_extension(*, verbose: bool = True):
    import torch
    from torch.utils.cpp_extension import load

    build_dir = build_directory()
    build_dir.mkdir(parents=True, exist_ok=True)
    with _build_lock(build_dir / ".build.lock"):
        extra_cflags = ["-O3", "-std=c++17"]
        extra_cuda_cflags = [
            "-O3",
            "-std=c++17",
            "-lineinfo",
        ]
        return load(
            name=extension_name(),
            sources=[str(path) for path in source_paths()],
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            build_directory=str(build_dir),
            verbose=verbose,
            with_cuda=True,
            is_python_module=True,
        )


@contextmanager
def _build_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w", encoding="utf-8") as handle:
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        _lock_file(handle)
        try:
            yield
        finally:
            _unlock_file(handle)


def _lock_file(handle) -> None:
    try:
        import fcntl
    except ImportError:
        return
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _unlock_file(handle) -> None:
    try:
        import fcntl
    except ImportError:
        return
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    metadata = build_metadata()
    print(f"extension_name={metadata['extension_name']}")
    print(f"source_root={metadata['source_root']}")
    for path in metadata["sources"]:
        print(f"source={path}")
    print(f"build_command={metadata['build_command']}")
    if args.metadata_only:
        return
    module = build_extension(verbose=not args.quiet)
    print(f"built_module={getattr(module, '__name__', '')}")


if __name__ == "__main__":
    main()
