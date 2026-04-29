from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


EXPECTED_LINES = {
    "leftist_definitions": [
        "greedy_components=leftmost,pure,eager",
        "active_index_from_p=p-1",
        "rewrite_start=p-len(alpha)",
        "rewrite_output=u1+beta+u2",
        "left_action=symbols insert or erase another symbol to their left while remaining unchanged",
    ],
    "leftist_semantics": [
        "case1_u1=xx",
        "case1_alpha=ab",
        "case1_u2=yy",
        "case1_rewritten=xxZyy",
        "case1_active_index=3",
        "case2_u1=abxx",
        "case2_u2=yy",
        "case2_rewritten=abxxZyy",
        "case2_active_index=5",
        "invalid_case=ValueError",
    ],
    "hamiltonicity_runtime": [
        "algorithm_type=Monte Carlo",
        "problem=undirected Hamiltonicity detection",
        "runtime=O*(1.657^n)",
    ],
    "farad_traceability": [
        "bridges=resistance ratio bridge,quadrature bridge",
        "frequency=1541 Hz",
        "farad_uncertainty=64E-9",
        "comparison_uncertainty=420E-9",
    ],
    "farad_table3": [
        "ten_pf_group=400",
        "ten_to_100_pf_stepup=40",
        "hundred_to_1000_pf_stepup=100",
        "new_traceability_chain=64",
        "rss=419",
    ],
    "herding_content": [
        "cow_grouping=group adjacent cows into clusters",
        "target_delegation=the leader assigns targets to each herder including itself",
        "fence_timing=the leader opens or coordinates the fence when cows are fleeing the right way",
    ],
    "fence_content": [
        "fence_timing=the leader opens or coordinates the fence when cows are fleeing the right way",
    ],
}


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _verify_cpython_reserved(path: Path) -> None:
    module = _load_module(path, "cpython_reserved_probe")
    fn = getattr(module, "is_reserved_name")
    assert fn(".") is False
    assert fn("..") is False
    assert fn("normal_name") is False
    assert fn("file.") is True
    assert fn("file ") is True
    assert fn("has|pipe") is True
    assert fn("name:stream") is True
    assert fn("nul") is True
    assert fn("nul .txt") is True


def _verify_transformers_flash(path: Path) -> None:
    module = _load_module(path, "transformers_flash_probe")
    top_left = getattr(module, "top_left_mask_supported")
    available = getattr(module, "flash_attention_available")
    assert top_left(fa3=True, fa2=True, fa2_ge_2_10=False, npu_top_left=True) is False
    assert top_left(fa3=False, fa2=True, fa2_ge_2_10=False, npu_top_left=False) is True
    assert top_left(fa3=False, fa2=True, fa2_ge_2_10=True, npu_top_left=True) is False
    assert top_left(fa3=False, fa2=False, fa2_ge_2_10=False, npu_top_left=True) is True
    assert available(fa3=True, fa2=False, npu=False) is True
    assert available(fa3=False, fa2=True, npu=False) is True
    assert available(fa3=False, fa2=False, npu=True) is True
    assert available(fa3=False, fa2=False, npu=False) is False


def _verify_pytorch_str2bool(path: Path) -> None:
    module = _load_module(path, "pytorch_str2bool_probe")
    fn = getattr(module, "str2bool")
    for value in (None, ""):
        assert fn(value) is False
    for value in ("1", "TRUE", " t ", "yes", "Y", "on", "enable", "enabled", "found"):
        assert fn(value) is True
    for value in ("0", "FALSE", " f ", "no", "N", "off", "disable", "disabled", "notfound", "none", "null", "nil", "undefined", "n/a"):
        assert fn(value) is False
    for value in ("  ", "maybe", 1):
        try:
            fn(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {value!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify research-library probe artifacts.")
    parser.add_argument("--case", required=True)
    parser.add_argument("--path", type=Path, required=True)
    args = parser.parse_args()

    if args.case == "repo_cpython_reserved":
        _verify_cpython_reserved(args.path)
        return
    if args.case == "repo_transformers_flash":
        _verify_transformers_flash(args.path)
        return
    if args.case == "repo_pytorch_str2bool":
        _verify_pytorch_str2bool(args.path)
        return

    expected = EXPECTED_LINES.get(args.case)
    if expected is None:
        raise SystemExit(f"unknown probe case: {args.case}")
    lines = [line.strip() for line in args.path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if lines != expected:
        raise SystemExit(f"unexpected lines: {lines!r}")


if __name__ == "__main__":
    main()
