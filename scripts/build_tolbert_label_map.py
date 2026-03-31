from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_label_map(source_paths: list[Path], model_paths: list[Path]) -> dict[str, dict[str, int]]:
    if len(source_paths) != len(model_paths):
        raise RuntimeError("source/model path counts do not align.")
    mapping: dict[int, dict[int, int]] = {}
    for source_path, model_path in zip(source_paths, model_paths, strict=True):
        with source_path.open("r", encoding="utf-8") as source_handle, model_path.open(
            "r",
            encoding="utf-8",
        ) as model_handle:
            line_number = 0
            while True:
                source_line = source_handle.readline()
                model_line = model_handle.readline()
                if not source_line and not model_line:
                    break
                line_number += 1
                if not source_line or not model_line:
                    raise RuntimeError(
                        f"Source/model file lengths do not align at line {line_number}: "
                        f"{source_path} vs {model_path}"
                    )
                if not source_line.strip() and not model_line.strip():
                    continue
                if not source_line.strip() or not model_line.strip():
                    raise RuntimeError(
                        f"Source/model blank-line alignment mismatch at line {line_number}: "
                        f"{source_path} vs {model_path}"
                    )
                source = json.loads(source_line)
                model = json.loads(model_line)
                span_id = str(source.get("span_id", ""))
                model_span_id = str(model.get("span_id", ""))
                if span_id != model_span_id:
                    raise RuntimeError(
                        f"Source/model span_id mismatch at line {line_number}: "
                        f"{span_id!r} vs {model_span_id!r}"
                    )
                source_node_path = source.get("node_path")
                model_node_path = model.get("node_path")
                if not isinstance(source_node_path, list) or not isinstance(model_node_path, list):
                    raise RuntimeError(f"Span {span_id} is missing node_path.")
                if len(source_node_path) != len(model_node_path):
                    raise RuntimeError(f"Span {span_id} has mismatched path lengths.")
                for level, (source_node_id, model_node_id) in enumerate(
                    zip(source_node_path, model_node_path)
                ):
                    if level == 0:
                        continue
                    mapping.setdefault(level, {})
                    local_id = int(model_node_id)
                    global_id = int(source_node_id)
                    prior = mapping[level].get(local_id)
                    if prior is not None and prior != global_id:
                        raise RuntimeError(
                            f"Inconsistent mapping at level {level}: local class {local_id} "
                            f"maps to both {prior} and {global_id}."
                        )
                    mapping[level][local_id] = global_id

    return {
        str(level): {str(local_id): global_id for local_id, global_id in sorted(values.items())}
        for level, values in sorted(mapping.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-spans", nargs="+", required=True)
    parser.add_argument("--model-spans", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    source_paths = [Path(value) for value in args.source_spans]
    model_paths = [Path(value) for value in args.model_spans]
    label_map = build_label_map(source_paths, model_paths)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(label_map, indent=2), encoding="utf-8")
    print(f"wrote={out_path}")


if __name__ == "__main__":
    main()
