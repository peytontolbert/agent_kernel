from __future__ import annotations

from pathlib import Path
import argparse
from datetime import UTC, datetime
import json
import shutil
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_swe_bench_predictions import _read_jsonl, validate_swe_predictions


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_live_predictions(records: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    failures = validate_swe_predictions(records)
    if failures:
        raise ValueError("invalid SWE predictions: " + "; ".join(failures))
    predictions: dict[str, dict[str, str]] = {}
    for record in records:
        instance_id = str(record["instance_id"]).strip()
        predictions[instance_id] = {
            "model_name_or_path": str(record["model_name_or_path"]).strip(),
            "model_patch": str(record["model_patch"]),
        }
    return predictions


def write_live_predictions_json(predictions_jsonl: Path, output_json: Path) -> dict[str, dict[str, str]]:
    predictions = build_live_predictions(_read_jsonl(predictions_jsonl))
    _write_json(output_json, predictions)
    return predictions


def _relative_or_absolute(path: Path) -> str:
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def build_submission_readme(
    *,
    model_name: str,
    system_name: str,
    subset: str,
    preds_json: Path,
    results_json: Path,
    run_command: str,
    prediction_count: int,
    generated_at: str | None = None,
) -> str:
    generated_at = generated_at or datetime.now(UTC).isoformat()
    lines = [
        f"# {system_name} on SWE-bench Live {subset}",
        "",
        f"- System: {system_name}",
        f"- Model: {model_name}",
        f"- Subset: {subset}",
        f"- Prediction count: {prediction_count}",
        f"- Generated at: {generated_at}",
        f"- Predictions: `{_relative_or_absolute(preds_json)}`",
        f"- Results: `{_relative_or_absolute(results_json)}`",
        "",
        "## Evaluation Command",
        "",
        "```bash",
        run_command.strip() or "not recorded",
        "```",
        "",
        "## Autonomy Contract",
        "",
        "The kernel generated patch predictions and this submission package. Human work is limited to launching, monitoring, and submitting the final PR.",
        "",
    ]
    return "\n".join(lines)


def package_live_submission(
    *,
    predictions_jsonl: Path,
    results_json: Path,
    output_dir: Path,
    model_name: str,
    system_name: str,
    subset: str,
    run_command: str,
) -> dict[str, Any]:
    if not results_json.exists():
        raise FileNotFoundError(f"results_json does not exist: {results_json}")
    output_dir.mkdir(parents=True, exist_ok=True)
    preds_json = output_dir / "preds.json"
    copied_results = output_dir / "results.json"
    predictions = write_live_predictions_json(predictions_jsonl, preds_json)
    shutil.copyfile(results_json, copied_results)
    readme = build_submission_readme(
        model_name=model_name,
        system_name=system_name,
        subset=subset,
        preds_json=preds_json,
        results_json=copied_results,
        run_command=run_command,
        prediction_count=len(predictions),
    )
    readme_path = output_dir / "README.md"
    readme_path.write_text(readme, encoding="utf-8")
    return {
        "preds_json": str(preds_json),
        "results_json": str(copied_results),
        "readme_md": str(readme_path),
        "prediction_count": len(predictions),
    }


def _preds_mode(args: argparse.Namespace) -> None:
    predictions = write_live_predictions_json(Path(args.predictions_jsonl), Path(args.output_json))
    print(f"live_prediction_count={len(predictions)} output_json={args.output_json}")


def _readme_mode(args: argparse.Namespace) -> None:
    preds_json = Path(args.preds_json)
    payload = json.loads(preds_json.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("preds_json must be a JSON object")
    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(
        build_submission_readme(
            model_name=args.model_name,
            system_name=args.system_name,
            subset=args.subset,
            preds_json=preds_json,
            results_json=Path(args.results_json),
            run_command=args.run_command,
            prediction_count=len(payload),
        ),
        encoding="utf-8",
    )
    print(f"readme_md={output_md}")


def _package_mode(args: argparse.Namespace) -> None:
    manifest = package_live_submission(
        predictions_jsonl=Path(args.predictions_jsonl),
        results_json=Path(args.results_json),
        output_dir=Path(args.output_dir),
        model_name=args.model_name,
        system_name=args.system_name,
        subset=args.subset,
        run_command=args.run_command,
    )
    print(
        f"submission_dir={args.output_dir} "
        f"prediction_count={manifest['prediction_count']} "
        f"preds_json={manifest['preds_json']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    preds_parser = subparsers.add_parser("preds")
    preds_parser.add_argument("--predictions-jsonl", required=True)
    preds_parser.add_argument("--output-json", required=True)
    preds_parser.set_defaults(func=_preds_mode)

    readme_parser = subparsers.add_parser("readme")
    readme_parser.add_argument("--preds-json", required=True)
    readme_parser.add_argument("--results-json", required=True)
    readme_parser.add_argument("--output-md", required=True)
    readme_parser.add_argument("--model-name", required=True)
    readme_parser.add_argument("--system-name", default="Agent Kernel")
    readme_parser.add_argument("--subset", required=True)
    readme_parser.add_argument("--run-command", default="")
    readme_parser.set_defaults(func=_readme_mode)

    package_parser = subparsers.add_parser("package")
    package_parser.add_argument("--predictions-jsonl", required=True)
    package_parser.add_argument("--results-json", required=True)
    package_parser.add_argument("--output-dir", required=True)
    package_parser.add_argument("--model-name", required=True)
    package_parser.add_argument("--system-name", default="Agent Kernel")
    package_parser.add_argument("--subset", required=True)
    package_parser.add_argument("--run-command", default="")
    package_parser.set_defaults(func=_package_mode)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
