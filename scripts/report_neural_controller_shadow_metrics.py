from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json

from agent_kernel.config import KernelConfig
from agent_kernel.neural_controller import (
    neural_controller_shadow_promotion_readiness,
    summarize_neural_controller_shadow_documents,
)
from agent_kernel.ops.episode_store import iter_episode_documents


def _normalize_path_filter(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return str(Path(text).expanduser().resolve())


def _shadow_manifest_path(shadow: object) -> str:
    if not isinstance(shadow, dict):
        return ""
    return _normalize_path_filter(shadow.get("manifest_path", ""))


def _filter_documents_by_shadow_manifest(
    documents: list[dict[str, object]],
    *,
    manifest_path: str,
) -> list[dict[str, object]]:
    normalized_manifest = _normalize_path_filter(manifest_path)
    if not normalized_manifest:
        return documents
    filtered: list[dict[str, object]] = []
    for document in documents:
        steps = _filter_step_list_by_shadow_manifest(document.get("steps", []), normalized_manifest)
        trace = _filter_policy_trace_by_shadow_manifest(document.get("policy_trace", []), normalized_manifest)
        if not steps and not trace:
            continue
        item: dict[str, object] = {}
        if steps:
            item["steps"] = steps
        if trace:
            item["policy_trace"] = trace
        filtered.append(item)
    return filtered


def _filter_step_list_by_shadow_manifest(steps: object, manifest_path: str) -> list[dict[str, object]]:
    if not isinstance(steps, list):
        return []
    filtered: list[dict[str, object]] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        metadata = step.get("proposal_metadata", {})
        if not isinstance(metadata, dict):
            continue
        shadow = metadata.get("neural_controller_shadow", {})
        if _shadow_manifest_path(shadow) != manifest_path:
            continue
        filtered.append(step)
    return filtered


def _filter_policy_trace_by_shadow_manifest(policy_trace: object, manifest_path: str) -> list[dict[str, object]]:
    if not isinstance(policy_trace, list):
        return []
    filtered: list[dict[str, object]] = []
    for item in policy_trace:
        if not isinstance(item, dict):
            continue
        neural = item.get("neural_controller", {})
        if not isinstance(neural, dict):
            continue
        shadow = neural.get("shadow", {})
        if _shadow_manifest_path(shadow) != manifest_path:
            continue
        filtered.append(item)
    return filtered


def _iter_report_documents(reports_dir: Path) -> list[dict[str, object]]:
    documents: list[dict[str, object]] = []
    if not reports_dir.exists():
        return documents
    for path in sorted(reports_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            if payload.get("report_kind") == "neural_controller_shadow_dataset_eval" and isinstance(
                payload.get("documents"),
                list,
            ):
                documents.extend(item for item in payload["documents"] if isinstance(item, dict))
            else:
                documents.append(payload)
    return documents


def _collect_shadow_manifest_paths(documents: list[dict[str, object]]) -> list[str]:
    paths: set[str] = set()
    for document in documents:
        for step in _document_shadow_steps(document):
            metadata = step.get("proposal_metadata", {})
            if not isinstance(metadata, dict):
                continue
            shadow = metadata.get("neural_controller_shadow", {})
            path = _shadow_manifest_path(shadow)
            if path:
                paths.add(path)
    return sorted(paths)


def _document_shadow_steps(document: dict[str, object]) -> list[dict[str, object]]:
    steps = document.get("steps", [])
    if isinstance(steps, list):
        out: list[dict[str, object]] = []
        for step in steps:
            if isinstance(step, dict):
                out.append(step)
        if out:
            return out
    policy_trace = document.get("policy_trace", [])
    if not isinstance(policy_trace, list):
        return []
    out = []
    for item in policy_trace:
        if not isinstance(item, dict):
            continue
        neural = item.get("neural_controller", {})
        if not isinstance(neural, dict):
            continue
        shadow = neural.get("shadow", {})
        if not isinstance(shadow, dict) or not shadow:
            continue
        out.append(
            {
                "proposal_metadata": {"neural_controller_shadow": shadow},
                "verification": {"passed": bool(item.get("verification_passed", False))},
            }
        )
    return out


def _manifest_breakdown(
    documents: list[dict[str, object]],
    *,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for manifest_path in _collect_shadow_manifest_paths(documents):
        filtered = _filter_documents_by_shadow_manifest(documents, manifest_path=manifest_path)
        summary = summarize_neural_controller_shadow_documents(filtered)
        if int(summary.get("shadow_steps") or 0) <= 0:
            continue
        readiness = neural_controller_shadow_promotion_readiness(
            summary,
            min_episodes=args.min_episodes,
            min_ready_steps=args.min_ready_steps,
            min_content_comparison_steps=args.min_content_comparison_steps,
            min_action_agreement_rate=args.min_action_agreement_rate,
            min_verified_action_agreement_rate=args.min_verified_action_agreement_rate,
            min_content_exact_agreement_rate=args.min_content_exact_agreement_rate,
            max_error_rate=args.max_error_rate,
            max_warning_rate=args.max_warning_rate,
        )
        rows.append(
            {
                "manifest_path": manifest_path,
                "summary": summary,
                "promotion_readiness": readiness,
            }
        )
    return sorted(
        rows,
        key=lambda item: (
            int(item.get("summary", {}).get("ready_steps") or 0),
            int(item.get("summary", {}).get("content_comparison_steps") or 0),
        ),
        reverse=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-root", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--write-config-output", action="store_true")
    parser.add_argument("--manifest-path", default="")
    parser.add_argument("--min-episodes", type=int, default=5)
    parser.add_argument("--min-ready-steps", type=int, default=25)
    parser.add_argument("--min-content-comparison-steps", type=int, default=5)
    parser.add_argument("--min-action-agreement-rate", type=float, default=0.70)
    parser.add_argument("--min-verified-action-agreement-rate", type=float, default=0.80)
    parser.add_argument("--min-content-exact-agreement-rate", type=float, default=0.80)
    parser.add_argument("--max-error-rate", type=float, default=0.0)
    parser.add_argument("--max-warning-rate", type=float, default=0.20)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    config = KernelConfig()
    episodes_root = Path(args.episodes_root) if args.episodes_root else config.trajectories_root
    documents = iter_episode_documents(episodes_root, config=None if args.episodes_root else config)
    if not args.episodes_root:
        documents.extend(_iter_report_documents(config.run_reports_dir))
    all_documents = list(documents)
    documents = _filter_documents_by_shadow_manifest(documents, manifest_path=args.manifest_path)
    summary = summarize_neural_controller_shadow_documents(documents)
    readiness = neural_controller_shadow_promotion_readiness(
        summary,
        min_episodes=args.min_episodes,
        min_ready_steps=args.min_ready_steps,
        min_content_comparison_steps=args.min_content_comparison_steps,
        min_action_agreement_rate=args.min_action_agreement_rate,
        min_verified_action_agreement_rate=args.min_verified_action_agreement_rate,
        min_content_exact_agreement_rate=args.min_content_exact_agreement_rate,
        max_error_rate=args.max_error_rate,
        max_warning_rate=args.max_warning_rate,
    )
    report = {
        "report_kind": "neural_controller_shadow_metrics",
        "episodes_root": str(episodes_root),
        "manifest_path_filter": str(args.manifest_path or "").strip(),
        "summary": summary,
        "promotion_readiness": readiness,
        "manifest_breakdown": _manifest_breakdown(all_documents, args=args),
    }

    output = args.output or (str(config.neural_controller_shadow_metrics_path) if args.write_config_output else "")
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    print(f"episodes_total={summary['episode_count']}")
    print(f"episodes_with_shadow={summary['episodes_with_shadow']}")
    print(f"shadow_steps={summary['shadow_steps']}")
    print(f"ready_steps={summary['ready_steps']}")
    print(f"action_agreement_steps={summary['action_agreement_steps']}")
    print(f"content_comparison_steps={summary.get('content_comparison_steps', 0)}")
    print(f"verified_ready_steps={summary['verified_ready_steps']}")
    print(f"verified_action_agreement_steps={summary['verified_action_agreement_steps']}")
    print(f"error_steps={summary['error_steps']}")
    print(f"warning_steps={summary['warning_steps']}")
    print(f"error_rate={summary['error_rate']:.3f}")
    print(f"warning_rate={summary['warning_rate']:.3f}")
    print(f"ready_rate={summary['ready_rate']:.3f}")
    print(f"action_agreement_rate={summary['action_agreement_rate']:.3f}")
    print(f"verified_action_agreement_rate={summary['verified_action_agreement_rate']:.3f}")
    print(f"content_exact_agreement_rate={summary.get('content_exact_agreement_rate', 0.0):.3f}")
    print(
        "unrepaired_content_exact_agreement_rate="
        f"{summary.get('unrepaired_content_exact_agreement_rate', 0.0):.3f}"
    )
    print(f"command_copy_target_repaired_rate={summary.get('command_copy_target_repaired_rate', 0.0):.3f}")
    if report["manifest_breakdown"]:
        print("manifest_breakdown:")
        for item in report["manifest_breakdown"][:8]:
            item_summary = item.get("summary", {})
            item_readiness = item.get("promotion_readiness", {})
            print(
                "  "
                f"ready_steps={item_summary.get('ready_steps', 0)} "
                f"content_comparison_steps={item_summary.get('content_comparison_steps', 0)} "
                f"content_exact_rate={float(item_summary.get('content_exact_agreement_rate') or 0.0):.3f} "
                f"warning_rate={float(item_summary.get('warning_rate') or 0.0):.3f} "
                f"shadow_compare_ready={str(item_readiness.get('shadow_compare_ready', False)).lower()} "
                f"manifest={item.get('manifest_path', '')}"
            )
    print(
        "promotion_readiness "
        f"shadow_compare_ready={str(readiness['shadow_compare_ready']).lower()} "
        f"kernel_guarded_content_ready={str(readiness.get('kernel_guarded_content_ready', False)).lower()} "
        f"content_authority_ready={str(readiness['content_authority_ready']).lower()} "
        f"pure_content_authority_ready={str(readiness.get('pure_content_authority_ready', False)).lower()} "
        f"primary_authority_ready={str(readiness['primary_authority_ready']).lower()} "
        f"blockers={','.join(readiness['blockers']) or 'none'} "
        f"content_authority_blockers={','.join(readiness['content_authority_blockers']) or 'none'} "
        f"pure_content_authority_blockers={','.join(readiness.get('pure_content_authority_blockers', [])) or 'none'}"
    )


if __name__ == "__main__":
    main()
