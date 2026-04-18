import json

from agent_kernel.config import KernelConfig
from agent_kernel.extensions.improvement.artifacts import stamp_artifact_generation_context
from agent_kernel.reflection import build_reflection_record
from agent_kernel.resource_registry import runtime_resource_registry
from agent_kernel.resource_types import subsystem_resource_id
from agent_kernel.selection import build_selection_record
from evals.metrics import EvalMetrics


def _write_policy_artifact(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "spec_version": "asi_v1",
                "artifact_kind": "prompt_proposal_set",
                "lifecycle_state": "retained",
                "lineage": {"parent_version_id": "sha256:baseline"},
                "proposals": [
                    {
                        "area": "decision",
                        "priority": 5,
                        "suggestion": "verify before mutating shared files",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_build_reflection_record_targets_resource_ids(tmp_path):
    prompt_policy_path = tmp_path / "trajectories" / "prompts" / "prompt_proposals.json"
    _write_policy_artifact(prompt_policy_path)
    config = KernelConfig(provider="mock", prompt_proposals_path=prompt_policy_path)
    registry = runtime_resource_registry(config)

    record = build_reflection_record(
        cycle_id="cycle:policy:1",
        target_subsystem="policy",
        reason="decision conversion remains weak",
        metrics=EvalMetrics(
            total=6,
            passed=3,
            generated_total=2,
            generated_passed=1,
            low_confidence_episodes=2,
            trusted_retrieval_steps=1,
        ),
        evidence={
            "transition_failure_counts": {"no_state_progress": 3},
            "failure_counts": {"command_failure": 2},
            "portfolio": {"reasons": ["recent_reject", "generated_gap"]},
        },
        observe_hypothesis={
            "status": "generated",
            "provider": "vllm",
            "summary": "policy under-converts decisions",
            "hypotheses": [
                {
                    "subsystem": "policy",
                    "confidence": 0.75,
                    "rationale": "decision conversion stalls after low-confidence traces",
                }
            ],
        },
        strategy_candidate={
            "strategy_candidate_id": "strategy:subsystem:policy",
            "strategy_candidate_kind": "subsystem_direct",
            "origin": "authored_strategy",
            "expected_signals": ["retained_gain"],
            "generation_basis": {
                "repeated_failure_motifs": ["no_state_progress"],
                "transfer_gaps": ["generated_task_transfer_gap"],
            },
        },
        resource_registry=registry,
    )

    payload = record.to_dict()

    assert payload["resource_id"] == subsystem_resource_id("policy")
    assert payload["resource_kind"] == "artifact"
    assert payload["summary"] == "policy under-converts decisions"
    assert payload["observe_status"] == "generated"
    assert payload["observe_provider"] == "vllm"
    assert payload["active_version_id"].startswith("sha256:")
    assert payload["hypothesis"]["source"] == "observe_hypothesis"
    assert payload["hypothesis"]["failure_modes"] == ["no_state_progress", "command_failure"]
    assert "generated_task_transfer_gap" in payload["hypothesis"]["evidence_labels"]
    assert payload["hypothesis"]["expected_signals"] == ["retained_gain"]


def test_build_selection_record_includes_edit_plan_and_search_context(tmp_path):
    prompt_policy_path = tmp_path / "trajectories" / "prompts" / "prompt_proposals.json"
    _write_policy_artifact(prompt_policy_path)
    config = KernelConfig(provider="mock", prompt_proposals_path=prompt_policy_path)
    registry = runtime_resource_registry(config)

    record = build_selection_record(
        cycle_id="cycle:policy:1",
        target_subsystem="policy",
        reason="decision conversion remains weak",
        variant_id="decision_caution",
        variant_description="bias toward verifier-backed command selection",
        variant_expected_gain=0.12,
        variant_estimated_cost=2,
        variant_score=0.06,
        variant_controls={"focus": "decision"},
        generation_kwargs={"focus": "decision"},
        expected_artifact_kind="prompt_proposal_set",
        strategy_candidate={
            "strategy_candidate_id": "strategy:subsystem:policy",
            "strategy_candidate_kind": "subsystem_direct",
            "origin": "authored_strategy",
            "strategy_label": "policy_direct",
            "strategy_node_id": "node:123",
            "portfolio_reasons": ["recent_reject", "generated_gap"],
        },
        campaign_index=1,
        campaign_width=2,
        variant_rank=1,
        variant_width=3,
        search_strategy="adaptive_history",
        campaign_budget={"width": 2, "max_width": 3},
        variant_budget={"width": 3, "max_width": 3},
        resource_registry=registry,
    )

    payload = record.to_dict()

    assert payload["resource_id"] == subsystem_resource_id("policy")
    assert payload["selected_variant"]["variant_id"] == "decision_caution"
    assert payload["strategy"]["strategy_node_id"] == "node:123"
    assert payload["portfolio_reasons"] == ["recent_reject", "generated_gap"]
    assert payload["search"]["campaign_width"] == 2
    assert payload["search"]["variant_width"] == 3
    assert payload["edit_plan"]["resource_id"] == subsystem_resource_id("policy")
    assert payload["edit_plan"]["expected_artifact_kind"] == "prompt_proposal_set"
    assert payload["edit_plan"]["controls"] == {"focus": "decision"}
    assert payload["edit_plan"]["generation_kwargs"] == {"focus": "decision"}
    assert payload["edit_plan"]["active_version_id"].startswith("sha256:")


def test_stamp_artifact_generation_context_merges_protocol_context(tmp_path):
    artifact_path = tmp_path / "candidate.json"
    artifact_path.write_text(
        json.dumps(
            {
                "spec_version": "asi_v1",
                "artifact_kind": "prompt_proposal_set",
                "lifecycle_state": "candidate",
            }
        ),
        encoding="utf-8",
    )
    active_path = tmp_path / "active.json"
    active_path.write_text("{}", encoding="utf-8")

    stamp_artifact_generation_context(
        artifact_path,
        cycle_id="cycle:policy:1",
        active_artifact_path=active_path,
        candidate_artifact_path=artifact_path,
        extra_context={
            "reflection_id": "reflect:cycle:policy:1",
            "selection_id": "select:cycle:policy:1",
            "resource_id": "subsystem:policy",
            "selection_record": {"selection_id": "select:cycle:policy:1"},
        },
    )

    payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert payload["generation_context"]["cycle_id"] == "cycle:policy:1"
    assert payload["generation_context"]["active_artifact_path"] == str(active_path)
    assert payload["generation_context"]["candidate_artifact_path"] == str(artifact_path)
    assert payload["generation_context"]["reflection_id"] == "reflect:cycle:policy:1"
    assert payload["generation_context"]["selection_id"] == "select:cycle:policy:1"
    assert payload["generation_context"]["resource_id"] == "subsystem:policy"
    assert payload["generation_context"]["selection_record"]["selection_id"] == "select:cycle:policy:1"
