from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_kernel.config import KernelConfig
from agent_kernel.loop import AgentKernel
from agent_kernel.research_library import DEFAULT_RESEARCH_LIBRARY_CONFIG, write_research_library_status
from agent_kernel.schemas import EpisodeRecord, TaskSpec


DEFAULT_OUTPUT = Path("var/research_library/hard_probe_report.json")
DEFAULT_STATUS = Path("var/research_library/status.json")


def _source(status: dict[str, Any], source_id: str) -> dict[str, Any]:
    for source in status.get("sources", []):
        if isinstance(source, dict) and source.get("id") == source_id:
            return source
    return {}


def _load_status(*, repo_root: Path, status_path: Path, config_path: Path, refresh: bool) -> dict[str, Any]:
    if refresh or not status_path.exists():
        return write_research_library_status(status_path, config_path=config_path, root=repo_root)
    return json.loads(status_path.read_text(encoding="utf-8"))


def _algorithm_record(status: dict[str, Any], algo_id: str) -> dict[str, Any]:
    algorithms = _source(status, "algorithms")
    catalog_path = Path(str(algorithms.get("path", ""))) / "algorithms.jsonl"
    try:
        lines = catalog_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    for line in lines:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("algo_id") == algo_id:
            return payload
    return {}


def _paper_probe_row(status: dict[str, Any]) -> dict[str, Any]:
    source = _source(status, "paper_chunks_p1")
    path = Path(str(source.get("path", "")))
    files = sorted(path.glob("train-*.parquet"))
    if not files:
        return {}
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return {}
    try:
        table = pq.read_table(
            files[0],
            columns=["id", "paper_id", "title", "categories", "year", "chunk_index", "text"],
        )
    except Exception:
        return {}
    rows = table.slice(0, 1).to_pylist()
    return dict(rows[0]) if rows and isinstance(rows[0], dict) else {}


def _normalized_text(value: object) -> str:
    return " ".join(str(value).split())


def _paper_evidence_sentence(row: dict[str, Any]) -> str:
    text = _normalized_text(row.get("text", ""))
    marker = "The strategy is to be implemented"
    start = text.find(marker)
    if start < 0:
        abstract_start = text.find("Abstract:")
        start = abstract_start + len("Abstract:") if abstract_start >= 0 else 0
    end = text.find(".", start)
    if end < 0:
        return text[start : start + 220].strip()
    return text[start : end + 1].strip()


def _dijkstra_worst_complexity(status: dict[str, Any]) -> str:
    payload = _algorithm_record(status, "dijkstra")
    complexity = payload.get("time_complexity", {})
    if isinstance(complexity, dict):
        return str(complexity.get("worst", "")).strip()
    return ""


def _expected_answer(status: dict[str, Any]) -> str:
    summary = status.get("summary", {}) if isinstance(status.get("summary", {}), dict) else {}
    dijkstra_worst = _dijkstra_worst_complexity(status) or "UNKNOWN"
    return "\n".join(
        [
            f"paper_rows={int(summary.get('paper_rows', 0) or 0)}",
            f"trained_model_assets={int(summary.get('trained_model_assets', 0) or 0)}",
            f"dijkstra_worst={dijkstra_worst}",
            "",
        ]
    )


def _expected_content_answer(status: dict[str, Any]) -> str:
    payload = _algorithm_record(status, "dijkstras_two_stack_algorithm")
    topics = payload.get("topics", [])
    return "\n".join(
        [
            f"algo_id={payload.get('algo_id', '')}",
            f"category={payload.get('category', '')}",
            "topics=" + ",".join(str(topic) for topic in topics if str(topic).strip())
            if isinstance(topics, list)
            else "topics=",
            f"notes={payload.get('notes', '')}",
            "",
        ]
    )


def _expected_paper_answer(status: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    row = _paper_probe_row(status)
    answer = "\n".join(
        [
            f"paper_id={row.get('paper_id', '')}",
            f"title={row.get('title', '')}",
            f"evidence_sentence={_paper_evidence_sentence(row)}",
            "",
        ]
    )
    return answer, row


def _expected_leftist_semantics_answer() -> str:
    return "\n".join(
        [
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
            "",
        ]
    )


def _expected_leftist_definitions_answer() -> str:
    return "\n".join(
        [
            "greedy_components=leftmost,pure,eager",
            "active_index_from_p=p-1",
            "rewrite_start=p-len(alpha)",
            "rewrite_output=u1+beta+u2",
            "left_action=symbols insert or erase another symbol to their left while remaining unchanged",
            "",
        ]
    )


def _expected_hamiltonicity_runtime_answer() -> str:
    return "\n".join(
        [
            "algorithm_type=Monte Carlo",
            "problem=undirected Hamiltonicity detection",
            "runtime=O*(1.657^n)",
            "",
        ]
    )


def _expected_farad_traceability_answer() -> str:
    return "\n".join(
        [
            "bridges=resistance ratio bridge,quadrature bridge",
            "frequency=1541 Hz",
            "farad_uncertainty=64E-9",
            "comparison_uncertainty=420E-9",
            "",
        ]
    )


def _expected_farad_table3_answer() -> str:
    return "\n".join(
        [
            "ten_pf_group=400",
            "ten_to_100_pf_stepup=40",
            "hundred_to_1000_pf_stepup=100",
            "new_traceability_chain=64",
            "rss=419",
            "",
        ]
    )


def _herding_success_command() -> str:
    return (
        "python -c \"from pathlib import Path; "
        "t=Path('strategy.txt').read_text().lower(); "
        "assert 'adjacent' in t and ('cluster' in t or 'group' in t); "
        "assert 'leader' in t and ('delegate' in t or 'assign' in t or 'target' in t) "
        "and ('himself' in t or 'including self' in t or 'including itself' in t or 'including the leader' in t); "
        "assert 'fence' in t and 'open' in t and ('right way' in t or 'right direction' in t or 'fleeing' in t)\""
    )


def _fence_success_command() -> str:
    return (
        "python -c \"from pathlib import Path; "
        "t=Path('strategy.txt').read_text().lower(); "
        "assert 'leader' in t and 'fence' in t and 'open' in t; "
        "assert ('fleeing the right way' in t or 'flee in the right way' in t); "
        "assert 'wrong way' not in t\""
    )


def _leftist_rewrite_success_command() -> str:
    return (
        "python -c 'import importlib.util\n"
        "spec=importlib.util.spec_from_file_location(\"leftist_rewrite\", \"leftist_rewrite.py\")\n"
        "m=importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(m)\n"
        "assert m.rewrite_at_position(\"xxabyy\", \"ab\", \"Z\", 4) == \"xxZyy\"\n"
        "assert m.rewrite_at_position(\"abyy\", \"ab\", \"Z\", 2) == \"Zyy\"\n"
        "assert m.rewrite_at_position(\"xxab\", \"ab\", \"\", 4) == \"xx\"\n"
        "try:\n"
        "    m.rewrite_at_position(\"xxabyy\", \"ab\", \"Z\", 2)\n"
        "except ValueError:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError(\"expected ValueError\")'"
    )


def _leftist_window_success_command() -> str:
    return (
        "python -c 'import importlib.util\n"
        "spec=importlib.util.spec_from_file_location(\"leftist_window\", \"leftist_window.py\")\n"
        "m=importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(m)\n"
        "assert m.replacement_start(\"ab\", 4) == 2\n"
        "assert m.replacement_start(\"abc\", 3) == 0\n"
        "assert m.replacement_start(\"x\", 1) == 0\n"
        "try:\n"
        "    m.replacement_start(\"abcd\", 3)\n"
        "except ValueError:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError(\"expected ValueError\")'"
    )


def _leftist_step_success_command() -> str:
    return (
        "python -c 'import importlib.util\n"
        "spec=importlib.util.spec_from_file_location(\"leftist_step\", \"leftist_step.py\")\n"
        "m=importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(m)\n"
        "r = m.rewrite_step(\"xxabyy\", \"ab\", \"Z\", 4)\n"
        "assert r[\"u1\"] == \"xx\" and r[\"alpha\"] == \"ab\" and r[\"u2\"] == \"yy\"\n"
        "assert r[\"rewritten\"] == \"xxZyy\" and r[\"active_index\"] == 3\n"
        "r = m.rewrite_step(\"abxxabyy\", \"ab\", \"Z\", 6)\n"
        "assert r[\"u1\"] == \"abxx\" and r[\"u2\"] == \"yy\" and r[\"rewritten\"] == \"abxxZyy\"\n"
        "assert r[\"active_index\"] == 5\n"
        "r = m.rewrite_step(\"abxxabyy\", \"ab\", \"Z\", 2)\n"
        "assert r[\"u1\"] == \"\" and r[\"u2\"] == \"xxabyy\" and r[\"rewritten\"] == \"Zxxabyy\"\n"
        "assert m.rewrite_step(\"xxab\", \"ab\", \"\", 4)[\"rewritten\"] == \"xx\"\n"
        "for args in ((\"xxabyy\", \"ab\", \"Z\", 2), (\"xxabyy\", \"ab\", \"Z\", 7), (\"abcd\", \"abcd\", \"Z\", 3)):\n"
        "    try:\n"
        "        m.rewrite_step(*args)\n"
        "    except ValueError:\n"
        "        pass\n"
        "    else:\n"
        "        raise AssertionError(f\"expected ValueError for {args!r}\")'"
    )


def _repo_probe_verify_command(case: str, path: str) -> str:
    return f"python /data/agentkernel/scripts/verify_research_library_probe.py --case {case} --path {path}"


def _hard_task(expected_answer: str, *, probe: str, paper_row: dict[str, Any] | None = None) -> TaskSpec:
    if probe == "repo_cpython_reserved":
        return TaskSpec(
            task_id="research_library_repo_cpython_reserved_name",
            prompt=(
                "Hard repository-code probe. Create cpython_reserved.py in the workspace. "
                "Use the provided context evidence only; do not inspect local files, repository checkouts, /data, "
                "or the internet. The provided code context is from CPython's Lib/ntpath.py and contains the "
                "_isreservedname implementation. Implement is_reserved_name(name: str) -> bool with that CPython "
                "behavior, including the special handling of trailing dots/spaces, reserved punctuation and "
                "file stream separators, and DOS device names with suffixes. The verifier uses behavioral tests."
            ),
            workspace_subdir="research_library_repo_probe_cpython_reserved",
            expected_files=["cpython_reserved.py"],
            success_command=_repo_probe_verify_command("repo_cpython_reserved", "cpython_reserved.py"),
            max_steps=3,
            metadata={
                "benchmark_family": "research_library_repo_code_probe",
                "capability": "retrieval_grounded_repository_code_implementation",
                "difficulty": "hard",
                "requires_retrieval": True,
                "repo": "cpython",
                "code_symbol": "_isreservedname",
                "research_topic": (
                    "CPython Lib/ntpath.py _isreservedname trailing dots spaces reserved chars colon "
                    "file streams DOS device names nul suffix"
                ),
                "step_floor": 1,
            },
        )
    if probe == "repo_transformers_flash":
        return TaskSpec(
            task_id="research_library_repo_transformers_flash_attention",
            prompt=(
                "Hard repository-code probe. Create transformers_flash.py in the workspace. "
                "Use the provided context evidence only; do not inspect local files, repository checkouts, /data, "
                "or the internet. The provided code context is from Hugging Face Transformers' "
                "modeling_flash_attention_utils.py. Implement top_left_mask_supported(*, fa3: bool, fa2: bool, "
                "fa2_ge_2_10: bool, npu_top_left: bool) -> bool with the same branch semantics as "
                "flash_attn_supports_top_left_mask, and implement flash_attention_available(*, fa3: bool, "
                "fa2: bool, npu: bool) -> bool with the same semantics as is_flash_attn_available. "
                "The verifier uses behavioral tests."
            ),
            workspace_subdir="research_library_repo_probe_transformers_flash",
            expected_files=["transformers_flash.py"],
            success_command=_repo_probe_verify_command("repo_transformers_flash", "transformers_flash.py"),
            max_steps=3,
            metadata={
                "benchmark_family": "research_library_repo_code_probe",
                "capability": "retrieval_grounded_repository_code_implementation",
                "difficulty": "hard",
                "requires_retrieval": True,
                "repo": "transformers",
                "code_symbol": "flash_attn_supports_top_left_mask",
                "research_topic": (
                    "Transformers modeling_flash_attention_utils flash_attn_supports_top_left_mask "
                    "is_flash_attn_available flash attention 3 2 npu top left mask"
                ),
                "step_floor": 1,
            },
        )
    if probe == "repo_pytorch_str2bool":
        return TaskSpec(
            task_id="research_library_repo_pytorch_str2bool",
            prompt=(
                "Hard repository-code probe. Create pytorch_str2bool.py in the workspace. "
                "Use the provided context evidence only; do not inspect local files, repository checkouts, /data, "
                "or the internet. The provided code context is from PyTorch's setup.py str2bool helper. "
                "Implement str2bool(value) with the same conversion semantics, including empty values, non-string "
                "values, all truthy tokens, all falsy tokens, whitespace/case normalization, and invalid-token "
                "errors. The verifier uses behavioral tests."
            ),
            workspace_subdir="research_library_repo_probe_pytorch_str2bool",
            expected_files=["pytorch_str2bool.py"],
            success_command=_repo_probe_verify_command("repo_pytorch_str2bool", "pytorch_str2bool.py"),
            max_steps=3,
            metadata={
                "benchmark_family": "research_library_repo_code_probe",
                "capability": "retrieval_grounded_repository_code_implementation",
                "difficulty": "hard",
                "requires_retrieval": True,
                "repo": "pytorch",
                "code_symbol": "str2bool",
                "research_topic": (
                    "PyTorch setup.py str2bool truthy falsy tokens enable enabled found notfound nil undefined n/a"
                ),
                "step_floor": 1,
            },
        )
    if probe == "farad_table3":
        return TaskSpec(
            task_id="research_library_hard_farad_table3",
            prompt=(
                "Hard applied provided-context content probe. Create answer.txt in the workspace. "
                "Use the provided context evidence only; do not inspect local files, dataset paths, /data, /arxiv, "
                "or the internet. The provided context contains a metrology paper about a farad traceability "
                "chain using digitally assisted impedance bridges. In the comparison uncertainty table for the "
                "maintained capacitance national standard, extract the relative uncertainty entries for the "
                "10 pF maintained group value, the 10 pF to 100 pF step-up, the 100 pF to 1000 pF step-up, "
                "the new traceability chain, and the table RSS. Write exactly five newline-delimited fields: "
                "ten_pf_group=..., ten_to_100_pf_stepup=..., hundred_to_1000_pf_stepup=..., "
                "new_traceability_chain=..., and rss=.... Use a simple printf command that writes to the "
                "relative path answer.txt; do not use /workspace paths, heredoc syntax, python -c writers, "
                "or extra lookup commands."
            ),
            workspace_subdir="research_library_hard_probe_farad_table3",
            expected_files=["answer.txt"],
            success_command=(
                "python /data/agentkernel/scripts/verify_research_library_probe.py "
                "--case farad_table3 --path answer.txt"
            ),
            max_steps=2,
            metadata={
                "benchmark_family": "research_library_behavior_probe",
                "capability": "retrieval_grounded_paper_table_content",
                "difficulty": "hard",
                "requires_retrieval": True,
                "research_topic": (
                    "farad traceability comparison uncertainty table maintained capacitance national standard "
                    "10 pF 100 pF 1000 pF step-up RSS new traceability chain"
                ),
                "step_floor": 1,
            },
        )
    if probe == "farad_traceability":
        return TaskSpec(
            task_id="research_library_hard_farad_traceability",
            prompt=(
                "Hard applied provided-context content probe. Create answer.txt in the workspace. "
                "Use the provided context evidence only; do not inspect local files, dataset paths, /data, /arxiv, "
                "or the internet. The provided context contains a metrology paper about deriving the farad "
                "from the dc quantum Hall effect with digitally assisted impedance bridges. Extract the two main "
                "bridge components, their operating frequency, the relative uncertainty for realizing the farad at "
                "1000 pF, and the relative combined uncertainty for the first comparison with the maintained national "
                "capacitance standard. Write exactly four newline-delimited fields: bridges=..., frequency=..., "
                "farad_uncertainty=..., and comparison_uncertainty=.... Use a simple printf command that writes "
                "to the relative path answer.txt; do not use /workspace paths, heredoc syntax, python -c writers, "
                "or extra lookup commands."
            ),
            workspace_subdir="research_library_hard_probe_farad_traceability",
            expected_files=["answer.txt"],
            success_command=(
                "python /data/agentkernel/scripts/verify_research_library_probe.py "
                "--case farad_traceability --path answer.txt"
            ),
            max_steps=2,
            metadata={
                "benchmark_family": "research_library_behavior_probe",
                "capability": "retrieval_grounded_paper_content",
                "difficulty": "hard",
                "requires_retrieval": True,
                "research_topic": (
                    "farad dc quantum Hall effect digitally assisted impedance bridges 1541 Hz "
                    "1000 pF relative uncertainty capacitance standard"
                ),
                "step_floor": 1,
            },
        )
    if probe == "hamiltonicity_runtime":
        return TaskSpec(
            task_id="research_library_hard_hamiltonicity_runtime",
            prompt=(
                "Hard applied provided-context algorithm probe. Create answer.txt in the workspace. "
                "Use the provided context evidence only; do not inspect local files, dataset paths, /data, /arxiv, "
                "or the internet. The provided context contains a paper about determinant-sum techniques for "
                "undirected Hamiltonicity. Extract the algorithm type, the exact graph problem, and the asymptotic "
                "runtime stated in that paper content. Write exactly three newline-delimited fields: "
                "algorithm_type=..., problem=..., and runtime=.... Use a simple printf command that writes to the "
                "relative path answer.txt; do not use /workspace paths, heredoc syntax, python -c writers, "
                "or extra lookup commands."
            ),
            workspace_subdir="research_library_hard_probe_hamiltonicity_runtime",
            expected_files=["answer.txt"],
            success_command=(
                "python /data/agentkernel/scripts/verify_research_library_probe.py "
                "--case hamiltonicity_runtime --path answer.txt"
            ),
            max_steps=2,
            metadata={
                "benchmark_family": "research_library_behavior_probe",
                "capability": "retrieval_grounded_algorithm_paper_content",
                "difficulty": "hard",
                "requires_retrieval": True,
                "research_topic": (
                    "determinant sums undirected Hamiltonicity Monte Carlo algorithm runtime O star 1.657 n"
                ),
                "step_floor": 1,
            },
        )
    if probe == "leftist_definitions":
        return TaskSpec(
            task_id="research_library_hard_leftist_definitions",
            prompt=(
                "Hard applied provided-context definition probe. Create answer.txt in the workspace. "
                "Use the provided context evidence only; do not inspect local files, dataset paths, /data, /arxiv, "
                "or the internet. The provided context discusses leftist grammars as semi-Thue systems, "
                "their one-step rewrite relation, the active letter of a step, and the definition of greedy "
                "derivations. Convert those definitions into exactly five newline-delimited implementation notes "
                "with these field names: greedy_components, active_index_from_p, rewrite_start, rewrite_output, "
                "and left_action. Use formulas where appropriate. Use a simple printf command that writes to the "
                "relative path answer.txt; do not use /workspace paths, heredoc syntax, python -c writers, "
                "or extra lookup commands."
            ),
            workspace_subdir="research_library_hard_probe_leftist_definitions",
            expected_files=["answer.txt"],
            success_command=(
                "python /data/agentkernel/scripts/verify_research_library_probe.py "
                "--case leftist_definitions --path answer.txt"
            ),
            max_steps=2,
            metadata={
                "benchmark_family": "research_library_behavior_probe",
                "capability": "retrieval_grounded_research_definitions",
                "difficulty": "hard",
                "requires_retrieval": True,
                "research_topic": (
                    "leftist grammars semi-Thue one-step rewrite relation active letter "
                    "greedy derivation leftmost pure eager insert erase left"
                ),
                "step_floor": 1,
            },
        )
    if probe == "leftist_semantics":
        return TaskSpec(
            task_id="research_library_hard_leftist_rewrite_semantics",
            prompt=(
                "Hard applied provided-context semantics probe. Create answer.txt in the workspace. "
                "Use the provided context evidence only; do not inspect local files, dataset paths, /data, /arxiv, "
                "or the internet. The provided context defines a one-step rewrite relation for leftist "
                "grammars as semi-Thue systems and later identifies the active letter for such a step. "
                "Use that paper convention to answer these cases under rule alpha -> beta. "
                "Case 1: word=xxabyy, alpha=ab, beta=Z, p=4. "
                "Case 2: word=abxxabyy, alpha=ab, beta=Z, p=6. "
                "Invalid case: word=xxabyy, alpha=ab, beta=Z, p=2. "
                "Write exactly these newline-delimited fields: case1_u1, case1_alpha, case1_u2, "
                "case1_rewritten, case1_active_index, case2_u1, case2_u2, case2_rewritten, "
                "case2_active_index, and invalid_case. For invalid_case, write ValueError if the requested "
                "step is invalid under the paper convention. Use a simple printf command that writes to the "
                "relative path answer.txt; do not use /workspace paths, heredoc syntax, python -c writers, "
                "or extra lookup commands."
            ),
            workspace_subdir="research_library_hard_probe_leftist_semantics",
            expected_files=["answer.txt"],
            success_command=(
                "python /data/agentkernel/scripts/verify_research_library_probe.py "
                "--case leftist_semantics --path answer.txt"
            ),
            max_steps=2,
            metadata={
                "benchmark_family": "research_library_behavior_probe",
                "capability": "retrieval_grounded_research_semantics",
                "difficulty": "hard",
                "requires_retrieval": True,
                "research_topic": (
                    "leftist grammars semi-Thue one-step rewrite relation rule alpha beta "
                    "decomposition position p active letter"
                ),
                "step_floor": 1,
            },
        )
    if probe == "leftist_step":
        return TaskSpec(
            task_id="research_library_hard_leftist_rewrite_step",
            prompt=(
                "Hard applied provided-context coding probe. Create leftist_step.py in the workspace. "
                "Use the provided context evidence only; do not inspect local files, dataset paths, /data, /arxiv, "
                "or the internet. Implement rewrite_step(word: str, alpha: str, beta: str, p: int) -> dict "
                "for the one-step rewrite relation used by the provided context on leftist grammars as "
                "semi-Thue systems. The paper context defines how a word is decomposed around alpha under a "
                "rule alpha -> beta, what p measures, what rewritten word is produced, and which original "
                "letter position is active in that step. Return a dict with exactly these semantic fields: "
                "u1, alpha, u2, rewritten, and active_index, where active_index is zero-based for Python. "
                "Raise ValueError when the requested step is not valid under the paper convention. "
                "The verifier uses behavioral tests, not exact wording. Use a simple printf command to write "
                "leftist_step.py; do not use heredoc syntax, cat heredocs, python -c writers, or extra lookup commands."
            ),
            workspace_subdir="research_library_hard_probe_leftist_step",
            expected_files=["leftist_step.py"],
            success_command=_leftist_step_success_command(),
            max_steps=3,
            metadata={
                "benchmark_family": "research_library_behavior_probe",
                "capability": "retrieval_grounded_research_implementation",
                "difficulty": "hard",
                "requires_retrieval": True,
                "research_topic": (
                    "leftist grammars semi-Thue one-step rewrite relation rule alpha beta "
                    "decomposition position p active letter"
                ),
                "step_floor": 1,
            },
        )
    if probe == "leftist_window":
        return TaskSpec(
            task_id="research_library_hard_leftist_rewrite_window",
            prompt=(
                "Hard applied provided-context coding probe. Create leftist_window.py in the workspace. "
                "Use the provided context evidence only; do not inspect local files, dataset paths, /data, /arxiv, "
                "or the internet. Implement replacement_start(alpha: str, p: int) -> int for the one-step rewrite "
                "relation convention used in the provided context on leftist grammars and semi-Thue systems. "
                "The difficult part is the paper's convention for what p measures in a decomposition u = u1 alpha u2. "
                "Return the zero-based start offset of alpha in the word. Raise ValueError if p cannot contain alpha "
                "under that convention. The verifier uses behavioral tests, not exact wording."
            ),
            workspace_subdir="research_library_hard_probe_leftist_window",
            expected_files=["leftist_window.py"],
            success_command=_leftist_window_success_command(),
            max_steps=2,
            metadata={
                "benchmark_family": "research_library_behavior_probe",
                "capability": "retrieval_grounded_research_implementation",
                "difficulty": "hard",
                "requires_retrieval": True,
                "research_topic": (
                    "leftist grammars semi-Thue one-step rewrite relation alpha beta u1 alpha u2 position p"
                ),
                "step_floor": 1,
            },
        )
    if probe == "leftist":
        return TaskSpec(
            task_id="research_library_hard_leftist_rewrite_relation",
            prompt=(
                "Hard applied provided-context coding probe. Create leftist_rewrite.py in the workspace. "
                "Use the provided context evidence only; do not inspect local files, dataset paths, /data, /arxiv, "
                "or the internet. Implement rewrite_at_position(word: str, alpha: str, beta: str, p: int) -> str "
                "for the one-step rewrite relation used in the provided context on leftist grammars and "
                "semi-Thue systems. The difficult part is the paper's convention for position p in a decomposition "
                "u = u1 alpha u2 under a rule alpha -> beta. Raise ValueError when alpha does not occur at the "
                "position required by that convention. The verifier uses behavioral tests, not exact wording. "
                "Use a simple allowed command to write the Python file; do not use heredoc syntax or extra lookup commands."
            ),
            workspace_subdir="research_library_hard_probe_leftist",
            expected_files=["leftist_rewrite.py"],
            success_command=_leftist_rewrite_success_command(),
            max_steps=3,
            metadata={
                "benchmark_family": "research_library_behavior_probe",
                "capability": "retrieval_grounded_research_implementation",
                "difficulty": "hard",
                "requires_retrieval": True,
                "research_topic": (
                    "leftist grammars semi-Thue one-step rewrite relation alpha beta u1 alpha u2 position p"
                ),
                "step_floor": 1,
            },
        )
    if probe == "herding_content":
        return TaskSpec(
            task_id="research_library_hard_herding_content_strategy",
            prompt=(
                "Hard applied provided-context behavior probe. Create strategy.txt in the workspace. "
                "Use the provided context evidence only; do not inspect local files, dataset paths, /data, /arxiv, "
                "or the internet. The provided context includes a paper about artificial herders in a "
                "Cows and Herders scenario. Extract the concrete behavior-level strategy choices a developer "
                "would implement. strategy.txt must contain exactly three concise labeled lines: "
                "cow_grouping=..., target_delegation=..., and fence_timing=.... "
                "Do not quote title, paper id, authors, or metadata; use the strategy content. "
                "Use a simple printf command that writes to the relative path strategy.txt; do not use /workspace "
                "paths, heredoc syntax, python -c writers, or extra lookup commands."
            ),
            workspace_subdir="research_library_hard_probe_herding_content",
            expected_files=["strategy.txt"],
            success_command=(
                "python /data/agentkernel/scripts/verify_research_library_probe.py "
                "--case herding_content --path strategy.txt"
            ),
            max_steps=2,
            metadata={
                "benchmark_family": "research_library_behavior_probe",
                "capability": "retrieval_grounded_applied_strategy",
                "difficulty": "hard",
                "requires_retrieval": True,
                "research_topic": (
                    "Cows and Herders artificial herders strategy grouping leader delegate targets "
                    "fence timing fleeing right way"
                ),
                "step_floor": 1,
            },
        )
    if probe == "fence_content":
        return TaskSpec(
            task_id="research_library_hard_fence_content_strategy",
            prompt=(
                "Hard applied provided-context behavior probe. Create strategy.txt in the workspace. "
                "Use the provided context evidence only; do not inspect local files, dataset paths, /data, /arxiv, "
                "or the internet. The provided context includes a paper about artificial herders in a "
                "Cows and Herders scenario. Extract the behavior-level decision for the fence action: who "
                "coordinates it and the concrete condition for opening it. Do not quote title, paper id, authors, "
                "or metadata; use the strategy content. Use a simple printf command that writes to the relative "
                "path strategy.txt; do not use /workspace paths, heredoc syntax, python -c writers, or extra lookup commands."
            ),
            workspace_subdir="research_library_hard_probe_fence_content",
            expected_files=["strategy.txt"],
            success_command=(
                "python /data/agentkernel/scripts/verify_research_library_probe.py "
                "--case fence_content --path strategy.txt"
            ),
            max_steps=2,
            metadata={
                "benchmark_family": "research_library_behavior_probe",
                "capability": "retrieval_grounded_applied_strategy",
                "difficulty": "hard",
                "requires_retrieval": True,
                "research_topic": "Cows and Herders artificial herders team leader fence open fleeing right way",
                "step_floor": 1,
            },
        )
    if probe == "fence":
        row = paper_row or {}
        title = row.get("title", "Developing Artificial Herders Using Jason")
        paper_id = row.get("paper_id", "1001.0115")
        return TaskSpec(
            task_id="research_library_hard_one_step_fence_timing",
            prompt=(
                "One-step hard applied-knowledge probe. Create strategy.txt in the workspace. "
                "Use the provided context evidence only; do not inspect local files, dataset paths, /data, /arxiv, "
                "or the internet. "
                f"The relevant local paper is paper_id {paper_id}, titled '{title}'. "
                "Synthesize who coordinates the fence action and the concrete timing condition for opening it. "
                "This is not a quote-copy task: write the behavior-level decision a developer should implement. "
                "The verifier checks semantic behavior, not exact wording. Use a simple allowed command such as "
                "printf to write strategy.txt; do not use cat, heredoc syntax, or extra lookup commands."
            ),
            workspace_subdir="research_library_hard_probe_fence",
            expected_files=["strategy.txt"],
            success_command=_fence_success_command(),
            max_steps=2,
            metadata={
                "benchmark_family": "research_library_behavior_probe",
                "capability": "retrieval_grounded_applied_strategy",
                "difficulty": "hard",
                "requires_retrieval": True,
                "paper_id": str(paper_id),
                "paper_title": str(title),
                "research_topic": "Cows and Herders fence open team leader fleeing right way strategy timing",
                "step_floor": 1,
            },
        )
    if probe == "herding":
        row = paper_row or {}
        title = row.get("title", "Developing Artificial Herders Using Jason")
        paper_id = row.get("paper_id", "1001.0115")
        return TaskSpec(
            task_id="research_library_hard_one_step_herding_strategy",
            prompt=(
                "One-step hard applied-knowledge probe. Create strategy.txt in the workspace. "
                "Use the provided context evidence only; do not inspect local files, dataset paths, /data, /arxiv, "
                "or the internet. "
                f"The relevant local paper is paper_id {paper_id}, titled '{title}'. "
                "Synthesize the concrete herd-control strategy choices from the research context. "
                "strategy.txt must contain three concise labeled behavior lines: cow_grouping=..., "
                "target_delegation=..., and fence_timing=.... This is not a quote-copy task: write decisions that "
                "would let a developer implement the strategy. The verifier checks semantic behavior, not exact wording. "
                "Use a simple allowed command such as printf to write strategy.txt; do not use cat, heredoc syntax, "
                "or extra lookup commands."
            ),
            workspace_subdir="research_library_hard_probe_herding",
            expected_files=["strategy.txt"],
            success_command=_herding_success_command(),
            max_steps=2,
            metadata={
                "benchmark_family": "research_library_behavior_probe",
                "capability": "retrieval_grounded_applied_strategy",
                "difficulty": "hard",
                "requires_retrieval": True,
                "paper_id": str(paper_id),
                "paper_title": str(title),
                "research_topic": (
                    "Cows and Herders strategy clustering adjacent cows leader delegate targets "
                    "including himself fence open fleeing right way"
                ),
                "step_floor": 1,
            },
        )
    if probe == "paper":
        row = paper_row or {}
        title = row.get("title", "Developing Artificial Herders Using Jason")
        paper_id = row.get("paper_id", "1001.0115")
        return TaskSpec(
            task_id="research_library_hard_one_step_paper_content",
            prompt=(
                "One-step hard paper-knowledge retrieval probe. Create answer.txt in the workspace. "
                "Use the provided context evidence only; do not inspect /data manually and do not use the internet. "
                "Do not import or call any local data API; all content needed is already in the provided context. "
                f"Find the local paper chunk for paper_id {paper_id}, titled '{title}'. "
                "Write exactly three newline-delimited fields from that paper content: "
                "paper_id=<exact paper id>, title=<exact title>, and "
                "evidence_sentence=<the exact sentence saying how the strategy is implemented>. "
                "Use one command shaped like: printf '%s\\n' 'paper_id=<exact>' 'title=<exact>' "
                "'evidence_sentence=<exact>' > answer.txt. "
                "The verifier requires exact content. Do not use echo, ellipsis, summaries, heredoc syntax, or extra lookup commands."
            ),
            workspace_subdir="research_library_hard_probe_paper",
            expected_file_contents={"answer.txt": expected_answer},
            success_command="test -f answer.txt",
            max_steps=1,
            metadata={
                "benchmark_family": "research_library_hard_probe",
                "capability": "retrieval_grounded_paper_content",
                "difficulty": "hard",
                "requires_retrieval": True,
                "research_exact_answer": True,
                "paper_id": str(paper_id),
                "paper_title": str(title),
                "research_topic": "paper content exact evidence sentence",
                "step_floor": 1,
            },
        )
    if probe == "content":
        return TaskSpec(
            task_id="research_library_hard_one_step_algorithm_content",
            prompt=(
                "One-step hard content retrieval probe. Create answer.txt in the workspace. "
                "Use the provided context evidence only; do not inspect /data manually and do not use the internet. "
                "Find the algorithm catalog entry named 'Dijkstras Two Stack Algorithm'. "
                "Write exactly four newline-delimited fields from that library entry: "
                "algo_id=<exact algo_id>, category=<exact category>, topics=<comma-separated topics in catalog order>, "
                "and notes=<exact notes text>. The verifier requires exact content. "
                "Use a simple allowed command such as printf; do not use heredoc syntax."
            ),
            workspace_subdir="research_library_hard_probe_content",
            expected_file_contents={"answer.txt": expected_answer},
            success_command="test -f answer.txt",
            max_steps=1,
            metadata={
                "benchmark_family": "research_library_hard_probe",
                "capability": "retrieval_grounded_content",
                "difficulty": "hard",
                "requires_retrieval": True,
                "step_floor": 1,
            },
        )
    return TaskSpec(
        task_id="research_library_hard_one_step_inventory_dijkstra",
        prompt=(
            "One-step hard retrieval probe. Create a file named answer.txt in the workspace. "
            "It must contain exactly three newline-delimited fields: "
            "paper_rows=<the full-paper row count from the provided context>, "
            "trained_model_assets=<the trained model asset count from the provided context>, "
            "and dijkstra_worst=<the worst-case complexity for dijkstra from the algorithm catalog>. "
            "Do not inspect /data manually and do not use the internet. If no provided context is "
            "available, make the best attempt in one command. The verifier requires exact content."
        ),
        workspace_subdir="research_library_hard_probe",
        expected_file_contents={"answer.txt": expected_answer},
        success_command="test -f answer.txt",
        max_steps=1,
        metadata={
            "benchmark_family": "research_library_hard_probe",
            "capability": "retrieval_grounded_synthesis",
            "difficulty": "hard",
            "requires_retrieval": True,
            "repo": "TOLBERT",
            "step_floor": 1,
        },
    )


def _variant_config(
    *,
    base: KernelConfig,
    output_root: Path,
    variant: str,
    research_enabled: bool,
    status_path: Path,
    config_path: Path,
    vllm_timeout: int,
    require_live_llm: bool,
    automated_mode: bool,
    step_budget: int,
) -> KernelConfig:
    config = KernelConfig(
        provider=base.provider,
        model_name=base.model_name,
        vllm_host=base.vllm_host,
        vllm_api_key=base.vllm_api_key,
        vllm_python_bin=base.vllm_python_bin,
        vllm_start_command=base.vllm_start_command,
        vllm_start_extra_args=base.vllm_start_extra_args,
        vllm_tensor_parallel_size=base.vllm_tensor_parallel_size,
        vllm_max_model_len=base.vllm_max_model_len,
        vllm_gpu_memory_utilization=base.vllm_gpu_memory_utilization,
        vllm_enforce_eager=base.vllm_enforce_eager,
        vllm_reasoning_parser=base.vllm_reasoning_parser,
        vllm_language_model_only=base.vllm_language_model_only,
        vllm_autostart=base.vllm_autostart,
        vllm_autostart_timeout_seconds=vllm_timeout,
        ollama_host=base.ollama_host,
        workspace_root=output_root / "workspaces" / variant,
        trajectories_root=output_root / "trajectories" / variant,
        runtime_database_path=output_root / "runtime" / f"{variant}.sqlite",
        learning_artifacts_path=output_root / "learning" / variant / "learning_candidates.jsonl",
        run_checkpoints_dir=output_root / "checkpoints" / variant,
        run_reports_dir=output_root / "reports" / variant,
        unattended_workspace_snapshot_root=output_root / "snapshots" / variant,
        use_tolbert_context=False,
        use_research_library_context=research_enabled,
        research_library_standalone_context=research_enabled,
        research_library_status_path=status_path,
        research_library_config_path=config_path,
        research_library_context_max_chunks=7,
        research_library_context_max_models=8,
        research_library_context_max_repositories=4,
        research_library_context_max_algorithms=4,
        research_library_context_max_paper_hits=4,
        research_library_paper_scan_file_limit=1,
        research_library_paper_scan_row_limit=512,
        use_skills=False,
        use_graph_memory=False,
        use_world_model=False,
        use_universe_model=False,
        use_state_estimation_proposals=False,
        use_trust_proposals=False,
        use_recovery_proposals=False,
        use_delegation_proposals=False,
        use_operator_policy_proposals=False,
        use_transition_model_proposals=False,
        use_tolbert_model_artifacts=False,
        use_planner=False,
        use_role_specialization=False,
        use_prompt_proposals=False,
        use_curriculum_proposals=False,
        use_retrieval_proposals=False,
        persist_episode_memory=False,
        persist_learning_candidates=False,
        asi_coding_require_live_llm=require_live_llm,
        command_timeout_seconds=20,
        max_task_steps_hard_cap=step_budget,
    )
    if automated_mode:
        config.use_tolbert_context = base.use_tolbert_context
        config.use_skills = base.use_skills
        config.use_graph_memory = base.use_graph_memory
        config.use_world_model = base.use_world_model
        config.use_universe_model = base.use_universe_model
        config.use_state_estimation_proposals = base.use_state_estimation_proposals
        config.use_trust_proposals = base.use_trust_proposals
        config.use_recovery_proposals = base.use_recovery_proposals
        config.use_delegation_proposals = base.use_delegation_proposals
        config.use_operator_policy_proposals = base.use_operator_policy_proposals
        config.use_transition_model_proposals = base.use_transition_model_proposals
        config.use_tolbert_model_artifacts = base.use_tolbert_model_artifacts
        config.use_planner = base.use_planner
        config.use_role_specialization = base.use_role_specialization
        config.use_prompt_proposals = base.use_prompt_proposals
        config.use_curriculum_proposals = base.use_curriculum_proposals
        config.use_retrieval_proposals = base.use_retrieval_proposals
        config.asi_coding_require_live_llm = require_live_llm
    return config


def _write_retrieval_primary_policy(path: Path, benchmark_family: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "artifact_kind": "retrieval_policy_set",
                "lifecycle_state": "proposed",
                "runtime_policy": {
                    "primary_benchmark_families": [benchmark_family],
                    "min_path_confidence": 0.0,
                    "allow_trusted_primary_without_min_confidence": True,
                    "trusted_primary_min_confidence": 0.0,
                    "require_trusted_retrieval": True,
                    "allow_direct_command_primary": True,
                    "allow_skill_primary": True,
                    "primary_min_command_score": 0,
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _step_summary(episode: EpisodeRecord) -> dict[str, Any]:
    if not episode.steps:
        return {}
    step = episode.steps[-1]
    command_result = step.command_result if isinstance(step.command_result, dict) else {}
    return {
        "action": step.action,
        "content": step.content,
        "exit_code": command_result.get("exit_code"),
        "stdout": str(command_result.get("stdout", ""))[-500:],
        "stderr": str(command_result.get("stderr", ""))[-500:],
        "verification_passed": bool(step.verification.get("passed", False)),
        "verification_reasons": list(step.verification.get("reasons", [])),
        "failure_origin": step.failure_origin,
        "failure_signals": list(step.failure_signals),
        "decision_source": step.decision_source,
        "tolbert_route_mode": step.tolbert_route_mode,
        "selected_retrieval_span_id": step.selected_retrieval_span_id,
        "retrieval_influenced": bool(step.retrieval_influenced),
        "trust_retrieval": bool(step.trust_retrieval),
        "research_context_chunk_count": int(step.research_context_chunk_count),
        "llm_visible_research_context_chunk_count": int(step.llm_visible_research_context_chunk_count),
        "research_retrieval_evidence_count": int(step.research_retrieval_evidence_count),
        "research_model_asset_count": int(step.research_model_asset_count),
        "research_repository_match_count": int(step.research_repository_match_count),
        "research_algorithm_match_count": int(step.research_algorithm_match_count),
    }


def _variant_failure_kind(variant: dict[str, Any]) -> str:
    if bool(variant.get("success")):
        return "success"
    if variant.get("runner_exception"):
        text = str(variant.get("runner_exception", ""))
        if "did not return a parseable JSON decision" in text or "inference" in text.lower():
            return "inference_failure"
        return "runner_exception"
    last_step = variant.get("last_step", {})
    last_step = dict(last_step) if isinstance(last_step, dict) else {}
    failure_origin = str(last_step.get("failure_origin", "")).strip()
    if failure_origin:
        return failure_origin
    content = str(last_step.get("content", ""))
    stderr = str(last_step.get("stderr", ""))
    combined = f"{content}\n{stderr}"
    if "did not return a parseable JSON decision" in combined or "[inference_failure]" in combined:
        return "inference_failure"
    exit_code = last_step.get("exit_code")
    if exit_code == 126:
        return "command_rejected"
    reasons = " ".join(str(item) for item in list(last_step.get("verification_reasons", []) or []))
    if "no state progress" in reasons:
        return "no_state_progress"
    if reasons:
        return "verification_failure"
    return str(variant.get("termination_reason", "")) or "unknown_failure"


def _comparison_summary(variants: dict[str, Any]) -> dict[str, Any]:
    without = dict(variants.get("without_research", {}) or {})
    with_research = dict(variants.get("with_research", {}) or {})
    without_kind = _variant_failure_kind(without)
    with_kind = _variant_failure_kind(with_research)
    invalid_negative_control_kinds = {"command_rejected", "inference_failure", "runner_exception"}
    return {
        "without_failure_kind": without_kind,
        "with_failure_kind": with_kind,
        "adapter_success": bool(with_research.get("success")),
        "negative_control_parseable": without_kind not in invalid_negative_control_kinds,
        "clean_behavioral_ab": (
            bool(with_research.get("success"))
            and not bool(without.get("success"))
            and without_kind not in invalid_negative_control_kinds
        ),
        "orchestration_ab": (
            bool(with_research.get("success"))
            and not bool(without.get("success"))
            and without_kind in invalid_negative_control_kinds
        ),
    }


def _run_variant(
    *,
    name: str,
    config: KernelConfig,
    task: TaskSpec,
    checkpoint_path: Path,
) -> dict[str, Any]:
    kernel = AgentKernel(config=config)
    try:
        episode = kernel.run_task(task, checkpoint_path=checkpoint_path, clean_workspace=True)
    except Exception as exc:
        return {
            "name": name,
            "success": False,
            "runner_exception": f"{exc.__class__.__name__}: {exc}",
            "steps": 0,
            "termination_reason": "runner_exception",
        }
    finally:
        kernel.close()
    answer_path = config.workspace_root / task.workspace_subdir / "answer.txt"
    try:
        answer_text = answer_path.read_text(encoding="utf-8")
    except OSError:
        answer_text = ""
    return {
        "name": name,
        "success": bool(episode.success),
        "steps": len(episode.steps),
        "termination_reason": episode.termination_reason,
        "answer_text": answer_text,
        "last_step": _step_summary(episode),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a hard one-step research-library retrieval A/B probe.")
    parser.add_argument(
        "--probe",
        choices=(
            "repo_cpython_reserved",
            "repo_transformers_flash",
            "repo_pytorch_str2bool",
            "leftist_definitions",
            "leftist_semantics",
            "leftist_step",
            "leftist_window",
            "leftist",
            "herding_content",
            "fence_content",
            "hamiltonicity_runtime",
            "farad_traceability",
            "farad_table3",
            "fence",
            "herding",
            "paper",
            "content",
            "structural",
        ),
        default="leftist_definitions",
    )
    parser.add_argument("--provider", default="vllm")
    parser.add_argument("--model", default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--config", type=Path, default=DEFAULT_RESEARCH_LIBRARY_CONFIG)
    parser.add_argument("--refresh-status", action="store_true")
    parser.add_argument("--vllm-timeout", type=int, default=240)
    parser.add_argument(
        "--allow-retrieval-primary",
        action="store_true",
        help="Allow trusted retrieval guidance to execute as primary policy instead of forcing live LLM control.",
    )
    parser.add_argument(
        "--automated-mode",
        action="store_true",
        help="Use the normal automated kernel policy surfaces instead of the isolated adapter-only harness.",
    )
    parser.add_argument("--step-budget", type=int, default=3)
    args = parser.parse_args()

    repo_root = Path.cwd()
    output_path = args.output if args.output.is_absolute() else repo_root / args.output
    output_root = output_path.parent / "hard_probe_runtime"
    status_path = args.status if args.status.is_absolute() else repo_root / args.status
    config_path = args.config if args.config.is_absolute() else repo_root / args.config
    status = _load_status(
        repo_root=repo_root,
        status_path=status_path,
        config_path=config_path,
        refresh=bool(args.refresh_status),
    )
    paper_row: dict[str, Any] | None = None
    if args.probe in {"fence", "herding", "paper"}:
        expected_answer, paper_row = _expected_paper_answer(status)
    elif args.probe == "content":
        expected_answer = _expected_content_answer(status)
    elif args.probe == "leftist_semantics":
        expected_answer = _expected_leftist_semantics_answer()
    elif args.probe == "leftist_definitions":
        expected_answer = _expected_leftist_definitions_answer()
    elif args.probe == "hamiltonicity_runtime":
        expected_answer = _expected_hamiltonicity_runtime_answer()
    elif args.probe == "farad_traceability":
        expected_answer = _expected_farad_traceability_answer()
    elif args.probe == "farad_table3":
        expected_answer = _expected_farad_table3_answer()
    else:
        expected_answer = _expected_answer(status)
    task = _hard_task(expected_answer, probe=args.probe, paper_row=paper_row)
    task.max_steps = max(1, int(args.step_budget))

    base = KernelConfig(provider=args.provider)
    if args.model:
        base.model_name = args.model
    variants = {}
    for name, enabled in (("without_research", False), ("with_research", True)):
        config = _variant_config(
            base=base,
            output_root=output_root,
            variant=name,
            research_enabled=enabled,
            status_path=status_path,
            config_path=config_path,
            vllm_timeout=args.vllm_timeout,
            require_live_llm=not bool(args.allow_retrieval_primary or args.automated_mode),
            automated_mode=bool(args.automated_mode),
            step_budget=max(1, int(args.step_budget)),
        )
        if args.allow_retrieval_primary:
            config.use_retrieval_proposals = True
            config.retrieval_proposals_path = output_root / "runtime" / name / "retrieval_policy.json"
            _write_retrieval_primary_policy(
                config.retrieval_proposals_path,
                str(task.metadata.get("benchmark_family", "bounded") or "bounded"),
            )
        variants[name] = _run_variant(
            name=name,
            config=config,
            task=task,
            checkpoint_path=output_root / "checkpoints" / f"{name}.json",
        )

    report = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "provider": args.provider,
        "probe": args.probe,
        "model": base.model_name,
        "task": task.to_dict(),
        "expected_answer": expected_answer,
        "variants": variants,
        "helps": bool(variants["with_research"].get("success")) and not bool(
            variants["without_research"].get("success")
        ),
    }
    report["comparison"] = _comparison_summary(variants)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        "research_hard_probe "
        f"without_success={str(bool(variants['without_research'].get('success'))).lower()} "
        f"with_success={str(bool(variants['with_research'].get('success'))).lower()} "
        f"helps={str(report['helps']).lower()} "
        f"output={output_path}"
    )
    for name in ("without_research", "with_research"):
        variant = variants[name]
        last_step = variant.get("last_step", {}) if isinstance(variant.get("last_step", {}), dict) else {}
        print(
            "research_hard_probe_variant "
            f"name={name} success={str(bool(variant.get('success'))).lower()} "
            f"steps={int(variant.get('steps', 0) or 0)} "
            f"research_chunks={int(last_step.get('research_context_chunk_count', 0) or 0)} "
            f"visible_research_chunks={int(last_step.get('llm_visible_research_context_chunk_count', 0) or 0)} "
            f"research_evidence={int(last_step.get('research_retrieval_evidence_count', 0) or 0)} "
            f"termination={variant.get('termination_reason', '')}"
        )


if __name__ == "__main__":
    main()
