from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import fnmatch
import hashlib
import json
import sqlite3
from threading import Lock
from typing import Any


_STORE_CACHE: dict[Path, "SQLiteKernelStore"] = {}
_STORE_CACHE_LOCK = Lock()


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


class SQLiteKernelStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self, *, ensure_schema: bool = True) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        if ensure_schema:
            self._initialize_connection(conn)
        return conn

    def _initialize(self) -> None:
        with self._connect(ensure_schema=False) as conn:
            self._initialize_connection(conn)

    def _initialize_connection(self, conn: sqlite3.Connection) -> None:
        existing_tables = {
            str(row["name"])
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        episode_columns = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(episodes)").fetchall()
        }
        expected_tables = {
            "episodes",
            "learning_candidates",
            "cycle_records",
            "delegated_jobs",
            "runtime_states",
            "export_manifests",
        }
        if expected_tables.issubset(existing_tables) and "episode_id" in episode_columns:
            return
        if episode_columns and "episode_id" not in episode_columns:
            conn.execute("ALTER TABLE episodes RENAME TO episodes_legacy_v1")
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS episodes (
                episode_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                summary_json TEXT NOT NULL,
                benchmark_family TEXT NOT NULL,
                success INTEGER NOT NULL,
                storage_json TEXT NOT NULL,
                storage_phase TEXT NOT NULL,
                storage_source_group TEXT NOT NULL,
                storage_relative_path TEXT NOT NULL,
                storage_depth INTEGER NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_episodes_task_updated
                ON episodes(task_id, updated_at);
            CREATE INDEX IF NOT EXISTS idx_episodes_family_success
                ON episodes(benchmark_family, success, updated_at);
            CREATE INDEX IF NOT EXISTS idx_episodes_storage_path
                ON episodes(storage_relative_path);

            CREATE TABLE IF NOT EXISTS learning_candidates (
                candidate_id TEXT PRIMARY KEY,
                artifact_kind TEXT NOT NULL,
                source_task_id TEXT NOT NULL,
                benchmark_family TEXT NOT NULL,
                memory_source TEXT NOT NULL,
                support_count INTEGER NOT NULL,
                payload_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_learning_candidates_kind
                ON learning_candidates(artifact_kind, source_task_id, updated_at);

            CREATE TABLE IF NOT EXISTS cycle_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                output_path TEXT NOT NULL,
                cycle_id TEXT NOT NULL,
                subsystem TEXT NOT NULL,
                state TEXT NOT NULL,
                selected_variant_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                recorded_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_cycle_records_path_id
                ON cycle_records(output_path, id);
            CREATE INDEX IF NOT EXISTS idx_cycle_records_cycle
                ON cycle_records(cycle_id, recorded_at);

            CREATE TABLE IF NOT EXISTS delegated_jobs (
                queue_path TEXT NOT NULL,
                job_id TEXT NOT NULL,
                state TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(queue_path, job_id)
            );
            CREATE INDEX IF NOT EXISTS idx_delegated_jobs_state
                ON delegated_jobs(queue_path, state, updated_at);

            CREATE TABLE IF NOT EXISTS runtime_states (
                runtime_path TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS export_manifests (
                export_key TEXT PRIMARY KEY,
                export_kind TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        legacy_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='episodes_legacy_v1'"
        ).fetchone()
        if legacy_exists is not None:
            legacy_rows = conn.execute(
                """
                SELECT task_id, payload_json, summary_json, benchmark_family, success, storage_json,
                       storage_phase, storage_source_group, storage_relative_path, storage_depth, updated_at
                FROM episodes_legacy_v1
                ORDER BY updated_at, task_id
                """
            ).fetchall()
            for row in legacy_rows:
                payload_json = str(row["payload_json"])
                storage_json = str(row["storage_json"])
                episode_id = self._episode_id_from_payload_json(
                    task_id=str(row["task_id"]),
                    payload_json=payload_json,
                    storage_json=storage_json,
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO episodes(
                        episode_id,
                        task_id,
                        payload_json,
                        summary_json,
                        benchmark_family,
                        success,
                        storage_json,
                        storage_phase,
                        storage_source_group,
                        storage_relative_path,
                        storage_depth,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        episode_id,
                        str(row["task_id"]),
                        payload_json,
                        str(row["summary_json"]),
                        str(row["benchmark_family"]),
                        int(row["success"]),
                        storage_json,
                        str(row["storage_phase"]),
                        str(row["storage_source_group"]),
                        str(row["storage_relative_path"]),
                        int(row["storage_depth"]),
                        str(row["updated_at"]),
                    ),
                )
            conn.execute("DROP TABLE episodes_legacy_v1")

    @staticmethod
    def _episode_id_from_payload_json(
        *,
        task_id: str,
        payload_json: str,
        storage_json: str,
    ) -> str:
        try:
            payload = json.loads(payload_json)
        except json.JSONDecodeError:
            payload = {}
        try:
            storage = json.loads(storage_json)
        except json.JSONDecodeError:
            storage = {}
        workspace = ""
        if isinstance(payload, dict):
            workspace = str(payload.get("workspace", "")).strip()
        storage_payload = dict(storage) if isinstance(storage, dict) else {}
        identity_payload = {
            "task_id": str(task_id).strip(),
            "workspace": workspace,
            "relative_path": str(storage_payload.get("relative_path", "")).strip(),
            "phase": str(storage_payload.get("phase", "")).strip(),
            "source_group": str(storage_payload.get("source_group", "")).strip(),
            "payload_sha1": hashlib.sha1(payload_json.encode("utf-8")).hexdigest(),
        }
        digest = hashlib.sha1(
            json.dumps(identity_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return f"{identity_payload['task_id']}:{digest}"

    def upsert_episode_document(
        self,
        *,
        payload: dict[str, object],
        storage: dict[str, object] | None = None,
    ) -> None:
        task_id = str(payload.get("task_id", "")).strip()
        if not task_id:
            raise ValueError("episode task_id must not be empty")
        summary = payload.get("summary", {})
        if not isinstance(summary, dict):
            summary = {}
        task_metadata = payload.get("task_metadata", {})
        if not isinstance(task_metadata, dict):
            task_metadata = {}
        storage_payload = dict(storage or {})
        storage_phase = str(storage_payload.get("phase", "primary")).strip() or "primary"
        storage_source_group = str(storage_payload.get("source_group", "")).strip()
        storage_relative_path = str(storage_payload.get("relative_path", f"{task_id}.json")).strip() or f"{task_id}.json"
        try:
            storage_depth = int(storage_payload.get("depth", 0) or 0)
        except (TypeError, ValueError):
            storage_depth = 0
        payload_json = json.dumps(payload, sort_keys=True)
        storage_json = json.dumps(storage_payload, sort_keys=True)
        episode_id = self._episode_id_from_payload_json(
            task_id=task_id,
            payload_json=payload_json,
            storage_json=storage_json,
        )
        updated_at = _utcnow()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO episodes(
                    episode_id,
                    task_id,
                    payload_json,
                    summary_json,
                    benchmark_family,
                    success,
                    storage_json,
                    storage_phase,
                    storage_source_group,
                    storage_relative_path,
                    storage_depth,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(episode_id) DO UPDATE SET
                    task_id=excluded.task_id,
                    payload_json=excluded.payload_json,
                    summary_json=excluded.summary_json,
                    benchmark_family=excluded.benchmark_family,
                    success=excluded.success,
                    storage_json=excluded.storage_json,
                    storage_phase=excluded.storage_phase,
                    storage_source_group=excluded.storage_source_group,
                    storage_relative_path=excluded.storage_relative_path,
                    storage_depth=excluded.storage_depth,
                    updated_at=excluded.updated_at
                """,
                (
                    episode_id,
                    task_id,
                    payload_json,
                    json.dumps(summary, sort_keys=True),
                    str(task_metadata.get("benchmark_family", "bounded")).strip() or "bounded",
                    1 if bool(payload.get("success", False)) else 0,
                    storage_json,
                    storage_phase,
                    storage_source_group,
                    storage_relative_path,
                    storage_depth,
                    updated_at,
                ),
            )

    def _row_to_episode_payload(self, row: sqlite3.Row) -> dict[str, object] | None:
        try:
            payload = json.loads(str(row["payload_json"]))
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        try:
            storage = json.loads(str(row["storage_json"]))
        except json.JSONDecodeError:
            storage = {}
        payload = dict(payload)
        storage_payload = dict(storage) if isinstance(storage, dict) else {}
        storage_payload.setdefault("episode_id", str(row["episode_id"]))
        storage_payload.setdefault("updated_at", str(row["updated_at"]))
        if storage_payload:
            payload["episode_storage"] = storage_payload
            task_metadata = dict(payload.get("task_metadata", {})) if isinstance(payload.get("task_metadata", {}), dict) else {}
            task_metadata.setdefault("episode_phase", str(storage_payload.get("phase", "")))
            task_metadata.setdefault("episode_source_group", str(storage_payload.get("source_group", "")))
            task_metadata.setdefault("episode_relative_path", str(storage_payload.get("relative_path", "")))
            payload["task_metadata"] = task_metadata
        return payload

    @staticmethod
    def _phase_preference(phase: str) -> int:
        normalized = str(phase).strip()
        if normalized == "primary":
            return 4
        if normalized == "generated_success":
            return 3
        if normalized == "generated_failure_seed":
            return 2
        if normalized == "generated_failure":
            return 1
        return 0

    def _aggregate_episode_documents(self, documents: list[dict[str, object]]) -> list[dict[str, object]]:
        grouped: dict[str, list[dict[str, object]]] = {}
        for payload in documents:
            task_id = str(payload.get("task_id", "")).strip()
            if not task_id:
                continue
            grouped.setdefault(task_id, []).append(payload)
        aggregated: list[dict[str, object]] = []
        for task_id, attempts in grouped.items():
            ranked_attempts = sorted(
                attempts,
                key=lambda payload: (
                    1 if bool(payload.get("success", False)) else 0,
                    float(dict(payload.get("summary", {})).get("final_completion_ratio", 0.0) or 0.0),
                    float(dict(payload.get("summary", {})).get("net_state_progress_delta", 0.0) or 0.0),
                    self._phase_preference(dict(payload.get("episode_storage", {})).get("phase", "")),
                    str(dict(payload.get("episode_storage", {})).get("updated_at", "")),
                ),
                reverse=True,
            )
            representative = dict(ranked_attempts[0])
            attempts_by_phase: dict[str, int] = {}
            successful_attempts = 0
            for attempt in attempts:
                storage = dict(attempt.get("episode_storage", {}))
                phase = str(storage.get("phase", "primary")).strip() or "primary"
                attempts_by_phase[phase] = attempts_by_phase.get(phase, 0) + 1
                if bool(attempt.get("success", False)):
                    successful_attempts += 1
            representative["attempt_aggregation"] = {
                "attempt_count": len(attempts),
                "successful_attempts": successful_attempts,
                "failed_attempts": max(0, len(attempts) - successful_attempts),
                "attempts_by_phase": attempts_by_phase,
                "latest_updated_at": max(
                    str(dict(attempt.get("episode_storage", {})).get("updated_at", ""))
                    for attempt in attempts
                ),
                "representative_episode_id": str(
                    dict(representative.get("episode_storage", {})).get("episode_id", "")
                ).strip(),
            }
            aggregated.append(representative)
        return sorted(aggregated, key=lambda payload: str(payload.get("task_id", "")).strip())

    def iter_episode_attempt_documents(self, *, flat_only: bool = False) -> list[dict[str, object]]:
        query = """
            SELECT episode_id, payload_json, storage_json, updated_at
            FROM episodes
        """
        params: list[object] = []
        if flat_only:
            query += " WHERE storage_depth = 0"
        query += " ORDER BY updated_at, storage_relative_path, task_id, episode_id"
        documents: list[dict[str, object]] = []
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        for row in rows:
            payload = self._row_to_episode_payload(row)
            if payload is not None:
                documents.append(payload)
        return documents

    def iter_episode_documents(self, *, flat_only: bool = False) -> list[dict[str, object]]:
        return self._aggregate_episode_documents(
            self.iter_episode_attempt_documents(flat_only=flat_only)
        )

    def load_episode_attempt_documents(self, task_id: str) -> list[dict[str, object]]:
        normalized = str(task_id).strip()
        if not normalized:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT episode_id, payload_json, storage_json, updated_at
                FROM episodes
                WHERE task_id = ?
                ORDER BY updated_at, episode_id
                """,
                (normalized,),
            ).fetchall()
        attempts = [payload for row in rows if (payload := self._row_to_episode_payload(row)) is not None]
        return attempts

    def load_episode_document(self, task_id: str) -> dict[str, object]:
        normalized = str(task_id).strip()
        attempts = self.load_episode_attempt_documents(normalized)
        if not attempts:
            raise FileNotFoundError(f"episode document not found for task_id={normalized}")
        return self._aggregate_episode_documents(attempts)[0]

    def iter_trajectory_payloads(self) -> list[dict[str, object]]:
        return self.iter_episode_documents(flat_only=True)

    def load_learning_candidates(self, *, artifact_kind: str = "") -> list[dict[str, object]]:
        query = "SELECT payload_json FROM learning_candidates"
        params: list[object] = []
        if artifact_kind.strip():
            query += " WHERE artifact_kind = ?"
            params.append(artifact_kind.strip())
        query += " ORDER BY candidate_id"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        candidates: list[dict[str, object]] = []
        for row in rows:
            try:
                payload = json.loads(str(row["payload_json"]))
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                candidates.append(payload)
        return candidates

    def load_learning_candidates_by_ids(self, candidate_ids: list[str]) -> list[dict[str, object]]:
        normalized = [str(candidate_id).strip() for candidate_id in candidate_ids if str(candidate_id).strip()]
        if not normalized:
            return []
        placeholders = ",".join("?" for _ in normalized)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT payload_json FROM learning_candidates WHERE candidate_id IN ({placeholders})",
                normalized,
            ).fetchall()
        payloads: list[dict[str, object]] = []
        for row in rows:
            try:
                payload = json.loads(str(row["payload_json"]))
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                payloads.append(payload)
        return payloads

    def upsert_learning_candidates(self, candidates: list[dict[str, object]]) -> None:
        now = _utcnow()
        rows: list[tuple[object, ...]] = []
        for candidate in candidates:
            candidate_id = str(candidate.get("candidate_id", "")).strip()
            if not candidate_id:
                continue
            rows.append(
                (
                    candidate_id,
                    str(candidate.get("artifact_kind", "")).strip(),
                    str(candidate.get("source_task_id", "")).strip(),
                    str(candidate.get("benchmark_family", "bounded")).strip() or "bounded",
                    str(candidate.get("memory_source", "")).strip(),
                    int(candidate.get("support_count", 1) or 1),
                    json.dumps(candidate, sort_keys=True),
                    now,
                )
            )
        if not rows:
            return
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO learning_candidates(
                    candidate_id,
                    artifact_kind,
                    source_task_id,
                    benchmark_family,
                    memory_source,
                    support_count,
                    payload_json,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(candidate_id) DO UPDATE SET
                    artifact_kind=excluded.artifact_kind,
                    source_task_id=excluded.source_task_id,
                    benchmark_family=excluded.benchmark_family,
                    memory_source=excluded.memory_source,
                    support_count=excluded.support_count,
                    payload_json=excluded.payload_json,
                    updated_at=excluded.updated_at
                """,
                rows,
            )

    def append_cycle_record(self, *, output_path: Path, payload: dict[str, object]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO cycle_records(
                    output_path,
                    cycle_id,
                    subsystem,
                    state,
                    selected_variant_id,
                    payload_json,
                    recorded_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(output_path),
                    str(payload.get("cycle_id", "")).strip(),
                    str(payload.get("subsystem", "")).strip(),
                    str(payload.get("state", "")).strip(),
                    _selected_variant_id(payload),
                    json.dumps(payload, sort_keys=True),
                    _utcnow(),
                ),
            )

    def replace_cycle_records(self, *, output_path: Path, records: list[dict[str, object]]) -> None:
        normalized_output_path = str(output_path)
        with self._connect() as conn:
            conn.execute("DELETE FROM cycle_records WHERE output_path = ?", (normalized_output_path,))
            for payload in records:
                conn.execute(
                    """
                    INSERT INTO cycle_records(
                        output_path,
                        cycle_id,
                        subsystem,
                        state,
                        selected_variant_id,
                        payload_json,
                        recorded_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        normalized_output_path,
                        str(payload.get("cycle_id", "")).strip(),
                        str(payload.get("subsystem", "")).strip(),
                        str(payload.get("state", "")).strip(),
                        _selected_variant_id(payload),
                        json.dumps(payload, sort_keys=True),
                        _utcnow(),
                    ),
                )

    def load_cycle_records(self, *, output_path: Path) -> list[dict[str, object]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT payload_json
                FROM cycle_records
                WHERE output_path = ?
                ORDER BY id
                """,
                (str(output_path),),
            ).fetchall()
        records: list[dict[str, object]] = []
        for row in rows:
            try:
                payload = json.loads(str(row["payload_json"]))
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
        return records

    def list_cycle_paths(self, *, parent: Path, pattern: str) -> list[Path]:
        matched: list[Path] = []
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT output_path FROM cycle_records ORDER BY output_path"
            ).fetchall()
        for row in rows:
            raw = str(row["output_path"]).strip()
            if not raw:
                continue
            path = Path(raw)
            if path.parent != parent:
                continue
            if fnmatch.fnmatch(path.name, pattern):
                matched.append(path)
        return matched

    def replace_job_queue(self, *, queue_path: Path, jobs: list[dict[str, object]]) -> None:
        normalized_queue_path = str(queue_path)
        now = _utcnow()
        with self._connect() as conn:
            conn.execute("DELETE FROM delegated_jobs WHERE queue_path = ?", (normalized_queue_path,))
            conn.executemany(
                """
                INSERT INTO delegated_jobs(queue_path, job_id, state, payload_json, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        normalized_queue_path,
                        str(job.get("job_id", "")).strip(),
                        str(job.get("state", "")).strip(),
                        json.dumps(job, sort_keys=True),
                        now,
                    )
                    for job in jobs
                    if str(job.get("job_id", "")).strip()
                ],
            )

    def load_job_queue(self, *, queue_path: Path) -> list[dict[str, object]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT payload_json
                FROM delegated_jobs
                WHERE queue_path = ?
                ORDER BY updated_at, job_id
                """,
                (str(queue_path),),
            ).fetchall()
        jobs: list[dict[str, object]] = []
        for row in rows:
            try:
                payload = json.loads(str(row["payload_json"]))
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                jobs.append(payload)
        return jobs

    def replace_runtime_state(self, *, runtime_path: Path, payload: dict[str, object]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runtime_states(runtime_path, payload_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(runtime_path) DO UPDATE SET
                    payload_json=excluded.payload_json,
                    updated_at=excluded.updated_at
                """,
                (
                    str(runtime_path),
                    json.dumps(payload, sort_keys=True),
                    _utcnow(),
                ),
            )

    def load_runtime_state(self, *, runtime_path: Path) -> dict[str, object]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM runtime_states WHERE runtime_path = ?",
                (str(runtime_path),),
            ).fetchone()
        if row is None:
            return {}
        try:
            payload = json.loads(str(row["payload_json"]))
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def record_export_manifest(self, *, export_key: str, export_kind: str, payload: dict[str, object]) -> None:
        normalized_key = str(export_key).strip()
        if not normalized_key:
            return
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO export_manifests(export_key, export_kind, payload_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(export_key) DO UPDATE SET
                    export_kind=excluded.export_kind,
                    payload_json=excluded.payload_json,
                    updated_at=excluded.updated_at
                """,
                (
                    normalized_key,
                    str(export_kind).strip(),
                    json.dumps(payload, sort_keys=True),
                    _utcnow(),
                ),
            )

    def load_export_manifest(self, *, export_key: str) -> dict[str, object]:
        normalized_key = str(export_key).strip()
        if not normalized_key:
            return {}
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM export_manifests WHERE export_key = ?",
                (normalized_key,),
            ).fetchone()
        if row is None:
            return {}
        try:
            payload = json.loads(str(row["payload_json"]))
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}


def _selected_variant_id(payload: dict[str, object]) -> str:
    metrics = payload.get("metrics_summary", {})
    if isinstance(metrics, dict):
        direct = str(metrics.get("selected_variant_id", "")).strip()
        if direct:
            return direct
        selected = metrics.get("selected_variant", {})
        if isinstance(selected, dict):
            return str(selected.get("variant_id", "")).strip()
    return ""


def store_for_config(config: Any) -> SQLiteKernelStore:
    path = Path(getattr(config, "runtime_database_path"))
    if not path.is_absolute():
        trajectories_root = getattr(config, "trajectories_root", None)
        if trajectories_root is not None:
            trajectories_path = Path(trajectories_root)
            if trajectories_path.is_absolute():
                path = trajectories_path.parent / path
    resolved = path.resolve()
    with _STORE_CACHE_LOCK:
        cached = _STORE_CACHE.get(resolved)
        if cached is None:
            cached = SQLiteKernelStore(resolved)
            _STORE_CACHE[resolved] = cached
        return cached
