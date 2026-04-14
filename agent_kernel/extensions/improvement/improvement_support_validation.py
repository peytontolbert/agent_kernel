from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...config import KernelConfig, current_external_task_manifests_paths
from ...tasking.task_bank import TaskBank


@dataclass(slots=True)
class TaskContractCatalog:
    bank: TaskBank

    def get(self, task_id: str) -> Any:
        return self.bank.get(task_id)


def build_default_task_contract_catalog() -> TaskContractCatalog:
    manifest_paths = current_external_task_manifests_paths()
    try:
        bank = TaskBank(
            config=KernelConfig(),
            external_task_manifests=manifest_paths if manifest_paths else None,
        )
    except TypeError:
        bank = TaskBank()
    return TaskContractCatalog(bank=bank)


__all__ = ["TaskContractCatalog", "build_default_task_contract_catalog"]
