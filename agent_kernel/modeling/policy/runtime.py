from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TolbertRoutingDecision:
    mode: str
    benchmark_family: str
    reason: str
    min_confidence: float
    require_trusted_retrieval: bool
    use_latent_state: bool


def choose_tolbert_route(
    *,
    runtime_policy: dict[str, object],
    benchmark_family: str,
    path_confidence: float,
    trust_retrieval: bool,
) -> TolbertRoutingDecision:
    family = str(benchmark_family or "bounded").strip() or "bounded"
    primary_families = {
        str(value).strip()
        for value in runtime_policy.get("primary_benchmark_families", [])
        if str(value).strip()
    }
    shadow_families = {
        str(value).strip()
        for value in runtime_policy.get("shadow_benchmark_families", [])
        if str(value).strip()
    }
    min_confidence = _float_value(runtime_policy.get("min_path_confidence"), 0.75)
    trusted_confidence_floor = _float_value(runtime_policy.get("trusted_primary_min_confidence"), 0.0)
    allow_trusted_primary_without_min_confidence = bool(
        runtime_policy.get("allow_trusted_primary_without_min_confidence", False)
    )
    allow_direct_primary = bool(runtime_policy.get("allow_direct_command_primary", True))
    allow_skill_primary = bool(runtime_policy.get("allow_skill_primary", True))
    require_trusted = bool(runtime_policy.get("require_trusted_retrieval", True))
    use_latent_state = bool(runtime_policy.get("use_latent_state", True))
    if family in primary_families:
        direct_or_skill_primary_allowed = allow_direct_primary or allow_skill_primary
        if require_trusted and not trust_retrieval and not direct_or_skill_primary_allowed:
            return TolbertRoutingDecision(
                mode="disabled",
                benchmark_family=family,
                reason="retained Tolbert primary route requires trusted retrieval",
                min_confidence=min_confidence,
                require_trusted_retrieval=require_trusted,
                use_latent_state=use_latent_state,
            )
        trusted_confidence_override = (
            trust_retrieval
            and allow_trusted_primary_without_min_confidence
            and path_confidence >= trusted_confidence_floor
        )
        low_confidence_direct_override = not trust_retrieval and direct_or_skill_primary_allowed
        if (
            path_confidence < min_confidence
            and not trusted_confidence_override
            and not low_confidence_direct_override
        ):
            return TolbertRoutingDecision(
                mode="disabled",
                benchmark_family=family,
                reason="path confidence below retained Tolbert primary threshold",
                min_confidence=min_confidence,
                require_trusted_retrieval=require_trusted,
                use_latent_state=use_latent_state,
            )
        if trusted_confidence_override:
            return TolbertRoutingDecision(
                mode="primary",
                benchmark_family=family,
                reason="trusted retrieval satisfied the retained Tolbert primary route override",
                min_confidence=min_confidence,
                require_trusted_retrieval=require_trusted,
                use_latent_state=use_latent_state,
            )
        if low_confidence_direct_override:
            return TolbertRoutingDecision(
                mode="primary",
                benchmark_family=family,
                reason="benchmark family is approved for retained Tolbert primary control via direct or skill primary routing despite low path confidence",
                min_confidence=min_confidence,
                require_trusted_retrieval=require_trusted,
                use_latent_state=use_latent_state,
            )
        if require_trusted and not trust_retrieval and direct_or_skill_primary_allowed:
            return TolbertRoutingDecision(
                mode="primary",
                benchmark_family=family,
                reason="benchmark family is approved for retained Tolbert primary control via direct or skill primary routing",
                min_confidence=min_confidence,
                require_trusted_retrieval=require_trusted,
                use_latent_state=use_latent_state,
            )
        return TolbertRoutingDecision(
            mode="primary",
            benchmark_family=family,
            reason="benchmark family is approved for retained Tolbert primary control",
            min_confidence=min_confidence,
            require_trusted_retrieval=require_trusted,
            use_latent_state=use_latent_state,
        )
    if family in shadow_families:
        return TolbertRoutingDecision(
            mode="shadow",
            benchmark_family=family,
            reason="benchmark family is enrolled in retained Tolbert shadow evaluation",
            min_confidence=min_confidence,
            require_trusted_retrieval=require_trusted,
            use_latent_state=use_latent_state,
        )
    return TolbertRoutingDecision(
        mode="disabled",
        benchmark_family=family,
        reason="benchmark family is not enrolled in retained Tolbert routing",
        min_confidence=min_confidence,
        require_trusted_retrieval=require_trusted,
        use_latent_state=use_latent_state,
    )


def _float_value(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
