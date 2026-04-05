# Unattended Controller

`agent_kernel.unattended_controller` is the model-based round planner for unattended improvement
campaigns. It is not the inner task policy. It chooses the next campaign policy after each round
using learned reward estimates, transition predictions, rollout value, and exploration pressure.

## Why This Lane Matters

Toward stronger autonomy, the outer loop has to do more than search. It has to avoid teaching
itself the wrong lesson from one noisy retained win.

The controller now treats thin evidence as exploration material, not as high-confidence
exploitation:

- unseen actions can still win on policy prior plus exploration
- lightly sampled actions have their learned reward and rollout value down-weighted
- lightly sampled actions also pay a thin-evidence penalty before they are treated as strong bets
- well-supported actions keep full empirical reward and rollout credit

That shifts the unattended kernel toward a better default: novelty is still allowed, but a single
lucky round is no longer enough to dominate the next policy choice.

## Depth Controls

The unattended layer now carries `task_step_floor` as an explicit outer-loop policy field instead of
only inferring depth from `cycles` or `task_limit`.

- `task_step_floor`
  - the per-round requested frontier floor for inner task step budgets
  - threaded through unattended policy selection and into child campaign runtime via
    `AGENT_KERNEL_FRONTIER_TASK_STEP_FLOOR`

Depth is not rewarded blindly. The controller now distinguishes:

- productive depth
  - deeper retained rounds with non-regressing quality signals
- depth drift
  - extra steps that correlate with flat or regressing retained outcomes

That means the unattended planner can continue through depth intentionally without treating every
longer successful run as suspicious cost, while still tightening the floor again when deeper rounds
start drifting.

The unattended stop counters now use the same round-level signal. `no_yield_rounds` and
`policy_stall_rounds` are no longer keyed only off raw retained-count or exact same-policy checks;
they wait for the full round rationale and only suppress stall when depth evidence is both
productive and non-drifting.

The unattended stop counters now use the same distinction:

- `no_yield_rounds`
  - resets on retained yield as before
  - also refuses to count a round as "no yield" when the round produced explicit productive-depth
    continuation or a retained liftoff outcome
- `policy_stall_rounds`
  - still increments when the outer policy repeats without meaningful new evidence
  - does not increment when the policy repeats but the round shows productive deep continuation
  - still increments on same-policy depth drift, so deeper runs do not get a free pass once quality
    flattens or regresses

The stop thresholds themselves are now adaptive instead of fixed-only:

- `depth_runway_credit`
  - bounded runway earned only by productive, non-drifting depth
  - grows more quickly when the retained depth was explicitly long-horizon
  - decays on ambiguous no-yield rounds
  - resets to zero on drift, failed decisions, or retained phase-gate failure
- `adaptive_stop_budget`
  - reported per round in the unattended campaign report
  - exposes both base and effective `max_no_yield_rounds` / `max_policy_stall_rounds`
  - lets a proven deep run survive a small amount of temporary ambiguity without making drift cheap

## Support Controls

The controller state now includes two support-aware knobs:

- `min_action_support`
  - minimum empirical count before an action is treated as fully supported exploitation
- `thin_evidence_penalty`
  - penalty applied when an action has some empirical wins but still falls short of
    `min_action_support`
- `support_confidence_power`
  - shapes how quickly partially supported actions graduate from thin-evidence discounting toward
    full empirical credit
  - lower values make near-threshold evidence mature faster without removing the thin-evidence
    penalty

Operationally:

- `count == 0`
  - no thin-evidence penalty; the action is still a pure exploration candidate
- `0 < count < min_action_support`
  - empirical reward and rollout value are scaled by graduated support confidence
  - a thin-evidence penalty is subtracted from the score
- `count >= min_action_support`
  - the controller uses the full empirical reward and rollout estimate

In practice this means:

- one lucky round still stays weak
- two or more consistent rounds can stop looking "paper thin" sooner, instead of waiting for a
  hard on/off threshold before the controller adapts

## Generalization Pressure

The unattended controller now also distinguishes narrow retained wins from broader retained
generalization across priority coding families.

- `generalization_gain`
  - rises when retained yield lands in more than one priority family in the same round window
  - rewards retention evidence that transfers beyond one local lane
- `generalization_gap`
  - rises when priority families still have signal gaps or non-retained yield around the retained
    winner
  - keeps the outer loop from over-crediting a narrow success while adjacent repo settings are
    still weak

That means retained evidence now carries a broader question than "did one lane improve?" The outer
loop also asks whether the same campaign is expanding useful coding pressure across repository,
workflow, tooling, and integration settings.

## Frontier Pressure

The unattended layer now also consumes the retained curriculum's concrete weak-spot clusters, not
just broad family yield.

- `frontier_failure_motif_gain` / `frontier_failure_motif_pressure`
  - tracks whether retained rounds are landing in families that the retained curriculum keeps
    surfacing alongside recurring failure motifs such as stalled progress or verifier regressions
- `frontier_repo_setting_gain` / `frontier_repo_setting_pressure`
  - tracks whether retained rounds are landing in the repo settings that keep showing up as weak,
    such as worker handoff, integrator handoff, validation, cleanup, or audit lanes

Operationally:

- retained curriculum controls now bias unattended `priority_benchmark_families` toward the
  families attached to repeated motif/signature weak spots
- the controller observation and reward also reflect whether a round reduced or ignored those
  retained weak-cluster families
- repo-setting pressure now enters candidate generation itself, so the controller evaluates
  setting-shaped policy options instead of only receiving a post-selection override
- repo-setting pressure now also adjusts candidate scoring, so setting-aligned policies receive
  explicit planner priors instead of relying only on learned rollout score ties
- those repo-setting priors are now updated from retained outcome history for the same weak
  setting, so handoff breadth and validation depth stop depending only on fixed heuristic weights
- learned repo-setting priors are now persisted in controller state, so a reused
  `--controller-state-path` compounds those setting-specific breadth/depth preferences across
  unattended campaigns instead of relearning them from only the current round window
- persisted repo-setting priors now also keep family-specialized memory under each signal, with
  lightweight decay on update, so `workflow:worker_handoff` and `integration:worker_handoff` can
  share broad handoff evidence without being forced into one frozen global weight forever
- family-specialized repo-setting priors now also borrow through an explicit neighborhood map
  before falling back to the shared signal pool, so `workflow:worker_handoff` can learn more from
  `tooling:worker_handoff` than from farther lanes such as `integration:worker_handoff`
- unattended policy proposals now also preserve the repo-setting distinction one level deeper than
  family selection:
  - `worker_handoff`, `integrator_handoff`, `shared_repo`, and `repo_sandbox` pressure can widen
    `campaign_width`
  - `validation_lane`, `cleanup_lane`, `audit_lane`, and other depth-heavy settings can raise
    `task_step_floor`
  - either class of signal can force `adaptive_search`, so a targeted weak setting changes search
    shape directly instead of only moving the family ranking

That keeps outer-loop search budget pointed at the same concrete repo bottlenecks the retained
curriculum is discovering, instead of flattening them back into a generic family-level pressure
signal.

## Files

- planner logic: [`unattended_controller.py`](/data/agentkernel/agent_kernel/unattended_controller.py)
- unattended orchestration entrypoint: [`run_unattended_campaign.py`](/data/agentkernel/scripts/run_unattended_campaign.py)
- focused tests: [`test_unattended_controller.py`](/data/agentkernel/tests/test_unattended_controller.py)
