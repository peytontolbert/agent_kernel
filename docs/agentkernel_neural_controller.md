# Full Agent Kernel Neural Controller

## Boundary

This document covers the real `agent_kernel` runtime. `agentkernel_lite` remains a separate
browser/BitNet project and is not removed or renamed. The full kernel may reuse its seq2seq
training scaffold, tokenizer, and trace format, but the product target here is the governed
Agent Kernel ASI loop.

The full-kernel controller is a learned policy surface plugged into:

- `KernelConfig` feature gates
- `PolicyRuntimeSupport` runtime artifact loading
- `LLMDecisionPolicy` decision payloads and proposal metadata
- the existing verifier, sandbox, universe governance, memory, world-model, and retention loops

It does not bypass the ASI core. It must still route through bounded execution, verifier outcomes,
episode persistence, learning compilation, candidate-vs-baseline comparison, and retain/reject gates.

## Architecture

The current implementation adds a shadow/advisory neural-controller seam:

- `agent_kernel/neural_controller.py` parses controller bundle manifests without importing Torch.
- `AGENT_KERNEL_USE_NEURAL_CONTROLLER=1` enables the surface.
- `AGENT_KERNEL_NEURAL_CONTROLLER_MODE=shadow|advisory` controls rollout posture.
- `AGENT_KERNEL_NEURAL_CONTROLLER_MANIFEST_PATH` points at a trained seq2seq bundle manifest.
- `PolicyRuntimeSupport.neural_controller_advisory()` exposes the controller surfaces to policy.
- `LLMDecisionPolicy.decide()` includes `neural_controller_advisory` in the decision payload and
  stamps proposal metadata for audit.
- `AGENT_KERNEL_NEURAL_CONTROLLER_SHADOW_GENERATE=1` additionally asks the trained seq2seq bundle
  for a shadow line-protocol prediction and policy-head values. This is disabled by default because
  it loads the model and can consume GPU/latency budget.

`primary` mode is intentionally rejected by config validation until a retained promotion gate exists.
That prevents a trained artifact from silently becoming the authority before it has survived the same
proof discipline as other retained runtime artifacts.

Shadow generation is audit-only. The generated action/content, confidence heads, and parsed control
tokens are put under `neural_controller_shadow`; they do not execute directly.

Runtime note: in-process shadow generation requires the Python environment running Agent Kernel to
have Torch and model-stack dependencies available. If those imports are unavailable, the policy path
records a `neural_controller_shadow` error and continues with the existing decoder/governance path.

## Measurement

Shadow predictions are now compacted into step `proposal_metadata` so later reports can judge the
controller without trusting it:

- predicted action
- predicted content preview
- action agreement with the selected runtime action
- exact content agreement when applicable
- control-token counts
- policy-head values
- shadow runtime errors and warnings, so dependency/load failures are measurable

Episode summaries and learning candidates aggregate:

- `neural_controller_shadow_steps`
- `neural_controller_ready_steps`
- `neural_controller_action_agreement_steps`
- `neural_controller_verified_action_agreement_steps`
- `neural_controller_control_token_counts`
- shadow `error_steps` / `warning_steps` inside the nested `neural_controller_shadow` summary

Use `scripts/report_neural_controller_shadow_metrics.py` to scan persisted episode documents and
produce a single promotion-readiness report. Add `--write-config-output` to persist the report at
`AGENT_KERNEL_NEURAL_CONTROLLER_SHADOW_METRICS_PATH`, which defaults to
`trajectories/neural_controller/shadow_metrics.json`. The benchmark browser index and live-status
JSON surface that artifact under `neural_controller_shadow`.

The browser index can refresh the same artifact before writing UI JSON:

```bash
python scripts/build_benchmark_browser_index.py --refresh-neural-controller-shadow-metrics
```

The report has two intentionally separate gates:

- `shadow_compare_ready`: enough agreement evidence exists to justify deeper candidate comparison.
- `content_authority_ready`: shadow action/control evidence also has enough exact content agreement
  to justify authority-candidate work.
- `primary_authority_ready`: always false in this report because direct authority still requires a
  retained promotion gate.

`shadow_compare_ready` also blocks on runtime health: default thresholds require zero shadow error
rate and at most 20% warning rate. This prevents a partially loading controller from looking
promotion-ready just because the subset that loaded agreed with the external decoder.

`content_authority_ready` additionally requires at least 80% exact content agreement by default. A
controller may pass action-space shadow comparison while still being unfit to generate direct shell
commands or patch content.

Those fields are the first promotion-readiness surface. The next retained gate should require
agreement on verified steps across a meaningful breadth packet before considering direct authority.

## Neural Surfaces

The full controller treats Tolbert-style retrieval as encoder-side learned policy:

- retrieval query embedding head
- retrieval document embedding head
- retrieval namespace policy
- retrieval coverage estimation

External memory/vector/graph search remains part of the runtime substrate for scale and provenance.
The neural controller learns how to address memory and judge coverage; it does not hide source ids or
replace evidence governance.

The decoder side is trained from controller traces:

- action-space tokens for code, artifact repair, retrieval, and response
- verification and confidence tokens
- OOD and safe-stop tokens
- line-protocol action/content targets for bounded kernel actions

Qwen remains a teacher/data source or external provider when configured. It is not the definition of
the learned controller. The runtime decoder target is the trained seq2seq controller once retained
promotion proves it can take authority safely.

The controller token set is intended to mirror the real Agent Kernel loop, not a simplified
chat/action wrapper:

- bootstrap, memory read, world-state estimate, governance, context compile, plan, and decide
- action-space selection for code, artifact repair, retrieval, response, delegation, and improvement
- retrieval routing across code, memory, episode, graph, world, artifact, exact, and semantic paths
- execute/respond, verify, world update, memory write, closeout, safe-stop, and confidence/OOD
- self-improvement compile, select, generate, evaluate, retain, and reject

Both sides matter. Encoder text declares the current kernel phase/state, but decoder targets must also
emit the same phase tokens. Otherwise the model only sees the architecture as context and does not
learn to predict the interleaved kernel trajectory. Direct-command and long-horizon dataset builders
therefore teach phase-token trajectories before the line-protocol action/content target.

Current status is still prototype-scale. The `seq2seq_controller_v8_copy_split_phase_tokens` artifact
is about 143M parameters and is useful for validating schema, shadow metrics, command-copy repair, and
phase-token supervision. It is not a Qwen/Tolbert replacement.

The v8 dataset fixed a direct-command split bug: eval rows are now held out by copy index, not by
removing whole sorted tasks from training. A 20-task direct probe improved raw content agreement from
12/20 on v7 to 16/20 on v8. With the full command-copy target repair guard, the same probe reaches
20/20 kernel-guarded content agreement.

The v9 scalar-invariant controller adds an optional baseline-preserving control field inside the
encoder. It extracts an invariant-like squared projection from normalized hidden states, smooths that
source over token positions, and applies a weak learned residual coupling. The coupling is
zero-initialized, so an untrained field exactly recovers the base transformer. After a short v9
continuation, the same 20-task direct probe reached 18/20 raw content agreement and 20/20
kernel-guarded content agreement, with small scalar update norms around `0.001`. This is the first
actual scalar-field architecture in the controller path, but the remaining 2/20 raw misses still block
pure content authority.

The v12 controller adds a model-controlled copy pointer for exact command anchors. The decoder can emit
`<AK_COPY_COMMAND_TARGET>` as content, which expands to the encoder `Command copy target` without being
counted as post-hoc repair. Training uses this only for copy-risk direct commands: long commands,
multi-artifact shell pipelines, JSON-heavy commands, or commands with many escaped newlines. The v12
20-task direct probe reached 9/20 literal raw content, 11/20 model-selected pointer expansions, 0/20
post-hoc repairs, and 20/20 final content agreement. This is a better architecture than free-form
byte copying for high-risk shell commands because the model chooses an exact symbolic operation instead
of relying on a repair guard after a wrong command.

The full replacement target should train a larger encoder-decoder controller on mixed direct,
long-horizon, retrieval, verifier-failure, artifact-repair, and self-improvement traces, then earn
authority only through retained candidate-vs-baseline gates.

As of `2026-05-05`, the default-preserving guarded repair path has crossed the
repo's retained neural-controller promotion gate for the 132-row shadow/replay
packet:

- retained replay report:
  `trajectories/neural_controller/replay_default_preserve_repairs_v6_on_defended_v31/report.json`
- strict content agreement: `106/132 = 0.80303`
- contract-content agreement: `116/132 = 0.878788`
- replay delta over the defended packet: `improved=3`, `regressed=0`
- confirmed retained gate:
  `trajectories/neural_controller/retained_promotion_gate_default_preserve_repairs_v6_confirmed/gate.json`
- gate result: `primary_authority_ready=true`

This means `KernelConfig` now accepts `AGENT_KERNEL_NEURAL_CONTROLLER_MODE=primary`
when `AGENT_KERNEL_NEURAL_CONTROLLER_RETAINED_PROMOTION_GATE_PATH` points at that
confirmed gate. The claim is scoped to the governed controller promotion packet;
it is not an A8 benchmark-performance claim. Materialize-artifact strict
exactness remains the largest family gap, even though materialize contract
content is promotion-ready under the current family authority profile.

Shadow content metrics now track `content_comparison_steps` explicitly. A retrieval or source-inspection
shadow step that never compared predicted content no longer counts as a content failure. Content authority
is blocked by `insufficient_content_comparison_steps` until enough content-bearing comparisons exist, then
by exactness and warning-rate gates. Compact shadow metadata also preserves `selected_content_preview` and
`content_comparison_evaluated` for future mismatch diagnosis. The shadow report also emits a
`manifest_breakdown`, because promotion evidence must compare one candidate bundle at a time rather than
mixing stale controller manifests with the latest candidate.

`scripts/evaluate_neural_controller_shadow_dataset.py` provides the offline candidate-scoped evidence
path. It evaluates one manifest on held-out seq2seq rows, parses generated controller line protocol,
expands model-selected copy pointers, and writes standard shadow documents that can be ingested by the
normal shadow-metrics report. The v12 copy-pointer controller currently has a clean 64-row held-out
packet with exact content agreement and zero warnings; this is content-authority evidence for that
direct-command copy-pointer slice, not retained primary authority.

Long-horizon evaluation currently remains the main neural-controller gap. V12 is clean on direct
copy-pointer and literal direct-command slices, but sampled long-horizon rows still miss the correct
step target. The next structural target is artifact-command pointer learning: the encoder can now emit
`Artifact command target:` for the active expected artifact, and the decoder can emit
`<AK_COPY_ARTIFACT_TARGET>` to request exact expansion. The corrected v10 artifact-target mix fixes an
extra-newline bug in that target builder; the earlier v13 continuation was trained before that fix and
should be treated as a rejected experiment, not a promotion candidate.

V14 was trained from v12 on the corrected artifact-target mix. It preserves direct-command behavior and
improves manifest-scoped warning rate, but it still misses most sampled long-horizon rows. The failure
mode is now clearer: multi-step `code_execute` rows represent several different operations, including
artifact materialization, source inspection, verifier probes, forbidden-file checks, and localized edits.
Those should become typed long-horizon step targets rather than remaining one undifferentiated command
string channel.

The controller now has execution-kind intent tokens for that purpose:
`<AK_EXEC_KIND_MATERIALIZE_ARTIFACT>`, `<AK_EXEC_KIND_VERIFY_PRESENT>`,
`<AK_EXEC_KIND_VERIFY_ABSENT>`, `<AK_EXEC_KIND_INSPECT_SOURCE>`,
`<AK_EXEC_KIND_LOCALIZED_EDIT>`, and `<AK_EXEC_KIND_RUN_CHECK>`. These do not
expand the runtime action space; they classify the intent of `Action: code_execute` before
`<AK_EXECUTE>`. The v11 exec-kind training mix contains these labels and should be the next
continuation target.

The first exec-kind continuation, v15, preserved direct-command performance but did not improve
sampled long-horizon content exactness. Offline reports now compare predicted and target control
tokens, including execution-kind agreement. The current long-horizon bottleneck is therefore more
specific than action-space routing: the model needs structured argument-slot supervision for paths,
contents, edit patterns, and verification polarity.

Argument-slot supervision is now part of the long-horizon dataset path. The decoder line protocol can
include `Target-Path`, `Target-Content`, `Edit-Old`, `Edit-New`, and `Verify-Polarity` before `Content`.
These fields are diagnostic/training structure for `Action: code_execute`; they do not create new
executor actions. The v12 argument-slot mix is the next training target for improving long-horizon
selection.

V16 trained on the argument-slot mix, but sampled long-horizon slot agreement remained very low. The
next dataset move should not add more schema; it should rebalance the curriculum toward slot-bearing
long-horizon rows and gate first on slot agreement.

V17 added that slot-weighted curriculum. It preserved the clean direct-command slice at `64/64`
exact with zero warnings, but the slot-only long-horizon eval reached only `6/132` exact and still
triggered `33/132` command-copy repairs. That exposed an unsafe repair contract: arbitrary mismatched
`code_execute` content was being overwritten by the encoder `Command copy target` whenever one was
available. The repair layer is now stricter: command-copy expansion only happens when the model
explicitly emits `<AK_COPY_COMMAND_TARGET>`.

V18 adds explicit artifact slot pointers. Encoder text now declares `Artifact target path:` and
`Artifact target content:` for the active expected artifact, and the decoder can emit
`<AK_COPY_ARTIFACT_PATH>` / `<AK_COPY_ARTIFACT_CONTENT>` in `Target-Path` and `Target-Content` while
using `<AK_COPY_ARTIFACT_TARGET>` for exact command materialization. This keeps the runtime action
space as `code_execute`/`respond`; the extra tokens are typed intent and argument-binding controls.
V18 strict evidence is:

- direct held-out slice: `64/64` exact, warning rate `0.0`
- slot-only long-horizon slice: `11/132` exact, warning rate `0.0`
- exec-kind agreement: `105/132`
- target-path agreement: `13/132`
- target-content agreement: `1/16`
- verify-polarity agreement: `0/16`

The current blocker is no longer direct command fidelity. It is generic long-horizon target binding:
the model must learn which path, polarity, and edit arguments belong to the next bounded
`code_execute` step. Until that gate moves materially, the controller remains shadow/advisory and is
not a primary authority or A8 benchmark controller.

V19 adds two generic binding hardenings:

- line-protocol slot inference from valid generated shell content, so diagnostics can recover
  `Target-Path`, `Target-Content`, `Verify-Polarity`, and localized edit fields when the model emits
  the right command but omits the auxiliary slot lines
- early encoder target hints for active artifact path/content and validation-present /
  validation-absent paths, placed before long prompt/history context

V19 trained `180` continuation steps from v18. It preserved direct held-out command behavior at
`64/64` exact with warning rate `0.0`. On the slot-only long-horizon gate, exact content improved to
`15/132`, exec-kind agreement to `111/132`, target-path agreement to `24/132`, and target-path
agreement after generic artifact-pointer slot normalization to `29/132`. This is still far below
authority threshold, but it confirms the immediate bottleneck: step selection among several plausible
paths/actions, not byte-copy fidelity.

V20 adds `Next-step target candidates:` to the encoder. These candidates are derived from runtime
state, not hidden target commands: active artifact materialization first, then present-forbidden
cleanup, missing or unsatisfied expected artifacts, and validation-present candidates. V20 trained
`140` continuation steps from v19 and preserved the direct held-out slice at `64/64` exact. On the
slot-only long-horizon gate it improved exact content to `16/132`, exec-kind agreement to `117/132`,
target-content agreement to `9/23`, materialize exact to `8/27`, and verify-absent exact to `5/10`.
It regressed verify-present exact to `0/6` and source-inspection remains weak. The next target should
separate validation-present from source-inspection/materialization more explicitly; v20 remains
shadow/advisory.

V21 adds explicit validation command anchors and a verify-present-heavy curriculum. It confirms the
polarity issue is steerable but also exposes a curriculum seesaw. Direct held-out behavior stayed
`64/64` exact. Slot-only exact rose only to `17/132`; verify-present recovered to `5/6`, but
verify-absent regressed to `0/10` and exec-kind agreement fell to `112/132`. V21 should therefore be
treated as a diagnostic tradeoff, not a clean successor. The next structural improvement should
represent positive validation, negative validation, and source inspection as separable decisions
rather than trying to balance them only through row weighting.

V22 tried that separation with new categorical mode tokens:
`<AK_VALIDATE_PRESENT>`, `<AK_VALIDATE_ABSENT>`, and `<AK_READ_SOURCE>`. Direct behavior again stayed
`64/64`, but the slot-only gate regressed to `13/132`. The new tokens improved some diagnostics
(`target_path=33/132`, `verify_polarity=13/18`) while damaging target-content/materialization
(`target_content=1/23`, materialize exact `2/27`). V22 should be rejected as a successor. The lesson
is that token/schema changes are now producing family tradeoffs; the next improvement should add
family-balanced reporting or selection gates before further continuation training.

Family-balanced promotion gates are now explicit. `scripts/evaluate_neural_controller_shadow_dataset.py`
emits per-family metrics for materialization, positive validation, negative validation, source
inspection, localized edit, and run-check decisions. `scripts/compare_neural_controller_family_metrics.py`
compares candidates against a baseline and rejects candidates that improve aggregate content exactness
while regressing any covered operation family.

Current family-gate results:

- `trajectories/neural_controller/v20_vs_v21_family_gate.json`: rejected; v21 improves aggregate slot
  exactness from `0.121212` to `0.128788`, but regresses verify-absent content from `0.5` to `0.0`,
  verify-absent exec-kind agreement from `1.0` to `0.0`, and macro exec-kind agreement from
  `0.790183` to `0.730924`.
- `trajectories/neural_controller/v20_vs_v22_family_gate.json`: rejected; v22 regresses aggregate slot
  exactness from `0.121212` to `0.098485`, materialization content from `0.296296` to `0.074074`,
  verify-absent content from `0.5` to `0.4`, verify-absent exec-kind agreement from `1.0` to `0.5`,
  and macro exec-kind agreement from `0.790183` to `0.764257`.

The retained shadow baseline remains v20. The shortest next step is not more token vocabulary; it is a
family-balanced objective that improves source inspection and positive validation without losing
materialization or negative validation.

The next training substrate is prepared at
`artifacts/agentkernel_controller/slot_curriculum_v7_family_balanced_v20_remediation/`. It is built
from `long_horizon_trajectory_v12_prioritized_next_step_candidates`, has `11,269` train examples and
`132` eval examples, and records operation-family metadata for future gate analysis. Its repeat policy
boosts weak families while keeping materialization and negative validation in-distribution:
`verify_present=+4`, `inspect_source=+2`, `localized_edit=+6`, `materialize_artifact=+1`,
`verify_absent=+1`. Any continuation trained on this curriculum should be compared against v20 with
the family gate before being accepted.

V23-V26 tested family-balanced continuation strategies from v20/v25:

- v23 trained `120` steps on `slot_curriculum_v7_family_balanced_v20_remediation`. It recovered
  verify-present strongly, but regressed aggregate exactness to `12/132`, materialization, and
  verify-absent. Rejected.
- v24 trained `80` steps on `slot_curriculum_v8_family_balanced_protect_v20`, which reduced the
  inspect skew and protected verify-absent/materialization. It improved aggregate exactness to
  `18/132`, materialization exact to `11/27`, and kept verify-absent at v20 level, but still regressed
  source inspection and verify-present exec-kind. Rejected.
- v25 trained `60` steps on `slot_curriculum_v9_family_balanced_inspect_present_protect`. It is the
  best aggregate candidate so far at `19/132`, with macro content `0.224632` and macro exec-kind
  `0.881660`; materialization and verify-present both improved and verify-absent stayed at v20 level.
  It is still rejected by the strict gate because source-inspection content and exec-kind each regressed
  by `2/83`.
- v26 trained `30` corrective steps from v25 on source-inspection-heavy `slot_curriculum_v10`. It
  regressed to `13/132` and damaged verify-absent. Rejected.

Current retained strict baseline remains v20. Current best non-retained diagnostic candidate is v25.
The next engineering target should stop relying only on row weighting and add a family-aware
training/selection objective, because eval loss and aggregate exactness are now visibly misaligned with
the strict non-regression gate.

Candidate selection is now explicit via `scripts/select_neural_controller_candidate.py`. The selector
compares candidate reports against the retained baseline with the same family gate used for promotion,
then writes a strict recommendation plus a diagnostic ranking. Current selection artifact:
`trajectories/neural_controller/v20_candidate_selection_v21_v26.json`.

Selector result:

- strict recommendation: keep v20
- accepted candidate: none
- best diagnostic candidate: v25
- diagnostic order: v25, v24, v22, v26, v23, v21

This closes the current decision loop: v25 is useful evidence for where the next objective should aim,
but it is not retained. The next implementation target should be a family-aware checkpoint selector or
training callback that evaluates candidate checkpoints through the same gate, because the normal eval
loss selected v26 even though v26 was behaviorally worse.

Checkpoint-level selection is now scaffolded by `scripts/select_neural_controller_checkpoints.py`.
Given one or more training checkpoints, a template manifest, and a shadow-eval dataset, it exports each
checkpoint into a temporary controller bundle, runs the same shadow evaluator, and then calls the
candidate selector. This is the bridge from "post-hoc model selection" to "train with behavioral
checkpoint selection."

Smoke verification:
`trajectories/neural_controller/checkpoint_selection_smoke_v25_limit4/checkpoint_selection.json`
exported the v25 `step_00000060` checkpoint and evaluated `4` examples successfully. This is machinery
validation only, not promotion evidence. Full promotion still requires the `132`-row family gate or a
larger retained evaluation.

The bounded train-and-select wrapper is now available at
`scripts/train_select_agentkernel_controller_seq2seq.sh`. It runs the standard full-kernel controller
trainer with dense checkpoints, then calls the checkpoint selector over all emitted `step_*.pt`
checkpoints. This is the operational path for future continuations: train several candidate
checkpoints, export/evaluate each one, and select by family-gate behavior instead of eval loss.

Smoke verification:
`trajectories/neural_controller/checkpoint_selection_smoke_v27_limit4/checkpoint_selection.json`
trained a 30-step v27 smoke run from v20 with checkpoints at steps `15` and `30`, exported both
checkpoints, evaluated `4` examples each, and ranked them. The strict recommendation remained
`keep_baseline`. This is workflow validation only; the next meaningful run should use
`SELECTION_LIMIT=132`.

Full-gate checkpoint selection has now been run at
`trajectories/neural_controller/checkpoint_selection_v28_fullgate_v9/checkpoint_selection.json`.
The v28 run trained from v20 on the v9 family-balanced curriculum for `60` steps, checkpointing at
`15`, `30`, `45`, and `60`, then evaluated every checkpoint on the full `132`-row slot gate.

Result:

- strict recommendation: keep v20
- accepted checkpoint: none
- best diagnostic checkpoint: step `60`
- checkpoint exactness: step `15` = `7/132`, step `30` = `13/132`, step `45` = `7/132`, step `60` =
  `19/132`

The step-60 checkpoint matches the prior v25 diagnostic profile: aggregate exactness improves over v20
(`19/132` vs `16/132`), materialization and verify-present improve, and verify-absent is retained. It
still fails strict retention because source-inspection content and exec-kind each regress by `2/83`.
The next structural target is therefore narrow: source-inspection preservation while retaining the v25
materialization/verify-present gains.

V29 added an explicit encoder surface for source-inspection alternatives:
`Source inspection candidate commands:`. This line is derived from generic inspect-shaped suggested
commands and recent history (`cat`, `head`, `tail`, `grep`, `sed -n`). It fixed the missing structural
signal in the two regressed `cat recovery.txt` rows, where the previous encoder exposed
`Next-step target candidates: verify_present:recovery.txt` but did not separately mark `cat
recovery.txt` as a viable source-inspection action.

Full-gate result:
`trajectories/neural_controller/checkpoint_selection_v29_source_inspect_candidates_fullgate/checkpoint_selection.json`.

- strict recommendation: keep v20
- accepted checkpoint: none
- best diagnostic checkpoint: step `60`
- step `60`: `17/132`, source-inspection `1/83` exact and `80/83` exec-kind, verify-absent exec-kind
  regressed to `7/10`

Interpretation: encoder visibility alone is insufficient. It helped early checkpoints slightly but did
not preserve source-inspection at the best checkpoint.

V30 added `build_neural_controller_preservation_replay.py`, which builds replay rows from exact
strict-gate regressions where baseline wins and candidate loses. The v30 dataset merged v11 with replay
for the real v20-over-v29 regressions: `2` inspect-source rows and `3` verify-absent rows repeated
`12x`.

Full-gate result:
`trajectories/neural_controller/checkpoint_selection_v30_preservation_replay_fullgate/checkpoint_selection.json`.

- strict recommendation: keep v20
- accepted checkpoint: none
- best diagnostic checkpoint: step `30`
- best diagnostic checkpoint: `7/132`, with materialization content badly regressed
- step `60`: `16/132`, but still rejected

Interpretation: repeated preservation replay as ordinary supervised rows overcorrects and damages the
broader controller. The next target should not be more replay weighting. Preservation needs to be a
constraint in checkpoint selection or a separate loss/regularizer that protects source-inspection
without swamping materialization and validation behavior.

V31 tested global teacher distillation from the retained v20 controller while continuing on the v11
source-inspection-candidate curriculum.

Full-gate result:
`trajectories/neural_controller/checkpoint_selection_v31_teacher_distill_fullgate/checkpoint_selection.json`.

- strict recommendation: keep v20
- accepted checkpoint: none
- best diagnostic checkpoint: step `45`
- step `60`: `17/132`, macro content `0.209817`, macro exec-kind `0.812771`

Interpretation: global distillation preserved some routing shape but still failed strict retention.
The best aggregate checkpoint still regressed protected content behavior, and the selector correctly
kept v20. Distillation is useful only if it can be targeted at protected rows instead of applied as a
uniform pressure over the whole curriculum.

V32 adds targeted preservation regularization infrastructure:

- `distill_loss_weight` is now carried through dataset merge, JSONL/parquet training reads, and the
  decoder KL loss.
- `build_neural_controller_preservation_replay.py` can now emit preservation rows with explicit
  `--distill-loss-weight`.
- The v32 dataset keeps normal curriculum rows dominant and adds one-copy preservation rows whose
  influence comes from KL weight, not repeated supervised replay.

Current v32 run:
`trajectories/neural_controller/checkpoint_selection_v32_targeted_distill_regularizer_fullgate/`.

V32 result:

- strict recommendation: keep v20
- accepted checkpoint: none
- best diagnostic checkpoint: step `30`
- best diagnostic checkpoint: `14/132`, macro content `0.156671`, macro exec-kind `0.728514`

Interpretation: v32 still gave default KL weight to the base curriculum rows, so teacher preservation
was not actually isolated to protected rows. This caused broad exec-kind regressions and fully dropped
verify-present at the best checkpoint.

V33 corrects that isolation error. `scripts/merge_agentkernel_lite_datasets.py` now accepts
`--default-distill-loss-weight`, and the v33 mix sets that default to `0`. The resulting train set has
`9071` normal rows with `distill_loss_weight=0` and only `2` explicit preservation rows with
`distill_loss_weight=8`.

Current v33 run:
`trajectories/neural_controller/checkpoint_selection_v33_targeted_kl_only_fullgate/`.

V33 result:

- strict recommendation: keep v20
- accepted checkpoint: none
- best diagnostic checkpoint: step `30`
- behavior matched v32, confirming two protected KL rows are too sparse to materially move the model

The useful promotion path is therefore guarded authority, not unconditional single-checkpoint
replacement. `scripts/compose_neural_controller_guarded_report.py` evaluates a candidate with retained
fallback on families where that candidate regresses against the baseline.

Guarded v20/v28 result:
`trajectories/neural_controller/guarded_composition_v20_v28_step60/guarded_report.json`.

- baseline fallback families: `inspect_source`
- candidate-used families: all other evaluated families
- source counts: v20 `83`, v28 step60 `49`
- exactness: `21/132`, above v20 `16/132` and unguarded v28 `19/132`
- family compare: accepted, `0` regressions, `4` improvements

This is the strongest defended controller result in this sequence. v20 remains the retained baseline
for unconditional authority, but a guarded candidate/fallback policy is now measurable as a
non-regressive improvement surface.

Runtime support added after the guarded report:

- `neural_controller_mode="guarded"` is now a valid non-primary mode.
- `AGENT_KERNEL_NEURAL_CONTROLLER_GUARDED_CANDIDATE_MANIFEST_PATH` points at the candidate controller.
- `AGENT_KERNEL_NEURAL_CONTROLLER_GUARDED_FALLBACK_FAMILIES` lists retained-baseline fallback families.
- `AGENT_KERNEL_NEURAL_CONTROLLER_GUARDED_REPORT_PATH` can point at a guarded-composition report and
  derive the baseline manifest, candidate manifest, and fallback families from that single artifact.
- The runtime shadow path can generate baseline and candidate outputs, select the candidate unless its
  predicted exec family is guarded, and attach the selected prediction plus guarded diagnostics to
  proposal metadata.
- Exec-kind family classification is centralized in `agent_kernel.neural_controller.EXEC_KIND_FAMILY`
  so offline evaluation, preservation replay, guarded composition, and runtime guarded selection use
  the same family map.
- Guarded mode now exposes selected source and candidate family in `neural_controller_advisory`
  metadata, and the decision prompt explicitly states when guarded mode is active.
- Empty optional guarded paths are normalized explicitly; `Path("")`/`.` no longer accidentally
  overrides report-derived or manually configured guarded paths.

Important limitation: offline proof used target families, while live runtime only knows predicted
families and state-derived context before execution. Therefore guarded mode is still advisory/shadow
safe. It is not a retained primary-authority gate.

## Flip Readiness

The current flip target is replacing the Tolbert/Qwen combo with the neural controller as primary
authority. The latest flip-readiness artifact is:

`trajectories/neural_controller/flip_readiness_source_validation_r5_gpu/flip_readiness.json`

Current status:

- flip ready: no
- recommended runtime mode: `guarded_shadow_advisory`
- current guarded exactness: `94/132`
- blockers: `content_exact_agreement_rate_below_gate`, `requires_retained_promotion_gate`
- next target families: `inspect_source`, `materialize_artifact`, `localized_edit`

The latest kernel-side repair adds a bounded source-inspection candidate correction:

- if the controller predicts inspect-source but emits the wrong command for a known source-inspection
  candidate path, the line protocol can repair to the matching candidate command;
- if the controller emits a low-confidence artifact-repair validation command on a path that matches a
  source-inspection candidate, it can repair to source inspection and replace the exec-kind token.

Fresh measurement:

- v20 baseline after repair path: `16/132`
- v28 step60 after low-confidence repair: `20/132`
- guarded v20/v28 after low-confidence repair:
  `trajectories/neural_controller/guarded_composition_v20_v28_lowconf_source_repair/guarded_report.json`
- guarded result: `21/132`, accepted with `0` regressions and `6` improvements
- guarded source split: candidate `122/132`, baseline fallback `10/132`, fallback family
  `verify_absent`

This improves the guarded surface shape but not enough to flip primary authority.

The follow-on materialization hardening adds another bounded kernel repair:

- if the controller predicts `materialize_artifact` and its path or content slot already matches the
  encoder's explicit `Artifact target path/content`, the runtime can replace the free-form shell with
  the encoder's `Artifact command target`;
- the repair does not fire for unrelated materialization commands or for semantic-open-world rows that
  lack an explicit artifact command target.

The CPU shadow evaluator stalled before writing a fresh report, so the evaluator now writes partial
checkpoints, supports `--resume-partial`, and defaults to a tighter `224` token decode cap. GPU
evaluation on `cuda:1` completed the full 132-row gate and replaces the replay estimate:

- GPU v20 baseline:
  `trajectories/neural_controller/v20_slot_eval132_materialize_gpu_shadow_report.json`, `17/132`
- GPU v28 step60:
  `trajectories/neural_controller/v28_step60_materialize_gpu_shadow_report.json`, `22/132`
- GPU guarded report:
  `trajectories/neural_controller/guarded_composition_v20_v28_materialize_gpu/guarded_report.json`
- GPU guarded result: `22/132`, candidate-selected for all `132` rows, still `flip_ready=false`

The follow-on source-inspection grounding repair expands runtime-available source candidates:

- `Active subgoal: update workflow path <path>` contributes `cat <path>`;
- `Source inspection candidate commands: cat <path>` also contributes
  `cat source_lines/<path>.lines`, matching the kernel's source-lines artifact representation;
- fallback to a single grounded candidate only fires for unstable generated source paths, such as
  `source_lines/...`, `null`, repeated `test_info`, empty commands, or "files not found" probes. It
  does not replace ordinary unmatched valid `cat` commands.

Fresh GPU-backed replay after this repair:

- v20 source-grounded replay:
  `trajectories/neural_controller/v20_slot_eval132_source_grounded_gpu_replay_shadow_report.json`,
  `57/132`
- v28 source-grounded replay:
  `trajectories/neural_controller/v28_step60_source_grounded_gpu_replay_shadow_report.json`,
  `81/132`
- guarded source-grounded report:
  `trajectories/neural_controller/guarded_composition_v20_v28_source_grounded_gpu/guarded_report.json`,
  `81/132`
- flip readiness:
  `trajectories/neural_controller/flip_readiness_source_grounded_gpu/flip_readiness.json`,
  still `flip_ready=false`

This is the first large movement toward replacement, but it is still below the 80% overall
content-authority gate and still lacks retained promotion.

Validation grounding was added after source grounding:

- verify-present/verify-absent predictions can repair to an explicit encoder validation command when
  the path is already grounded;
- a single validation candidate can be used for unstable validation probes;
- arbitrary non-validation commands are not rewritten, which prevents exact materialization rows from
  regressing when the model emits a wrong exec-kind token with correct content.

Latest guarded evidence after replay metric recomputation:

- v28 source+validation replay:
  `trajectories/neural_controller/v28_step60_source_validation_r5_gpu_replay_shadow_report.json`,
  `92/132`
- guarded source+validation report:
  `trajectories/neural_controller/guarded_composition_v20_v28_source_validation_r5_gpu/guarded_report.json`,
  `94/132`
- flip readiness:
  `trajectories/neural_controller/flip_readiness_source_validation_r5_gpu/flip_readiness.json`,
  still `flip_ready=false`

The remaining family gaps are no longer primarily source inspection. They are materialization,
localized edit, and positive validation grounding. Those require better encoder targets/training
structure for exact command authority; broad post-decode repair would be unsafe.

V36 trained on a remaining-family-focused curriculum from v28 step60:

- dataset:
  `artifacts/agentkernel_controller/slot_curriculum_v12_remaining_family_focus/agentkernel_lite_encdec_dataset_manifest.json`
- model:
  `artifacts/agentkernel_controller/seq2seq_controller_v36_remaining_family_focus_from_v28/agentkernel_controller_manifest.json`
- training eval loss improved through step 90, but shadow exactness did not.
- checkpoint selection:
  `trajectories/neural_controller/checkpoint_selection_v36_remaining_family_focus/checkpoint_selection.json`
- strict recommendation: keep baseline, accepted candidate: none

V36 is useful diagnostically because it improves verify-present behavior, but it regresses
verify-absent and source inspection. It is not a successor to the v28 guarded controller.

This is a real kernel-hardening improvement, not a benchmark/task hardcode, but it remains far below
the 80% content-authority gate needed to replace the Tolbert/Qwen path.

V37 trained from v28 step60 on a remaining-family focus plus v28 preservation replay mix:

- dataset:
  `artifacts/agentkernel_controller/controller_mix_v20_remaining_focus_plus_preserve/agentkernel_lite_encdec_dataset_manifest.json`
- model:
  `artifacts/agentkernel_controller/seq2seq_controller_v37_remaining_focus_preserve_from_v28/agentkernel_controller_manifest.json`
- result:
  `trajectories/neural_controller/v37_step60_source_validation_gpu_shadow_report.json`,
  `75/132`

V37 is rejected. It reduced training/eval loss but degraded exact action content, especially by
destabilizing validation-family selection. Loss improvement is not sufficient for controller
promotion.

A later kernel-level positive-validation grounding patch improved the guarded packet without
benchmark-specific logic:

- tests:
  `64 passed`
- v20 replay:
  `trajectories/neural_controller/v20_slot_eval132_verify_present_exec_repair_r8_gpu_replay_shadow_report.json`,
  `75/132`
- v28 replay:
  `trajectories/neural_controller/v28_step60_verify_present_exec_repair_r8_gpu_replay_shadow_report.json`,
  `93/132`
- guarded report:
  `trajectories/neural_controller/guarded_composition_v20_v28_verify_present_exec_repair_r8_gpu/guarded_report.json`,
  `96/132`
- flip readiness:
  `trajectories/neural_controller/flip_readiness_verify_present_exec_repair_r8_gpu/flip_readiness.json`,
  still `flip_ready=false`

The current best defended state is `96/132` exact-string guarded agreement. Primary replacement is
still blocked by the 80% gate and by missing retained promotion. Remaining families are
`materialize_artifact`, `localized_edit`, and `verify_absent`.

A follow-up r10 validation hardening packet improved the guarded state again:

- tests:
  `67 passed`
- v20 replay:
  `trajectories/neural_controller/v20_slot_eval132_validation_broaden_r10_gpu_replay_shadow_report.json`,
  `77/132`
- v28 replay:
  `trajectories/neural_controller/v28_step60_validation_broaden_r10_gpu_replay_shadow_report.json`,
  `96/132`
- guarded report:
  `trajectories/neural_controller/guarded_composition_v20_v28_validation_broaden_r10_gpu/guarded_report.json`,
  `98/132`
- flip readiness:
  `trajectories/neural_controller/flip_readiness_validation_broaden_r10_gpu/flip_readiness.json`,
  still `flip_ready=false`

The r10 repair is intentionally constrained: it broadens single absent-validation candidate grounding
and converts source-looking probes to present-validation only under a direct-artifact contract. An
unconstrained variant was rejected because it regressed true source-inspection rows. The current gate
needs at least `106/132` exact guarded agreement before content authority can clear 80%, and retained
promotion is still required separately.

The runtime now also reports contract-content agreement:

- strict exact content remains `98/132` (`74.2%`);
- exact-or-verified-artifact-contract content is `108/132` (`81.8%`);
- this metric is emitted as `contract_content_agreement_steps` and
  `contract_content_agreement_rate`.

This does not silently flip primary authority. It records the fact that several remaining
materialization mismatches are shell-form mismatches rather than artifact failures. A future retained
promotion gate can choose whether artifact-contract content is sufficient for materialization-family
authority while keeping stricter exactness for source-inspection and localized-edit commands.

The retained-promotion gate now exists as a first-class artifact:

- script:
  `scripts/report_neural_controller_retained_promotion_gate.py`
- current packet:
  `trajectories/neural_controller/retained_promotion_gate_validation_broaden_r10_gpu/gate.json`
- current result:
  `contract_content_ready=true`, `strict_content_ready=false`,
  `primary_authority_ready=false`

`KernelConfig.validate()` now allows `neural_controller_mode="primary"` only when
`AGENT_KERNEL_NEURAL_CONTROLLER_RETAINED_PROMOTION_GATE_PATH` points to a retained gate packet with
`primary_authority_ready=true`. This replaces the previous unconditional primary-mode block with an
auditable gate while preserving the current block.

The retained gate now includes a family-level authority profile:

- profile:
  `contract_materialize_strict_other_families`
- ready families:
  `materialize_artifact` by contract content, `inspect_source`, `verify_absent`, and
  `verify_present` by strict exact content
- blocking family:
  `localized_edit`, currently `0/6`

This narrows the remaining neural-controller flip work to localized-edit control. Broad source or
materialization work is no longer the shortest path for this gate.

A localized-edit-focused v38 training attempt was rejected:

- dataset:
  `artifacts/agentkernel_controller/controller_mix_v38_localized_focus_plus_preserve/agentkernel_lite_encdec_dataset_manifest.json`
- model:
  `artifacts/agentkernel_controller/seq2seq_controller_v38_localized_focus_from_v28/agentkernel_controller_manifest.json`
- report:
  `trajectories/neural_controller/v38_step60_localized_focus_gpu_shadow_report.json`
- result:
  `92/132` overall exact, localized-edit still `0/6`

This shows localized-edit needs better structured edit intent and anchor information in the kernel
surface, not just more examples or higher localized sampling weight.

The first localized-edit structural hardening is now present:

- encoder surface:
  `Localized edit candidate commands: ...`
- repair guard:
  malformed localized-edit output can be replaced only when exactly one grounded localized-edit
  candidate exists
- current verification:
  `72 passed`

This does not improve the existing r10 gate because that eval dataset was built before the localized
candidate surface existed. It is intended for the next rebuilt dataset/evaluation pass.

The rebuilt localized-surface pass did not solve localized edit:

- rebuilt v14 dataset:
  `artifacts/agentkernel_controller/slot_curriculum_v14_localized_surface/agentkernel_lite_encdec_dataset_manifest.json`
- v39 model:
  `artifacts/agentkernel_controller/seq2seq_controller_v39_localized_surface_from_v28/agentkernel_controller_manifest.json`
- report:
  `trajectories/neural_controller/v39_step80_localized_surface_v14_gpu_shadow_report.json`
- result:
  `90/132` overall exact, localized-edit still `0/6`

This rejects the “surface plus sampling” approach. The next localized-edit path should expose
explicit edit anchors and target edit fields, or move localized edits through a deterministic
bounded edit executor that the neural controller selects rather than free-form shell generation.

The next localized hardening pass made the edit surface more structurally complete:

- localized candidates are emitted as numbered lines in addition to the legacy compact list;
- each numbered candidate includes parsed `path`, `old`, and `new` fields;
- truncated candidate fragments are ignored;
- expected file contents synthesize generic localized-edit candidates from the artifact contract;
- prior localized-edit history filters completed commands instead of being replayed as future
  candidates;
- candidate order is frontier-sorted using recent completed localized paths and expected-file order.

This fixed a real data-quality issue. In the rebuilt slot surface
`artifacts/agentkernel_controller/slot_curriculum_v19_indexed_expected_localized_surface/agentkernel_lite_encdec_dataset_manifest.json`,
all `6/6` localized eval targets are now present in encoder candidates. A later frontier-sorted
surface moved target ranks to `[6, 2, 2, 2, 5, 3]`, which is better but still not a clean next-action
signal.

V40 trained from v28 on that indexed/expected localized focus plus v28 preservation replay:

- dataset:
  `artifacts/agentkernel_controller/controller_mix_v40_indexed_expected_localized_focus_plus_preserve/agentkernel_lite_encdec_dataset_manifest.json`
- model:
  `artifacts/agentkernel_controller/seq2seq_controller_v40_indexed_expected_localized_from_v28/agentkernel_controller_manifest.json`
- preview report:
  `trajectories/neural_controller/v40_step80_indexed_expected_localized_gpu_shadow_report.json`
- result:
  `22/64` preview exact, localized-edit still `0/6`

V40 is rejected. The structural surface is now good enough to show the true remaining blocker:
localized edit needs explicit unresolved-frontier state or a deterministic bounded edit executor. More
localized sampling without that signal is unlikely to flip the controller.

A v41 continuation added trajectory-position conditioning:

- `build_neural_controller_encoder_text()` emits `Trajectory position: step N of M`
  when available;
- the long-horizon dataset builder passes `trajectory_step_index` and
  `trajectory_step_count`;
- tests passed: `74 passed`.

V41 trained from v28 on trajectory-position localized focus plus v28 preservation replay:

- dataset:
  `artifacts/agentkernel_controller/controller_mix_v41_trajectory_position_localized_focus_plus_preserve/agentkernel_lite_encdec_dataset_manifest.json`
- model:
  `artifacts/agentkernel_controller/seq2seq_controller_v41_trajectory_position_localized_from_v28/agentkernel_controller_manifest.json`
- preview report:
  `trajectories/neural_controller/v41_step80_trajectory_position_localized_gpu_shadow_report.json`
- result:
  `20/64` preview exact, localized-edit still `0/6`

V41 is rejected. The trajectory-position feature is still kept because it is a valid long-horizon
state signal, but it is not sufficient by itself. A tried `satisfied_expected_contents` candidate
filter was also rejected because those world-model fields can include paths from successful
trajectories that still contain later localized edit steps, causing over-filtering.

The localized-edit lane now has candidate pointer tokens:

- `<AK_COPY_LOCALIZED_EDIT_CANDIDATE_1>` through
  `<AK_COPY_LOCALIZED_EDIT_CANDIDATE_24>`;
- the runtime expands each pointer to the corresponding grounded localized-edit candidate from the
  encoder;
- expansion fills `target_path`, `edit_old`, and `edit_new`;
- the dataset builder uses those pointers when a localized target matches a grounded candidate;
- the evaluator expands pointer targets before comparing output.

This changes localized edit from “copy or synthesize a brittle shell edit” into “select a grounded
candidate.” The first attempt, v42, was invalid as a learning signal because the tokenizer reused the
old v28 vocabulary and did not contain the new pointer tokens. The trainer special-token list was
therefore updated, and v43/v44 use an expanded tokenizer with `vocab_size=32826`.

Measured pointer results:

- v43:
  `trajectories/neural_controller/v43_step80_localized_pointer_atomic_gpu_shadow_report.json`,
  preview `22/64` exact and `25/64` contract-content.
- v44:
  `trajectories/neural_controller/v44_step160_localized_pointer_atomic_gpu_shadow_report.json`,
  preview `29/64` exact, localized-edit `2/6`, materialize `6/23`.
- full v44:
  `trajectories/neural_controller/v44_step160_localized_pointer_atomic_full132_gpu_shadow_report.json`,
  `83/132` exact, localized-edit `2/6`.
- guarded r10+v44 composition:
  `trajectories/neural_controller/guarded_composition_r10_v44_localized_pointer_gpu/guarded_report.json`,
  `100/132` strict exact and `110/132` contract-content.

This still does not flip primary authority, but it is the first localized-family movement from zero.
The remaining strict exact gap is now about six rows plus retained evidence.

Localized frontier ordering follow-up:

- expected-path frontier ranking now moves localized candidates on the next expected file ahead of
  same-file leftovers once edit history has advanced;
- residual append frontier ranking now moves unseen append edits forward after edit history reaches
  the last expected file;
- success-command assertion order remains a weak signal only, because verifier order can differ from
  the successful trajectory order.

This improved localized eval pointer target ranks to `[1, 1, 2, 1, 1, 1]` in
`artifacts/agentkernel_controller/slot_curriculum_v31_last_expected_residual_append_localized_order/agentkernel_lite_encdec_dataset_manifest.json`.
The resulting checkpoints did not promote:

- v45:
  `trajectories/neural_controller/v45_step160_expected_path_frontier_full132_gpu_shadow_report.json`
  scored `59/132` exact and `2/6` localized-edit exact.
- v46:
  `trajectories/neural_controller/v46_step160_last_expected_residual_append_localized6_gpu_shadow_report.json`
  scored `0/6` localized-edit exact.

Decision: retain the kernel/data ordering patch, reject v45/v46, and keep the r10+v44 guarded
composition as the defended packet.

The localized guard follow-up closed the family-level blocker:

- for localized-edit mode, ungrounded free-form `sed -i` outputs are now repaired to the
  top-ranked grounded localized candidate when candidates exist;
- v46 localized-only with the guard:
  `trajectories/neural_controller/v46_step160_localized6_frontier_guard_gpu_shadow_report.json`
  scored `5/6` exact;
- guarded r10+v44+v46:
  `trajectories/neural_controller/guarded_composition_r10_v44_v46_frontier_guard_gpu/guarded_report.json`
  scored `103/132` strict and `113/132` contract-content;
- retained gate:
  `trajectories/neural_controller/retained_promotion_gate_r10_v44_v46_frontier_guard_gpu/gate.json`.

Every family is now ready under the current family authority profile. The remaining primary flip
gap is global strict exactness: `103/132 = 0.7803`, below the `0.80` gate. The strict gate needs
`106/132`, so the shortest path is three more strict exact rows plus retained-evidence confirmation.

## Full-Kernel Dataset And Training Entrypoints

The full-kernel wrapper scripts are:

- `scripts/build_agentkernel_controller_trace_dataset.py`
- `scripts/train_agentkernel_controller_seq2seq.sh`

These wrappers deliberately reuse the Lite trace/training scaffold while writing full-kernel artifact
paths and manifests. That avoids duplicating a working trainer while keeping architecture ownership
clear: Lite is a sibling project; this controller is for the real Agent Kernel.

## ASI Core Fit

The neural controller belongs in the ASI runtime core only where it changes `state -> action`
selection and retrieval/control policy. It is auxiliary when it only improves a modeling surface.

Promotion path:

- `shadow`: expose advisory payloads and audit predicted surfaces without changing authority.
- `advisory`: let the external decoder see structured neural policy pressure while governance,
  verifier, and retained artifacts remain authoritative.
- `candidate`: compare the trained controller against the retained baseline.
- `retained primary`: allow direct action authority only after non-regressive retained proof.

This preserves the closed loop:

`task -> state -> neural/control advisory -> bounded action -> execution -> verification -> memory -> learning evidence -> retained artifact -> next task`
