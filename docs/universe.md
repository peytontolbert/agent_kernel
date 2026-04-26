# Universe Layer

The universe layer is the slow-moving governance surface above the task-local world model.

It now has two artifact roles:

- `universe_constitution`: near-immutable machine-law invariants
- `operating_envelope`: learned and attested runtime operating assumptions

Legacy `universe_contract` artifacts are still loaded as a backward-compatible fallback, but the runtime prefers the split artifacts when they exist.

**Artifact Paths**

- `trajectories/universe/universe_constitution.json`
- `trajectories/universe/operating_envelope.json`
- legacy fallback: `trajectories/universe/universe_contract.json`

Improvement-cycle entrypoints can now target either split surface directly:

- `run_improvement_cycle.py --subsystem universe_constitution`
- `run_improvement_cycle.py --subsystem operating_envelope`

**Universe Constitution**

`universe_constitution` holds the narrow set of rules that should change rarely:

- verifier requirement
- bounded-action requirement
- destructive-reset prohibitions
- reversibility bias
- preserved-artifact protections
- stable command-pattern exclusions and preferred bounded prefixes

Schema:

- `spec_version`: `asi_v1`
- `artifact_kind`: `universe_constitution`
- `lifecycle_state`: `proposed`, `retained`, or `rejected`
- `control_schema`: `universe_constitution_v1`
- `retention_gate`
- `governance`
- `invariants`
- `forbidden_command_patterns`
- `preferred_command_prefixes`
- `proposals`

Retention expectations:

- multiple prior retained universe wins
- cross-family support
- constitution cooldown before another constitutional mutation

**Operating Envelope**

`operating_envelope` holds the adaptive runtime envelope around those laws:

- structured `action_risk_controls`
- `environment_assumptions`
- HTTP allowlists
- writable-path scope policies
- toolchain requirements
- learned calibration priors

Schema:

- `spec_version`: `asi_v1`
- `artifact_kind`: `operating_envelope`
- `lifecycle_state`: `proposed`, `retained`, or `rejected`
- `control_schema`: `operating_envelope_v1`
- `retention_gate`
- `action_risk_controls`
- `environment_assumptions`
- `allowed_http_hosts`
- `writable_path_prefixes`
- `toolchain_requirements`
- `learned_calibration_priors`
- `proposals`

Retention expectations:

- outcome-weighted non-regression
- faster adaptation than the constitution
- support from multiple retained runtime episodes when possible

**Runtime Summary**

`agent_kernel/universe_model.py` now computes:

- `constitution`
- `operating_envelope`
- `constitutional_compliance`
- `envelope_alignment`
- `runtime_attestation`
- `plan_risk_summary`

The runtime still exposes legacy top-level compatibility fields like:

- `invariants`
- `forbidden_command_patterns`
- `preferred_command_prefixes`
- `action_risk_controls`
- `environment_assumptions`
- `environment_alignment`

**Runtime Attestation**

The universe runtime persists attestation-oriented evidence about the actual execution environment:

- repo dirty/clean state
- writable paths
- actual network reach mode
- available hosts
- toolchain availability
- sandbox / containment mode

This evidence is used to distinguish:

- constitution compliance: whether the machine-law layer remains intact
- envelope alignment: whether the current runtime matches the retained operating assumptions

**Plan Risk**

Command-level governance scoring still exists, but the universe layer is no longer only advisory. `UniverseModel.should_block_command(...)` now turns deterministic conflicts into explicit pre-execution block decisions, and `AgentKernel.run_task()` consults that result before `Sandbox.run()`. Blocked commands fail the step as `governance_rejected` instead of using sandbox rejection as the first hard stop.

The deterministic block surface currently covers cases such as:

- forbidden command-pattern matches
- network access conflicts
- writable-path scope conflicts
- git-write conflicts
- destructive or privileged command classes
- remote execution
- unbounded execution under bounded-step governance

The universe layer also evaluates the plan surface across the available task commands:

- cumulative mutation surface
- rollback coverage
- verifier coverage
- network escalation
- git escalation
- scope escape risk

This is meant to keep the universe layer focused on stable operating boundaries rather than only per-command string heuristics.
