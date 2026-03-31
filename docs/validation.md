# Validation

## Unit tests

Run:

```bash
pytest -q
```

Current status as of March 19, 2026:

```text
187 passed in 16.28s
```

## Coverage areas

The suite currently contains 187 passing tests across runtime, eval, verifier, curriculum, TOLBERT asset generation, CLI flags, extraction/proposal scripts, and improvement-cycle behavior.

Covered areas include:

- loop execution and verifier behavior
- Ollama JSON extraction and policy shape
- curriculum generation
- task-bank replay task generation
- operator extraction and transfer evaluation
- skill/tool/memory extraction
- eval harness metrics and comparisons
- TOLBERT asset generation
- implementation verification script behavior
- improvement cycle selection and finalization

## Smoke checks

Single task:

```bash
python scripts/run_agent.py --task-id hello_task
```

Expected result shape:

```text
task=hello_task success=True steps=1
```

Full eval:

```bash
python scripts/run_eval.py
```

Expected output shape:

```text
passed=... total=... pass_rate=...
average_steps=... average_success_steps=...
```

Implementation verification:

```bash
./scripts/verify_impl.sh
```

Expected phases:

- live task run
- full eval with TOLBERT and skills
- TOLBERT comparison
- TOLBERT feature-mode comparison
- skill, operator, and tool extraction refresh
- skill comparison
- baseline without TOLBERT or skills
- empirical gate validation over captured outputs

## Common failure points

- Ollama not running on `http://127.0.0.1:11434`
- `Qwen/Qwen3.5-9B` not available from the configured `vllm` server
- stale or missing TOLBERT asset paths
- CUDA requested in the TOLBERT env but not actually available
- artifact format drift between extractor/proposal output and replay loaders or retention logic
- env vars overriding workspace or trajectory paths unexpectedly
