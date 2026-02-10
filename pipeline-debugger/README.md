# Pipeline Debugger

## 1. What It Tests
`pipeline-debugger` measures whether an agent can diagnose and fix realistic ETL bugs across multiple files under tool-use constraints (file inspection/editing + shell commands), while preserving pipeline correctness guarantees.

## 2. How It Is Verified
Verification is programmatic and binary on core checks: all instance tests pass, `run_pipeline.py` exits 0, output schema conforms to `expected_schema.json`, outputs are deterministic across two runs, and test files remain unchanged via hash validation (`tests_hashes.json` + before/after hashes).

## 3. One Command To Run
From repo root (`/Users/josh/code/evals`):

```bash
uv run python pipeline-debugger/verifier/verify.py pipeline-debugger/instances/instance_001
```

Model eval via `verifiers` (OpenRouter-compatible endpoint example):

```bash
OPENROUTER_API_KEY=... PYTHONPATH=./pipeline-debugger uv run vf-eval pipeline-debugger -p ./pipeline-debugger -k OPENROUTER_API_KEY -b https://openrouter.ai/api/v1 -m openai/gpt-4.1-mini -n 5 -r 1
```

## 4. What It Reveals
It surfaces whether a model actually debugs root causes versus reward-hacking shortcuts (editing tests, hardcoding outputs, or shallow fixes), and provides partial-credit telemetry (`tests_passed`, `schema_valid`, `deterministic`, `test_files_untouched`) to analyze failure patterns.
