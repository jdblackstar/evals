# repo-health-audit

Environment ID: `repo-health-audit`

## What it tests
This environment evaluates repository-health auditing under ambiguity and tool constraints. The model must generate `audit_report.json` with:
- factual metrics
- architecture judgments
- grounded evidence
- explicit degradation handling when tools are unavailable
- no edits to `target_repo/`

## Prime workflow
Install from this workspace:

```bash
prime env install repo-health-audit -p ./environments
```

Smoke evaluation:

```bash
prime eval run repo-health-audit -m gpt-5-mini -n 5
```

Rescore an existing `results.jsonl` with the current verifier:

```bash
uv run python -m repo_health_audit.verifier.rescore <path/to/results.jsonl>
```

## Package layout
- `repo_health_audit/environment.py`: Stateful tool environment + rubric wiring
- `repo_health_audit/verifier/verify.py`: audit verifier
- `repo_health_audit/verifier/rescore.py`: offline rescoring utility
- `repo_health_audit/instances/`: tiered audit instances
