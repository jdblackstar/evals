# Prime Lab Environments

This workspace is configured by `prime lab setup`.

## Local environments
- `pipeline-debugger` at `environments/pipeline_debugger`
- `repo-health-audit` at `environments/repo_health_audit`

## Common commands
Install local environments:

```bash
prime env install pipeline-debugger repo-health-audit -p ./environments
```

Run smoke evals:

```bash
prime eval run pipeline-debugger -m gpt-5-mini -n 5
prime eval run repo-health-audit -m gpt-5-mini -n 5
```

Run both via config:

```bash
prime eval run configs/eval/multi-env.toml
```
