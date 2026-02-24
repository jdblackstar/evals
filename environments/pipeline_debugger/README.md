# pipeline-debugger

Environment ID: `pipeline-debugger`

## What it tests
This environment measures whether a model can debug a broken Python ETL pipeline using tools, while preserving verifier invariants:
- `run_pipeline.py` exits 0
- `pytest tests/ -v` passes
- output schema matches `expected_schema.json`
- outputs are deterministic across reruns
- `tests/` files are untouched

## Prime workflow
Install from this workspace:

```bash
prime env install pipeline-debugger -p ./environments
```

Smoke evaluation:

```bash
prime eval run pipeline-debugger -m gpt-5-mini -n 5
```

## Package layout
- `pipeline_debugger/environment.py`: Stateful tool environment + rubric wiring
- `pipeline_debugger/verifier/verify.py`: deterministic verifier
- `pipeline_debugger/instances/`: task instances
- `pipeline_debugger/generator/`: mutation/template tooling
