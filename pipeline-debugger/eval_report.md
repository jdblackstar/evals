# Eval Report: Pipeline Debugger

## Models Tested
- `openai/gpt-4.1-mini` (OpenRouter-compatible API)
- `deepseek/deepseek-chat` (OpenRouter-compatible API)

## Reproduction Command
```bash
OPENROUTER_API_KEY=... PYTHONPATH=. uv run vf-eval pipeline-debugger -p . -k OPENROUTER_API_KEY -b https://openrouter.ai/api/v1 -m <model> -n 15 -r 1 -a '{"max_instances": 15, "max_turns": 25}'
```

## Results
This report file is scaffolded, but model runs have not been executed in this sandbox session yet. Populate the table below after running the command above.

| Model | Pass rate | Avg tests_passed | Schema valid rate | Deterministic rate | Test files untouched rate | Avg tokens | Avg tool calls |
| --- | --- | --- | --- | --- | --- | --- | --- |
| openai/gpt-4.1-mini | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| deepseek/deepseek-chat | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## Failure Mode Breakdown (Template)
- Test patching attempts: TBD
- Hardcoded-output attempts: TBD
- Symptom-only fixes: TBD
- Performance vs bug count/difficulty: TBD
- README-first vs stack-trace-first behavior: TBD
- Token/tool-call efficiency tradeoff: TBD

## Interesting Qualitative Finding
TBD after runs.

## Limitations
- Report values are placeholders until endpoint-backed eval runs complete.
- Default launch set includes 15 instances; expanding to 20-50 should improve confidence intervals per bug type.
