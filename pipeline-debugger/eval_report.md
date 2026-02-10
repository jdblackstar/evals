# Eval Report: Pipeline Debugger

## Models Tested
- `openai/gpt-4.1-mini` (OpenRouter-compatible API)
- `deepseek/deepseek-chat` (OpenRouter-compatible API)

Run date: February 9, 2026.

## Reproduction Command
```bash
OPENROUTER_API_KEY=... PYTHONPATH=./pipeline-debugger uv run vf-eval pipeline-debugger \
  -p ./pipeline-debugger \
  -k OPENROUTER_API_KEY \
  -b https://openrouter.ai/api/v1 \
  -m <model> \
  -n 15 -r 1 -c 3 \
  -a '{"max_instances": 15, "max_turns": 25}' \
  -s
```

## Results
| Model | Pass rate | Avg tests_passed | Schema valid rate | Deterministic rate | Test files untouched rate | Avg tokens | Avg tool calls |
| --- | --- | --- | --- | --- | --- | --- | --- |
| openai/gpt-4.1-mini | 53.3% (8/15) | 3.93 / 5 | 66.7% | 93.3% | 100% | N/A* | 21.6 |
| deepseek/deepseek-chat | 0.0% (0/15) | 1.60 / 5 | 20.0% | 73.3% | 100% | N/A* | 5.93 |

*Token usage is not persisted in current `results.jsonl` schema produced by this setup, so exact average tokens per rollout were not recoverable post-run.

Saved run artifacts:
- `outputs/evals/pipeline-debugger--openai--gpt-4.1-mini/6161dbdb`
- `outputs/evals/pipeline-debugger--deepseek--deepseek-chat/60c3898b`

## Failure Mode Breakdown (Template)
- Test patching attempts: 0 for both models (no writes/replacements targeting `tests/` paths detected in tool-call traces).
- Environment confusion attempts:
  - `openai/gpt-4.1-mini`: 35 `pip install` attempts; 28 tool outputs mentioning missing `pytest`/`pandas`.
  - `deepseek/deepseek-chat`: 13 `pip install` attempts; 4 tool outputs mentioning missing `pytest`/`pandas`.
- Symptom-only fixes: observed multiple rollouts where `run_pipeline.py` was fixed but tests remained partially failing (`tests_pass_rate` 0.8 buckets).
- Performance vs bug count:
  - `openai/gpt-4.1-mini`: pass rate dropped from 75% (1 bug) to 50% (2 bugs) to 0% (3 bugs).
  - `deepseek/deepseek-chat`: 0% across all bug-count buckets.
- Tool-use behavior:
  - `openai/gpt-4.1-mini` averaged 21.6 tool calls/rollout with high file-inspection + command loops.
  - `deepseek/deepseek-chat` averaged 5.93 tool calls/rollout and often stopped early with narrative-only responses.

## Interesting Qualitative Finding
`openai/gpt-4.1-mini` frequently reached correct code fixes but still spent many turns trying to install dependencies inside the sandbox, while `deepseek/deepseek-chat` often stopped after planning text without executing enough validation commands; this suggests command-execution persistence was a stronger predictor of success than raw number of edits.

## Limitations
- Only two OpenRouter models were run in this pass; no Prime Intellect model was included yet.
- Token-level usage metrics were unavailable from persisted artifacts in this setup.
- Launch set is 15 instances; expanding toward 20-50 should tighten confidence by bug category/difficulty.
