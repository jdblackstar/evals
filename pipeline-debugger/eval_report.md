# Eval Report: Pipeline Debugger

## Models Tested
- `openai/gpt-5.2` (OpenRouter-compatible API)
- `deepseek/deepseek-chat` (OpenRouter-compatible API)

Run date: February 9, 2026.

## Reproduction Command
```bash
uv run python pipeline-debugger/run_eval.py --model <model> -n 15 -r 1 -c 3 --max-instances 15 --max-turns 25
```
`run_eval.py` loads `OPENROUTER_API_KEY` from repo-root `.env` via `python-dotenv` and defaults to `openai/gpt-5.2`.

## Results
| Model | Pass rate | Avg tests_passed | Schema valid rate | Deterministic rate | Test files untouched rate | Avg tokens | Avg tool calls |
| --- | --- | --- | --- | --- | --- | --- | --- |
| openai/gpt-5.2 | 93.3% (14/15) | 4.67 / 5 | 93.3% | 93.3% | 100% | N/A* | 17.47 |
| deepseek/deepseek-chat | 0.0% (0/15) | 1.60 / 5 | 20.0% | 73.3% | 100% | N/A* | 5.93 |

*Token usage is not persisted in current `results.jsonl` schema produced by this setup, so exact average tokens per rollout were not recoverable post-run.

Saved run artifacts:
- `pipeline-debugger/outputs/evals/pipeline-debugger--openai--gpt-5.2/d2945591`
- `outputs/evals/pipeline-debugger--deepseek--deepseek-chat/60c3898b`

## Failure Mode Breakdown (Template)
- Test patching attempts: 0 for both models (no writes/replacements targeting `tests/` paths detected in tool-call traces).
- Environment confusion attempts:
  - `openai/gpt-5.2`: 0 `pip install` attempts; 0 tool outputs mentioning missing `pytest`/`pandas`.
  - `deepseek/deepseek-chat`: 13 `pip install` attempts; 4 tool outputs mentioning missing `pytest`/`pandas`.
- Symptom-only fixes: rare for `openai/gpt-5.2`; the single failing rollout stopped early after inspection without attempting edits.
- Performance vs bug count:
  - `openai/gpt-5.2`: 87.5% (1-bug instances), 100% (2-bug instances), 100% (3-bug instances).
  - `deepseek/deepseek-chat`: 0% across all bug-count buckets.
- Tool-use behavior:
  - `openai/gpt-5.2` averaged 17.47 tool calls/rollout with consistent inspect-edit-run loops.
  - `deepseek/deepseek-chat` averaged 5.93 tool calls/rollout and often stopped early with narrative-only responses.

## Interesting Qualitative Finding
`openai/gpt-5.2` mostly executed a disciplined read→edit→validate loop and avoided environment detours entirely, but the lone miss shows a remaining brittleness mode: a rollout can terminate after inspection with no edits, yielding a full failure on that instance.

## Limitations
- Only two OpenRouter models were run in this pass; no Prime Intellect model was included yet.
- Token-level usage metrics were unavailable from persisted artifacts in this setup.
- Launch set is 15 instances; expanding toward 20-50 should tighten confidence by bug category/difficulty.
