"""Run pipeline-debugger evals against an OpenAI-compatible endpoint.

Loads OPENROUTER_API_KEY from the repo root .env via python-dotenv.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable


def _load_run_eval_main() -> Callable[..., int]:
    """Load shared eval runner after making shared package importable."""
    shared_root = Path(__file__).resolve().parents[1]
    shared_root_str = str(shared_root)
    if shared_root_str not in sys.path:
        sys.path.insert(0, shared_root_str)

    from shared.run_eval_common import run_eval_main

    return run_eval_main


def main() -> int:
    """Run pipeline-debugger eval with project defaults."""
    run_eval_main = _load_run_eval_main()
    repo_root = Path(__file__).resolve().parents[2]
    env_root = repo_root / "evals" / "pipeline-debugger"
    return run_eval_main(
        env_id="pipeline-debugger",
        env_root=env_root,
        repo_root=repo_root,
        default_model="openai/gpt-5.2",
        default_num_examples=15,
        default_rollouts=1,
        default_concurrency=3,
        default_max_instances=15,
        default_max_turns=25,
    )


if __name__ == "__main__":
    raise SystemExit(main())
