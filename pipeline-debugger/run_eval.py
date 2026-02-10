"""Run pipeline-debugger evals against an OpenAI-compatible endpoint.

Loads OPENROUTER_API_KEY from the repo root .env via python-dotenv.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

API_KEY_ENV = "OPENROUTER_API_KEY"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run pipeline-debugger evals.")
    parser.add_argument(
        "--model",
        default="openai/gpt-5.2",
        help="Model name (default: openai/gpt-5.2).",
    )
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("-n", "--num-examples", type=int, default=15)
    parser.add_argument("-r", "--rollouts", type=int, default=1)
    parser.add_argument("-c", "--concurrency", type=int, default=3)
    parser.add_argument("--max-instances", type=int, default=15)
    parser.add_argument("--max-turns", type=int, default=25)
    parser.add_argument(
        "-s",
        "--save",
        action="store_true",
        default=True,
        help="Pass -s to vf-eval (default: true).",
    )
    parser.add_argument("--no-save", dest="save", action="store_false")
    parser.add_argument(
        "--print-command", action="store_true", help="Print vf-eval command and exit."
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    env_root = repo_root / "pipeline-debugger"
    dotenv_path = repo_root / ".env"
    load_dotenv(dotenv_path=dotenv_path, override=False)

    api_key = os.getenv(API_KEY_ENV)
    if not api_key:
        print(
            f"Missing {API_KEY_ENV}. Add it to {dotenv_path} or export it in your shell.",
            file=sys.stderr,
        )
        return 1

    vf_args = [
        sys.executable,
        "-m",
        "verifiers.scripts.eval",
        "pipeline-debugger",
        "-p",
        ".",
        "-k",
        API_KEY_ENV,
        "-b",
        args.base_url,
        "-m",
        args.model,
        "-n",
        str(args.num_examples),
        "-r",
        str(args.rollouts),
        "-c",
        str(args.concurrency),
        "-a",
        json.dumps({"max_instances": args.max_instances, "max_turns": args.max_turns}),
    ]
    if args.save:
        vf_args.append("-s")

    if args.print_command:
        print(" ".join(shlex.quote(part) for part in vf_args))
        return 0

    completed = subprocess.run(vf_args, cwd=env_root, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
