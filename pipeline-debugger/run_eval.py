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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run pipeline-debugger evals.")
    parser.add_argument("--model", required=True, help="Model name (for example: openai/gpt-4.1-mini).")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--env-file", default=".env", help="Path to .env file (relative to repo root if not absolute).")
    parser.add_argument("-n", "--num-examples", type=int, default=15)
    parser.add_argument("-r", "--rollouts", type=int, default=1)
    parser.add_argument("-c", "--concurrency", type=int, default=3)
    parser.add_argument("--max-instances", type=int, default=15)
    parser.add_argument("--max-turns", type=int, default=25)
    parser.add_argument("-s", "--save", action="store_true", default=True, help="Pass -s to vf-eval (default: true).")
    parser.add_argument("--no-save", dest="save", action="store_false")
    parser.add_argument("--print-command", action="store_true", help="Print vf-eval command and exit.")
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Additional args forwarded to vf-eval. Use '--' before these args.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    dotenv_path = Path(args.env_file)
    if not dotenv_path.is_absolute():
        dotenv_path = repo_root / dotenv_path
    load_dotenv(dotenv_path=dotenv_path, override=False)

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        print(
            f"Missing {args.api_key_env}. Add it to {dotenv_path} or export it in your shell.",
            file=sys.stderr,
        )
        return 1

    vf_args = [
        "vf-eval",
        "pipeline-debugger",
        "-p",
        "./pipeline-debugger",
        "-k",
        args.api_key_env,
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
    if args.extra_args:
        vf_args.extend(args.extra_args)

    if args.print_command:
        print(" ".join(shlex.quote(part) for part in vf_args))
        return 0

    completed = subprocess.run(vf_args, cwd=repo_root, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
