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


def _build_parser(
    *,
    description: str,
    default_model: str,
    default_num_examples: int,
    default_rollouts: int,
    default_concurrency: int,
    default_max_instances: int,
    default_max_turns: int,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--model",
        default=default_model,
        help=f"Model name (default: {default_model}).",
    )
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("-n", "--num-examples", type=int, default=default_num_examples)
    parser.add_argument("-r", "--rollouts", type=int, default=default_rollouts)
    parser.add_argument("-c", "--concurrency", type=int, default=default_concurrency)
    parser.add_argument("--max-instances", type=int, default=default_max_instances)
    parser.add_argument("--max-turns", type=int, default=default_max_turns)
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


def run_eval_main(
    *,
    env_id: str,
    env_root: Path,
    repo_root: Path,
    default_model: str,
    default_num_examples: int,
    default_rollouts: int,
    default_concurrency: int,
    default_max_instances: int,
    default_max_turns: int,
) -> int:
    parser = _build_parser(
        description=f"Run {env_id} evals.",
        default_model=default_model,
        default_num_examples=default_num_examples,
        default_rollouts=default_rollouts,
        default_concurrency=default_concurrency,
        default_max_instances=default_max_instances,
        default_max_turns=default_max_turns,
    )
    args = parser.parse_args()

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
        env_id,
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
