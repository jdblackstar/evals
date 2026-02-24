from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from repo_health_audit import RepoHealthAuditEnv, load_environment

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "repo_health_audit"


def test_load_environment_max_instances() -> None:
    env = load_environment(
        instances_dir=str(PACKAGE_ROOT / "instances"),
        max_instances=2,
        max_turns=14,
    )
    assert isinstance(env, RepoHealthAuditEnv)

    eval_dataset = env.get_eval_dataset()
    assert len(eval_dataset) == 2
    assert env.max_turns == 14


def test_environment_registers_expected_tools() -> None:
    env = load_environment(
        instances_dir=str(PACKAGE_ROOT / "instances"),
        max_instances=1,
    )
    tool_names = sorted(tool.__name__ for tool in env.tools)
    assert tool_names == [
        "list_files",
        "read_file",
        "replace_text",
        "run_command",
        "write_file",
    ]


def test_entrypoint_importable_from_env_root() -> None:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-c", "import repo_health_audit"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"entrypoint import failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
