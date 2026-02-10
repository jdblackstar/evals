from __future__ import annotations

from pathlib import Path

from environments.pipeline_debugger import PipelineDebuggerEnv, load_environment

ROOT = Path(__file__).resolve().parents[1]


def test_load_environment_max_instances() -> None:
    env = load_environment(
        instances_dir=str(ROOT / "instances"),
        max_instances=3,
        max_turns=12,
    )
    assert isinstance(env, PipelineDebuggerEnv)

    eval_dataset = env.get_eval_dataset()
    assert len(eval_dataset) == 3
    assert env.max_turns == 12


def test_environment_registers_expected_tools() -> None:
    env = load_environment(
        instances_dir=str(ROOT / "instances"),
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
