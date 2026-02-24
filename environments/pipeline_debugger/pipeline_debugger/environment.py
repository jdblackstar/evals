from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path

import verifiers as vf
from datasets import Dataset

from .verifier.verify import VerificationResult, verify_submission
from .workspace_tools import WorkspaceToolMixin


TASK_PROMPT_TEMPLATE = """You are debugging a broken Python ETL pipeline.

Instance: __INSTANCE_ID__
Workspace root: __WORKSPACE_ROOT__

Task objective:
1. Make `python run_pipeline.py` exit 0.
2. Make `pytest tests/ -v` pass.
3. Ensure outputs match `expected_schema.json` and remain deterministic across repeated runs.
4. Do not modify files under `tests/`.

Use the available tools to inspect, edit, and run commands in this workspace.
When you are done, respond with a concise summary and do not call more tools.

Initial files:
__INITIAL_TREE__
"""


class PipelineDebuggerEnv(WorkspaceToolMixin, vf.StatefulToolEnv):
    def __init__(
        self,
        instances_dir: str,
        max_instances: int = -1,
        max_turns: int = 25,
    ):
        self.instances_dir = Path(instances_dir).resolve()
        self._workspace_parents: set[str] = set()
        dataset = self._build_dataset(max_instances=max_instances)

        super().__init__(
            dataset=dataset,
            eval_dataset=dataset,
            max_turns=max_turns,
            tools=[],
            env_id="pipeline-debugger",
        )

        # Hidden workspace_root arg is injected per rollout.
        self.add_tool(self.list_files, args_to_skip=["workspace_root"])
        self.add_tool(self.read_file, args_to_skip=["workspace_root"])
        self.add_tool(self.write_file, args_to_skip=["workspace_root"])
        self.add_tool(self.replace_text, args_to_skip=["workspace_root"])
        self.add_tool(self.run_command, args_to_skip=["workspace_root"])

        scoring_rubric = vf.Rubric(funcs=[self.task_passed], weights=[1.0])
        scoring_rubric.add_metric(self.tests_pass_rate)
        scoring_rubric.add_metric(self.schema_valid)
        scoring_rubric.add_metric(self.deterministic)
        scoring_rubric.add_metric(self.test_files_untouched)
        scoring_rubric.add_metric(self.run_pipeline_exit_zero)
        self.add_rubric(scoring_rubric)

    def _build_dataset(self, max_instances: int) -> Dataset:
        instance_dirs = sorted(
            path
            for path in self.instances_dir.iterdir()
            if path.is_dir() and path.name.startswith("instance_")
        )

        if max_instances > 0:
            instance_dirs = instance_dirs[:max_instances]

        rows = []
        for instance_dir in instance_dirs:
            rows.append(
                {
                    "question": TASK_PROMPT_TEMPLATE,
                    "answer": "pipeline-fixed",
                    "info": {
                        "instance_id": instance_dir.name,
                        "instance_path": str(instance_dir),
                    },
                }
            )

        if not rows:
            raise ValueError(f"No instances found in {self.instances_dir}")

        return Dataset.from_list(rows)

    async def setup_state(self, state: vf.State, **kwargs) -> vf.State:
        instance_path = Path(state["info"]["instance_path"]).resolve()
        instance_id = str(state["info"]["instance_id"])

        workspace_parent = Path(
            tempfile.mkdtemp(prefix=f"pipeline-debugger-{instance_id}-")
        ).resolve()
        workspace_root = workspace_parent / "workspace"
        shutil.copytree(instance_path, workspace_root, dirs_exist_ok=True)

        state["workspace_parent"] = str(workspace_parent)
        state["workspace_root"] = str(workspace_root)
        self._workspace_parents.add(str(workspace_parent))

        prompt = state["prompt"]
        if isinstance(prompt, list) and prompt:
            user_content = str(prompt[-1].get("content", ""))
            user_content = user_content.replace("__INSTANCE_ID__", instance_id)
            user_content = user_content.replace(
                "__WORKSPACE_ROOT__", str(workspace_root)
            )
            user_content = user_content.replace(
                "__INITIAL_TREE__",
                self._render_workspace_tree(workspace_root, max_entries=120),
            )
            prompt[-1]["content"] = user_content
            state["prompt"] = prompt

        return await super().setup_state(state, **kwargs)

    @vf.teardown
    async def teardown_workspaces(self) -> None:
        for workspace_parent in list(self._workspace_parents):
            shutil.rmtree(workspace_parent, ignore_errors=True)
        self._workspace_parents.clear()

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict,
        messages: vf.Messages,
        state: vf.State,
        **kwargs,
    ) -> dict:
        updated = dict(tool_args)
        updated["workspace_root"] = state["workspace_root"]
        return updated

    def list_files(
        self,
        path: str = ".",
        max_entries: int = 200,
        workspace_root: str = "",
    ) -> str:
        """List files/directories under `path` (workspace-relative)."""
        return self._workspace_list_files(
            path=path,
            max_entries=max_entries,
            workspace_root=workspace_root,
        )

    def read_file(
        self,
        path: str,
        start_line: int = 1,
        end_line: int = 220,
        workspace_root: str = "",
    ) -> str:
        """Read a text file with line numbers."""
        return self._workspace_read_file(
            path=path,
            start_line=start_line,
            end_line=end_line,
            workspace_root=workspace_root,
        )

    def write_file(self, path: str, content: str, workspace_root: str = "") -> str:
        """Overwrite a text file with `content`."""
        return self._workspace_write_file(
            path=path,
            content=content,
            workspace_root=workspace_root,
        )

    def replace_text(
        self,
        path: str,
        old_text: str,
        new_text: str,
        count: int = 1,
        workspace_root: str = "",
    ) -> str:
        """Replace text in a file. Returns number of replacements applied."""
        return self._workspace_replace_text(
            path=path,
            old_text=old_text,
            new_text=new_text,
            count=count,
            workspace_root=workspace_root,
        )

    def run_command(
        self,
        command: str,
        timeout_seconds: int = 30,
        workspace_root: str = "",
    ) -> str:
        """Run a shell command in the workspace and return exit code/stdout/stderr."""
        return self._workspace_run_command(
            command=command,
            timeout_seconds=timeout_seconds,
            workspace_root=workspace_root,
            output_limit=8000,
        )

    async def _verification_result(self, state: vf.State) -> VerificationResult:
        if "verification_result" not in state:
            result = await asyncio.to_thread(
                verify_submission,
                Path(state["workspace_root"]),
            )
            state["verification_result"] = result.to_dict()

        raw = dict(state["verification_result"])
        return VerificationResult(
            passed=bool(raw["passed"]),
            run_pipeline_exit_zero=bool(raw["run_pipeline_exit_zero"]),
            tests_passed=int(raw["tests_passed"]),
            tests_total=int(raw["tests_total"]),
            schema_valid=bool(raw["schema_valid"]),
            deterministic=bool(raw["deterministic"]),
            test_files_untouched=bool(raw["test_files_untouched"]),
            details=dict(raw.get("details", {})),
        )

    async def task_passed(self, state: vf.State) -> float:
        result = await self._verification_result(state)
        return 1.0 if result.passed else 0.0

    async def tests_pass_rate(self, state: vf.State) -> float:
        result = await self._verification_result(state)
        if result.tests_total == 0:
            return 0.0
        return result.tests_passed / result.tests_total

    async def schema_valid(self, state: vf.State) -> float:
        result = await self._verification_result(state)
        return 1.0 if result.schema_valid else 0.0

    async def deterministic(self, state: vf.State) -> float:
        result = await self._verification_result(state)
        return 1.0 if result.deterministic else 0.0

    async def test_files_untouched(self, state: vf.State) -> float:
        result = await self._verification_result(state)
        return 1.0 if result.test_files_untouched else 0.0

    async def run_pipeline_exit_zero(self, state: vf.State) -> float:
        result = await self._verification_result(state)
        return 1.0 if result.run_pipeline_exit_zero else 0.0


def load_environment(
    instances_dir: str | None = None,
    max_instances: int = -1,
    max_turns: int = 25,
) -> vf.Environment:
    root = Path(__file__).resolve().parent
    resolved_instances_dir = instances_dir or str(root / "instances")
    return PipelineDebuggerEnv(
        instances_dir=resolved_instances_dir,
        max_instances=max_instances,
        max_turns=max_turns,
    )
