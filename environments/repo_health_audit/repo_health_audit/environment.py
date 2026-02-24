from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from pathlib import Path

import verifiers as vf
from datasets import Dataset

from .verifier.verify import VerificationResult, hash_repo_files, verify_submission
from .workspace_tools import WorkspaceToolMixin

TASK_PROMPT_TEMPLATE = """You are auditing repository health for an engineering snapshot.

Instance: __INSTANCE_ID__
Tier: __TIER__
Workspace root: __WORKSPACE_ROOT__
Target repo: __WORKSPACE_ROOT__/target_repo
Audit spec: __WORKSPACE_ROOT__/audit_spec.json

Task objective:
1. Inspect the repository and relevant artifacts.
2. Produce `audit_report.json` at the workspace root, following `audit_spec.json`.
3. If a tool is unavailable or results are ambiguous, use `null` or `"inconclusive"` and explain the gap in `measurement_gaps`.
4. Do not modify files under `target_repo/`.

Instance notes:
__TASK_NOTES__

Initial files:
__INITIAL_TREE__
"""


class RepoHealthAuditEnv(WorkspaceToolMixin, vf.StatefulToolEnv):
    def __init__(
        self,
        instances_dir: str,
        max_instances: int = -1,
        max_turns: int = 30,
    ):
        self.instances_dir = Path(instances_dir).resolve()
        self._workspace_parents: set[str] = set()
        dataset = self._build_dataset(max_instances=max_instances)

        super().__init__(
            dataset=dataset,
            eval_dataset=dataset,
            max_turns=max_turns,
            tools=[],
            env_id="repo-health-audit",
        )

        self.add_tool(self.list_files, args_to_skip=["workspace_root"])
        self.add_tool(self.read_file, args_to_skip=["workspace_root"])
        self.add_tool(self.write_file, args_to_skip=["workspace_root"])
        self.add_tool(self.replace_text, args_to_skip=["workspace_root"])
        self.add_tool(self.run_command, args_to_skip=["workspace_root"])

        scoring_rubric = vf.Rubric(funcs=[self.task_passed], weights=[1.0])
        scoring_rubric.add_metric(self.factual_accuracy)
        scoring_rubric.add_metric(self.judgment_accuracy)
        scoring_rubric.add_metric(self.tooling_accuracy)
        scoring_rubric.add_metric(self.degradation_handling)
        scoring_rubric.add_metric(self.evidence_grounding)
        scoring_rubric.add_metric(self.repo_untouched)
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
            metadata_path = instance_dir / "metadata.json"
            metadata = (
                json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
            )
            rows.append(
                {
                    "question": TASK_PROMPT_TEMPLATE,
                    "answer": "audit-complete",
                    "info": {
                        "instance_id": metadata.get("instance_id", instance_dir.name),
                        "tier": int(metadata.get("tier", 1)),
                        "task_notes": list(metadata.get("task_notes", [])),
                        "instance_path": str(instance_dir),
                    },
                }
            )

        if not rows:
            raise ValueError(f"No instances found in {self.instances_dir}")

        return Dataset.from_list(rows)

    async def setup_state(self, state: vf.State, **kwargs) -> vf.State:
        instance_path = Path(state["info"]["instance_path"]).resolve()
        workspace_seed = instance_path / "workspace"
        if not workspace_seed.exists():
            raise ValueError(f"Missing workspace seed directory: {workspace_seed}")

        instance_id = str(state["info"]["instance_id"])
        tier = int(state["info"]["tier"])
        task_notes = list(state["info"]["task_notes"])

        workspace_parent = Path(
            tempfile.mkdtemp(prefix=f"repo-health-audit-{instance_id}-")
        ).resolve()
        workspace_root = workspace_parent / "workspace"
        shutil.copytree(workspace_seed, workspace_root, dirs_exist_ok=True)

        target_repo = workspace_root / "target_repo"
        if not target_repo.exists():
            raise ValueError(f"Missing target_repo in {workspace_seed}")

        state["workspace_parent"] = str(workspace_parent)
        state["workspace_root"] = str(workspace_root)
        state["answer_key_path"] = str(instance_path / "answer_key.json")
        state["initial_repo_hashes"] = hash_repo_files(target_repo)
        self._workspace_parents.add(str(workspace_parent))

        prompt = state["prompt"]
        if isinstance(prompt, list) and prompt:
            user_content = str(prompt[-1].get("content", ""))
            user_content = user_content.replace("__INSTANCE_ID__", instance_id)
            user_content = user_content.replace("__TIER__", str(tier))
            user_content = user_content.replace(
                "__WORKSPACE_ROOT__", str(workspace_root)
            )
            user_content = user_content.replace(
                "__TASK_NOTES__",
                "\n".join(f"- {note}" for note in task_notes)
                if task_notes
                else "- No extra notes.",
            )
            user_content = user_content.replace(
                "__INITIAL_TREE__",
                self._render_workspace_tree(workspace_root, max_entries=180),
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
        max_entries: int = 250,
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
        end_line: int = 260,
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
            output_limit=10000,
        )

    async def _verification_result(self, state: vf.State) -> VerificationResult:
        if "verification_result" not in state:
            result = await asyncio.to_thread(
                verify_submission,
                Path(state["workspace_root"]),
                Path(state["answer_key_path"]),
                dict(state["initial_repo_hashes"]),
            )
            state["verification_result"] = result.to_dict()

        raw = dict(state["verification_result"])
        return VerificationResult(
            passed=bool(raw["passed"]),
            factual_accuracy=float(raw["factual_accuracy"]),
            judgment_accuracy=float(raw["judgment_accuracy"]),
            tooling_accuracy=float(raw["tooling_accuracy"]),
            degradation_handling=float(raw["degradation_handling"]),
            evidence_grounding=float(raw["evidence_grounding"]),
            repo_untouched=bool(raw["repo_untouched"]),
            details=dict(raw.get("details", {})),
        )

    async def task_passed(self, state: vf.State) -> float:
        result = await self._verification_result(state)
        return 1.0 if result.passed else 0.0

    async def factual_accuracy(self, state: vf.State) -> float:
        result = await self._verification_result(state)
        return result.factual_accuracy

    async def judgment_accuracy(self, state: vf.State) -> float:
        result = await self._verification_result(state)
        return result.judgment_accuracy

    async def tooling_accuracy(self, state: vf.State) -> float:
        result = await self._verification_result(state)
        return result.tooling_accuracy

    async def degradation_handling(self, state: vf.State) -> float:
        result = await self._verification_result(state)
        return result.degradation_handling

    async def evidence_grounding(self, state: vf.State) -> float:
        result = await self._verification_result(state)
        return result.evidence_grounding

    async def repo_untouched(self, state: vf.State) -> float:
        result = await self._verification_result(state)
        return 1.0 if result.repo_untouched else 0.0


def load_environment(
    instances_dir: str | None = None,
    max_instances: int = -1,
    max_turns: int = 30,
) -> vf.Environment:
    root = Path(__file__).resolve().parent
    resolved_instances_dir = instances_dir or str(root / "instances")
    return RepoHealthAuditEnv(
        instances_dir=resolved_instances_dir,
        max_instances=max_instances,
        max_turns=max_turns,
    )
