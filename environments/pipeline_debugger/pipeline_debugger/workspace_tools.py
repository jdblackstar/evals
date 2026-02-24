from __future__ import annotations

import json
import subprocess
from pathlib import Path


class WorkspaceToolMixin:
    """Reusable workspace helpers for eval tool environments."""

    def _resolve_workspace_path(self, workspace_root: str, rel_path: str) -> Path:
        root = Path(workspace_root).resolve()
        candidate = (root / rel_path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError("path escapes workspace root") from exc
        return candidate

    def _render_workspace_tree(
        self,
        workspace_root: Path,
        max_entries: int = 120,
    ) -> str:
        entries: list[str] = []
        for path in sorted(workspace_root.rglob("*")):
            if len(entries) >= max_entries:
                entries.append("... (truncated)")
                break
            rel = path.relative_to(workspace_root)
            suffix = "/" if path.is_dir() else ""
            entries.append(f"- {rel}{suffix}")
        return "\n".join(entries)

    def _workspace_list_files(
        self,
        path: str,
        max_entries: int,
        workspace_root: str,
    ) -> str:
        target = self._resolve_workspace_path(workspace_root, path)
        if not target.exists():
            return f"error: path does not exist: {path}"

        root = Path(workspace_root).resolve()
        if target.is_file():
            return str(target.relative_to(root))

        entries: list[str] = []
        for child in sorted(target.rglob("*")):
            if len(entries) >= max_entries:
                entries.append("... (truncated)")
                break
            rel = child.relative_to(root)
            suffix = "/" if child.is_dir() else ""
            entries.append(f"{rel}{suffix}")
        return "\n".join(entries) if entries else "(empty directory)"

    def _workspace_read_file(
        self,
        path: str,
        start_line: int,
        end_line: int,
        workspace_root: str,
    ) -> str:
        target = self._resolve_workspace_path(workspace_root, path)
        if not target.exists() or not target.is_file():
            return f"error: file not found: {path}"

        lines = target.read_text().splitlines()
        start = max(start_line, 1)
        end = max(end_line, start)
        selected = lines[start - 1 : end]
        rendered = [f"{idx + start:04d}: {line}" for idx, line in enumerate(selected)]
        if not rendered:
            return "(no content)"
        return "\n".join(rendered)

    def _workspace_write_file(
        self, path: str, content: str, workspace_root: str
    ) -> str:
        target = self._resolve_workspace_path(workspace_root, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return f"wrote {len(content)} chars to {path}"

    def _workspace_replace_text(
        self,
        path: str,
        old_text: str,
        new_text: str,
        count: int,
        workspace_root: str,
    ) -> str:
        target = self._resolve_workspace_path(workspace_root, path)
        if not target.exists() or not target.is_file():
            return f"error: file not found: {path}"

        before = target.read_text()
        if old_text not in before:
            return "error: old_text not found"

        after = before.replace(old_text, new_text, count)
        target.write_text(after)
        replacements = before.count(old_text) - after.count(old_text)
        return f"replaced {replacements} occurrence(s) in {path}"

    def _workspace_run_command(
        self,
        command: str,
        timeout_seconds: int,
        workspace_root: str,
        output_limit: int = 8000,
    ) -> str:
        timeout_seconds = max(1, min(timeout_seconds, 120))
        try:
            result = subprocess.run(
                command,
                cwd=workspace_root,
                shell=True,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            payload = {
                "exit_code": -1,
                "stdout": exc.stdout or "",
                "stderr": f"timeout after {timeout_seconds}s",
            }
            return json.dumps(payload, indent=2)

        payload = {
            "exit_code": result.returncode,
            "stdout": result.stdout[-output_limit:],
            "stderr": result.stderr[-output_limit:],
        }
        return json.dumps(payload, indent=2)
