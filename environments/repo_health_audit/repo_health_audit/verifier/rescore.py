from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .verify import hash_repo_files, verify_submission


@dataclass
class RescoreEntry:
    instance_id: str
    example_id: int | None
    old_passed: bool
    new_passed: bool
    old_metrics: dict[str, float | bool]
    new_metrics: dict[str, float | bool]
    parse_error: str | None = None
    error: str | None = None
    verification_details: dict[str, Any] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return self.old_passed != self.new_passed

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["changed"] = self.changed
        return payload


@dataclass
class RescoreSummary:
    results_jsonl: str
    total_examples: int
    old_pass_count: int
    new_pass_count: int
    changed_instances: list[str]
    entries: list[RescoreEntry]

    def to_dict(self, include_entries: bool = False) -> dict[str, Any]:
        payload = {
            "results_jsonl": self.results_jsonl,
            "total_examples": self.total_examples,
            "old_pass_count": self.old_pass_count,
            "new_pass_count": self.new_pass_count,
            "changed_instances": self.changed_instances,
        }
        if include_entries:
            payload["entries"] = [entry.to_dict() for entry in self.entries]
        return payload


_METRIC_KEYS = (
    "task_passed",
    "factual_accuracy",
    "judgment_accuracy",
    "tooling_accuracy",
    "degradation_handling",
    "evidence_grounding",
    "repo_untouched",
)


def _bool_metric(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    try:
        return float(value) >= 1.0
    except (TypeError, ValueError):
        return False


def _collect_old_metrics(row: dict[str, Any]) -> dict[str, float | bool]:
    metrics: dict[str, float | bool] = {}
    for key in _METRIC_KEYS:
        value = row.get(key)
        if key == "repo_untouched":
            metrics[key] = _bool_metric(value)
            continue
        try:
            metrics[key] = float(value)
        except (TypeError, ValueError):
            metrics[key] = 0.0
    return metrics


def _result_metrics_from_verifier(result: Any) -> dict[str, float | bool]:
    return {
        "task_passed": 1.0 if result.passed else 0.0,
        "factual_accuracy": float(result.factual_accuracy),
        "judgment_accuracy": float(result.judgment_accuracy),
        "tooling_accuracy": float(result.tooling_accuracy),
        "degradation_handling": float(result.degradation_handling),
        "evidence_grounding": float(result.evidence_grounding),
        "repo_untouched": bool(result.repo_untouched),
    }


def _parse_json_maybe_lenient(payload: str) -> dict[str, Any]:
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        parsed = json.loads(payload, strict=False)
    if not isinstance(parsed, dict):
        raise ValueError("audit_report.json content must decode to a JSON object")
    return parsed


def _parse_tool_call(raw_tool_call: Any) -> dict[str, Any] | None:
    if isinstance(raw_tool_call, dict):
        return raw_tool_call
    if not isinstance(raw_tool_call, str):
        return None
    try:
        parsed = json.loads(raw_tool_call)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def extract_audit_report_from_result_row(
    row: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    """Extract and decode the final audit_report.json written in a result row."""
    completion = row.get("completion")
    if not isinstance(completion, list):
        return None, "completion is not a list"

    latest_content: str | None = None
    for message in completion:
        if not isinstance(message, dict):
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for raw_tool_call in tool_calls:
            tool_call = _parse_tool_call(raw_tool_call)
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if not isinstance(function, dict):
                continue
            if function.get("name") != "write_file":
                continue
            raw_args = function.get("arguments")
            if not isinstance(raw_args, str):
                continue
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                continue
            if not isinstance(args, dict):
                continue
            if args.get("path") != "audit_report.json":
                continue
            content = args.get("content")
            if isinstance(content, str):
                latest_content = content

    if latest_content is None:
        return None, "missing write_file(audit_report.json) tool call"

    try:
        report = _parse_json_maybe_lenient(latest_content)
    except (json.JSONDecodeError, ValueError) as exc:
        return None, f"invalid audit_report.json in trace: {exc}"
    return report, None


def _resolve_instance_dir(
    row: dict[str, Any],
    instances_root: Path,
) -> tuple[str, Path]:
    info = row.get("info")
    if not isinstance(info, dict):
        raise ValueError("result row missing info payload")

    instance_id_raw = info.get("instance_id")
    if not isinstance(instance_id_raw, str) or not instance_id_raw:
        raise ValueError("result row missing info.instance_id")
    instance_id = instance_id_raw

    instance_path_raw = info.get("instance_path")
    if isinstance(instance_path_raw, str) and instance_path_raw:
        return instance_id, Path(instance_path_raw)
    return instance_id, instances_root / instance_id


def rescore_result_row(row: dict[str, Any], instances_root: Path) -> RescoreEntry:
    """Re-run verification for a single results.jsonl row."""
    old_metrics = _collect_old_metrics(row)
    old_passed = _bool_metric(old_metrics.get("task_passed"))
    example_id_raw = row.get("example_id")
    example_id = int(example_id_raw) if isinstance(example_id_raw, int) else None

    try:
        instance_id, instance_dir = _resolve_instance_dir(row, instances_root)
        workspace_seed = instance_dir / "workspace"
        answer_key_path = instance_dir / "answer_key.json"
        if not workspace_seed.exists():
            raise FileNotFoundError(f"workspace seed not found: {workspace_seed}")
        if not answer_key_path.exists():
            raise FileNotFoundError(f"answer_key.json not found: {answer_key_path}")
    except Exception as exc:
        return RescoreEntry(
            instance_id=str(row.get("info", {}).get("instance_id", "unknown")),
            example_id=example_id,
            old_passed=old_passed,
            new_passed=False,
            old_metrics=old_metrics,
            new_metrics={key: 0.0 for key in _METRIC_KEYS[:-1]} | {"repo_untouched": False},
            error=str(exc),
        )

    report, parse_error = extract_audit_report_from_result_row(row)

    with tempfile.TemporaryDirectory(prefix=f"rescore-{instance_id}-") as tempdir:
        workspace_root = Path(tempdir) / "workspace"
        shutil.copytree(workspace_seed, workspace_root)

        if report is not None:
            (workspace_root / "audit_report.json").write_text(
                json.dumps(report, indent=2, sort_keys=True)
            )

        initial_hashes = hash_repo_files(workspace_root / "target_repo")
        verification_result = verify_submission(
            workspace_root=workspace_root,
            answer_key_path=answer_key_path,
            initial_repo_hashes=initial_hashes,
        )

    new_metrics = _result_metrics_from_verifier(verification_result)
    return RescoreEntry(
        instance_id=instance_id,
        example_id=example_id,
        old_passed=old_passed,
        new_passed=bool(verification_result.passed),
        old_metrics=old_metrics,
        new_metrics=new_metrics,
        parse_error=parse_error,
        verification_details=dict(verification_result.details),
    )


def load_results_rows(results_jsonl_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in results_jsonl_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if not isinstance(payload, dict):
            raise ValueError("results.jsonl contains a non-object row")
        rows.append(payload)
    return rows


def rescore_results_file(
    results_jsonl_path: Path,
    instances_root: Path | None = None,
) -> RescoreSummary:
    """Re-score an existing results.jsonl file with the current verifier."""
    resolved_results = results_jsonl_path.resolve()
    resolved_instances_root = (
        instances_root.resolve()
        if instances_root is not None
        else Path(__file__).resolve().parents[1] / "instances"
    )

    rows = load_results_rows(resolved_results)
    entries = [
        rescore_result_row(row=row, instances_root=resolved_instances_root)
        for row in rows
    ]

    old_pass_count = sum(1 for entry in entries if entry.old_passed)
    new_pass_count = sum(1 for entry in entries if entry.new_passed)
    changed_instances = sorted(
        entry.instance_id for entry in entries if entry.changed
    )

    return RescoreSummary(
        results_jsonl=str(resolved_results),
        total_examples=len(entries),
        old_pass_count=old_pass_count,
        new_pass_count=new_pass_count,
        changed_instances=changed_instances,
        entries=entries,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-score a repo-health-audit results.jsonl with the current verifier."
    )
    parser.add_argument(
        "results_jsonl",
        type=Path,
        help="Path to an existing results.jsonl file.",
    )
    parser.add_argument(
        "--instances-root",
        type=Path,
        default=None,
        help="Optional override for instances root (default: package instances directory).",
    )
    parser.add_argument(
        "--include-entries",
        action="store_true",
        help="Include per-instance rescoring payloads in output JSON.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional output path for rescoring JSON.",
    )
    args = parser.parse_args()

    summary = rescore_results_file(
        results_jsonl_path=args.results_jsonl,
        instances_root=args.instances_root,
    )
    payload = summary.to_dict(include_entries=args.include_entries)
    rendered = json.dumps(payload, indent=2, sort_keys=True)

    if args.json_out is not None:
        args.json_out.write_text(rendered)
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
