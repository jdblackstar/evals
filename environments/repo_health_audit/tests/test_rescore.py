from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from repo_health_audit.verifier.rescore import (
    extract_audit_report_from_result_row,
    rescore_results_file,
)

ROOT = Path(__file__).resolve().parents[1]
INSTANCES_ROOT = ROOT / "repo_health_audit" / "instances"


def _set_nested(payload: dict[str, Any], dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")
    cursor = payload
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def _render_perfect_report(answer_key: dict[str, Any]) -> dict[str, Any]:
    report: dict[str, Any] = {
        "instance_id": answer_key["instance_id"],
        "tier": answer_key["tier"],
        "audit": {},
        "tooling": {
            "commands_attempted": [],
            "blocked_tools": [],
            "fallback_methods": [],
        },
        "measurement_gaps": [],
        "evidence": {},
        "summary": "Audit complete.",
    }

    for path, value in answer_key.get("expected", {}).items():
        _set_nested(report, str(path), value)

    for field_path in answer_key.get("null_fields", []):
        _set_nested(report, str(field_path), None)

    for field_path in answer_key.get("required_gap_fields", []):
        report["measurement_gaps"].append(
            {"field": str(field_path), "reason": "tool unavailable in workspace"}
        )

    for tool_name in answer_key.get("required_blocked_tools", []):
        report["tooling"]["blocked_tools"].append(
            {"tool": str(tool_name), "reason": "command not found"}
        )

    for fallback in answer_key.get("required_fallback_methods", []):
        report["tooling"]["fallback_methods"].append(str(fallback))

    for key, required in answer_key.get("evidence_requirements", {}).items():
        required_list = [str(item) for item in required]
        report["evidence"][str(key)] = [f"Confirmed via {req}" for req in required_list]

    return report


def _write_tool_call(path: str, content: str) -> str:
    payload = {
        "id": "call_1",
        "function": {
            "name": "write_file",
            "arguments": json.dumps({"path": path, "content": content}),
        },
        "type": "function",
    }
    return json.dumps(payload)


def test_extract_uses_latest_audit_report_write() -> None:
    old_report = {"instance_id": "instance_001", "tier": 1}
    new_report = {"instance_id": "instance_001", "tier": 2}
    row = {
        "completion": [
            {
                "tool_calls": [
                    _write_tool_call("notes.txt", "ignored"),
                    _write_tool_call("audit_report.json", json.dumps(old_report)),
                ]
            },
            {
                "tool_calls": [
                    _write_tool_call("audit_report.json", json.dumps(new_report)),
                ]
            },
        ]
    }

    report, parse_error = extract_audit_report_from_result_row(row)
    assert parse_error is None
    assert report == new_report


def test_extract_allows_lenient_json_content() -> None:
    # The inner JSON has an unescaped newline in the summary string.
    lenient_content = '{"instance_id":"instance_001","summary":"line1\nline2"}'
    row = {
        "completion": [
            {"tool_calls": [_write_tool_call("audit_report.json", lenient_content)]}
        ]
    }

    report, parse_error = extract_audit_report_from_result_row(row)
    assert parse_error is None
    assert report is not None
    assert report["instance_id"] == "instance_001"


def test_rescore_results_file_smoke(tmp_path: Path) -> None:
    instance_id = "instance_001"
    instance_dir = INSTANCES_ROOT / instance_id
    answer_key = json.loads((instance_dir / "answer_key.json").read_text())
    report = _render_perfect_report(answer_key)

    row = {
        "example_id": 0,
        "task_passed": 0.0,
        "factual_accuracy": 0.0,
        "judgment_accuracy": 0.0,
        "tooling_accuracy": 0.0,
        "degradation_handling": 0.0,
        "evidence_grounding": 0.0,
        "repo_untouched": 1.0,
        "info": {
            "instance_id": instance_id,
            "instance_path": str(instance_dir),
        },
        "completion": [
            {
                "tool_calls": [
                    _write_tool_call("audit_report.json", json.dumps(report)),
                ]
            }
        ],
    }
    results_file = tmp_path / "results.jsonl"
    results_file.write_text(json.dumps(row) + "\n")

    summary = rescore_results_file(results_file)
    assert summary.total_examples == 1
    assert summary.old_pass_count == 0
    assert summary.new_pass_count == 1
    assert summary.changed_instances == [instance_id]
    assert summary.entries[0].parse_error is None
