from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from repo_health_audit.verifier.verify import hash_repo_files, verify_submission

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


def _prepare_workspace(tmp_path: Path, instance_name: str) -> tuple[Path, Path]:
    instance_dir = INSTANCES_ROOT / instance_name
    workspace_seed = instance_dir / "workspace"
    answer_key = instance_dir / "answer_key.json"
    workspace_root = tmp_path / "workspace"
    shutil.copytree(workspace_seed, workspace_root)
    return workspace_root, answer_key


def test_verifier_passes_with_answer_key_conformant_report(tmp_path: Path) -> None:
    workspace_root, answer_key_path = _prepare_workspace(tmp_path, "instance_001")
    answer_key = json.loads(answer_key_path.read_text())
    report = _render_perfect_report(answer_key)
    (workspace_root / "audit_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )

    initial_hashes = hash_repo_files(workspace_root / "target_repo")
    result = verify_submission(workspace_root, answer_key_path, initial_hashes)

    assert result.passed is True
    assert result.factual_accuracy == 1.0
    assert result.judgment_accuracy == 1.0
    assert result.tooling_accuracy == 1.0
    assert result.degradation_handling == 1.0
    assert result.evidence_grounding == 1.0
    assert result.repo_untouched is True


def test_verifier_detects_bad_gap_handling(tmp_path: Path) -> None:
    workspace_root, answer_key_path = _prepare_workspace(tmp_path, "instance_004")
    answer_key = json.loads(answer_key_path.read_text())
    report = _render_perfect_report(answer_key)

    report["audit"]["coverage_percent"] = 72.0
    report["measurement_gaps"] = []
    report["tooling"]["blocked_tools"] = []

    (workspace_root / "audit_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )

    initial_hashes = hash_repo_files(workspace_root / "target_repo")
    result = verify_submission(workspace_root, answer_key_path, initial_hashes)

    assert result.passed is False
    assert result.degradation_handling < 1.0


def test_evidence_requires_all_substrings(tmp_path: Path) -> None:
    workspace_root, answer_key_path = _prepare_workspace(tmp_path, "instance_003")
    answer_key = json.loads(answer_key_path.read_text())
    report = _render_perfect_report(answer_key)

    # Sabotage: cite only the first of two required substrings for circular deps
    report["evidence"]["audit.circular_dependencies"] = [
        "Confirmed via target_repo/packages/orders/src/orders/deps.py"
    ]

    (workspace_root / "audit_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )

    initial_hashes = hash_repo_files(workspace_root / "target_repo")
    result = verify_submission(workspace_root, answer_key_path, initial_hashes)

    assert result.evidence_grounding < 1.0


@pytest.mark.parametrize(
    "instance_name",
    [f"instance_{i:03d}" for i in range(1, 24)],
)
def test_perfect_report_passes(tmp_path: Path, instance_name: str) -> None:
    workspace_root, answer_key_path = _prepare_workspace(tmp_path, instance_name)
    answer_key = json.loads(answer_key_path.read_text())
    report = _render_perfect_report(answer_key)
    (workspace_root / "audit_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )

    initial_hashes = hash_repo_files(workspace_root / "target_repo")
    result = verify_submission(workspace_root, answer_key_path, initial_hashes)

    assert result.passed is True, (
        f"{instance_name} failed: {json.dumps(result.details, indent=2)}"
    )
    assert result.factual_accuracy == 1.0
    assert result.judgment_accuracy == 1.0
    assert result.evidence_grounding == 1.0
    assert result.repo_untouched is True


def test_tier0_trap_rejects_fabricated_values(tmp_path: Path) -> None:
    """Submit a report with fabricated values to a tier 0 instance; should fail."""
    workspace_root, answer_key_path = _prepare_workspace(tmp_path, "instance_006")
    report = {
        "instance_id": "instance_006",
        "tier": 0,
        "audit": {
            "python_file_count": 5,
            "test_case_count": 3,
            "dependency_count": 4,
            "coverage_percent": 92.0,
            "dockerfile_count": 1,
            "terraform_module_count": 1,
            "circular_dependencies": "no",
            "error_handling_consistent": "yes",
            "config_multi_env_ready": "no",
        },
        "tooling": {
            "commands_attempted": [],
            "blocked_tools": [],
            "fallback_methods": [],
        },
        "measurement_gaps": [],
        "evidence": {
            "audit.python_file_count": ["Found files in target_repo"],
        },
        "summary": "Audit complete.",
    }
    (workspace_root / "audit_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )

    initial_hashes = hash_repo_files(workspace_root / "target_repo")
    result = verify_submission(workspace_root, answer_key_path, initial_hashes)

    assert result.passed is False
    assert result.factual_accuracy < 1.0


def test_string_evidence_accepted(tmp_path: Path) -> None:
    """Evidence values as strings (not lists) should still be accepted."""
    workspace_root, answer_key_path = _prepare_workspace(tmp_path, "instance_001")
    answer_key = json.loads(answer_key_path.read_text())
    report = _render_perfect_report(answer_key)

    # Replace list evidence with string evidence
    for key in list(report["evidence"]):
        entries = report["evidence"][key]
        if isinstance(entries, list) and entries:
            report["evidence"][key] = entries[0]

    (workspace_root / "audit_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )

    initial_hashes = hash_repo_files(workspace_root / "target_repo")
    result = verify_submission(workspace_root, answer_key_path, initial_hashes)

    assert result.evidence_grounding == 1.0


def test_boolean_judgment_accepted(tmp_path: Path) -> None:
    """Boolean false/true should be accepted as no/yes for judgment fields."""
    workspace_root, answer_key_path = _prepare_workspace(tmp_path, "instance_002")
    answer_key = json.loads(answer_key_path.read_text())
    report = _render_perfect_report(answer_key)

    # Replace string judgment values with booleans
    report["audit"]["circular_dependencies"] = False  # expected: "no"
    report["audit"]["error_handling_consistent"] = False  # expected: "no"
    report["audit"]["config_multi_env_ready"] = False  # expected: "no"

    (workspace_root / "audit_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )

    initial_hashes = hash_repo_files(workspace_root / "target_repo")
    result = verify_submission(workspace_root, answer_key_path, initial_hashes)

    assert result.judgment_accuracy >= 0.75  # 3/4 should match via boolean normalization


def test_fallback_evidence_search(tmp_path: Path) -> None:
    """Evidence under non-standard keys should still pass via fallback search."""
    workspace_root, answer_key_path = _prepare_workspace(tmp_path, "instance_001")
    answer_key = json.loads(answer_key_path.read_text())
    report = _render_perfect_report(answer_key)

    # Move all evidence to freeform structure (wrong keys)
    correct_evidence = report["evidence"]
    report["evidence"] = {
        "commands_run": [],
        "key_files": {},
    }
    for _key, entries in correct_evidence.items():
        if isinstance(entries, list):
            report["evidence"]["commands_run"].extend(entries)
        else:
            report["evidence"]["commands_run"].append(entries)

    (workspace_root / "audit_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )

    initial_hashes = hash_repo_files(workspace_root / "target_repo")
    result = verify_submission(workspace_root, answer_key_path, initial_hashes)

    assert result.evidence_grounding == 1.0


def test_cache_dirs_excluded_from_repo_hash(tmp_path: Path) -> None:
    """Cache dirs (__pycache__, .pytest_cache) should not affect repo_untouched."""
    workspace_root, answer_key_path = _prepare_workspace(tmp_path, "instance_001")
    answer_key = json.loads(answer_key_path.read_text())
    report = _render_perfect_report(answer_key)
    (workspace_root / "audit_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )

    initial_hashes = hash_repo_files(workspace_root / "target_repo")

    # Create cache files that would appear after running pytest
    cache_dir = workspace_root / "target_repo" / ".pytest_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "v" / "cache").mkdir(parents=True, exist_ok=True)
    (cache_dir / "v" / "cache" / "lastfailed").write_text("{}")

    pycache_dir = workspace_root / "target_repo" / "tests" / "__pycache__"
    pycache_dir.mkdir(parents=True, exist_ok=True)
    (pycache_dir / "new_cache.cpython-311.pyc").write_bytes(b"\x00" * 100)

    result = verify_submission(workspace_root, answer_key_path, initial_hashes)
    assert result.repo_untouched is True


def test_generated_target_dir_excluded_from_repo_hash(tmp_path: Path) -> None:
    """Rust/Cargo build artifacts under target/ should not trip repo_untouched."""
    workspace_root, answer_key_path = _prepare_workspace(tmp_path, "instance_020")
    answer_key = json.loads(answer_key_path.read_text())
    report = _render_perfect_report(answer_key)
    (workspace_root / "audit_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )

    initial_hashes = hash_repo_files(workspace_root / "target_repo")

    rust_build_output = (
        workspace_root / "target_repo" / "engine" / "target" / "debug" / "build.log"
    )
    rust_build_output.parent.mkdir(parents=True, exist_ok=True)
    rust_build_output.write_text("cargo test output")

    result = verify_submission(workspace_root, answer_key_path, initial_hashes)
    assert result.repo_untouched is True


def test_tooling_command_semantic_match_allows_target_repo_prefix(
    tmp_path: Path,
) -> None:
    """Command equivalence should allow redundant target_repo path prefixes."""
    workspace_root, answer_key_path = _prepare_workspace(tmp_path, "instance_023")
    answer_key = json.loads(answer_key_path.read_text())
    report = _render_perfect_report(answer_key)

    report["tooling"]["canonical_test_command"] = (
        "python target_repo/.ci/run_tests.py --suite all"
    )
    report["tooling"]["canonical_coverage_command"] = (
        "python target_repo/.ci/run_tests.py --suite all --coverage"
    )

    (workspace_root / "audit_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )

    initial_hashes = hash_repo_files(workspace_root / "target_repo")
    result = verify_submission(workspace_root, answer_key_path, initial_hashes)

    assert result.tooling_accuracy == 1.0


def test_degradation_accepts_flexible_shapes(tmp_path: Path) -> None:
    """Blocked tools/fallbacks can be represented as strings, dicts, or nested maps."""
    workspace_root, answer_key_path = _prepare_workspace(tmp_path, "instance_023")
    answer_key = json.loads(answer_key_path.read_text())
    report = _render_perfect_report(answer_key)

    report["tooling"]["blocked_tools"] = [
        "pydeps command not found",
        {"tool": "trivy", "reason": "scanner unavailable"},
    ]
    report["tooling"]["fallback_methods"] = {
        "cycles": "used artifacts/import_graph_snapshot.txt",
        "security": "used artifacts/vuln_scan_2026-01-15.json",
    }

    (workspace_root / "audit_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )

    initial_hashes = hash_repo_files(workspace_root / "target_repo")
    result = verify_submission(workspace_root, answer_key_path, initial_hashes)

    assert result.degradation_handling == 1.0


def test_verifier_detects_repo_modification(tmp_path: Path) -> None:
    workspace_root, answer_key_path = _prepare_workspace(tmp_path, "instance_002")
    answer_key = json.loads(answer_key_path.read_text())
    report = _render_perfect_report(answer_key)
    (workspace_root / "audit_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )

    initial_hashes = hash_repo_files(workspace_root / "target_repo")
    target_file = workspace_root / "target_repo" / "engine" / "core" / "processor.py"
    target_file.write_text(target_file.read_text() + "\n# modified during audit\n")

    result = verify_submission(workspace_root, answer_key_path, initial_hashes)
    assert result.passed is False
    assert result.repo_untouched is False
