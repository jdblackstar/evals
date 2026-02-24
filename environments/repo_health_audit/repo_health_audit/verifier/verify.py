from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class VerificationResult:
    passed: bool
    factual_accuracy: float
    judgment_accuracy: float
    tooling_accuracy: float
    degradation_handling: float
    evidence_grounding: float
    repo_untouched: bool
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["partial_credit"] = {
            "factual_accuracy": self.factual_accuracy,
            "judgment_accuracy": self.judgment_accuracy,
            "tooling_accuracy": self.tooling_accuracy,
            "degradation_handling": self.degradation_handling,
            "evidence_grounding": self.evidence_grounding,
            "repo_untouched": self.repo_untouched,
        }
        return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8192)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


_IGNORED_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    "target",
}
_IGNORED_FILE_PATTERNS = {
    ".coverage",
    ".coverage.*",
    "**/.coverage",
    "**/.coverage.*",
    "*.pyc",
    "*.pyo",
    "**/*.pyc",
    "**/*.pyo",
}


def _is_ignored_repo_path(relative_path: Path) -> bool:
    if any(part in _IGNORED_DIR_NAMES for part in relative_path.parts):
        return True

    relative_text = relative_path.as_posix()
    for pattern in _IGNORED_FILE_PATTERNS:
        if Path(relative_text).match(pattern):
            return True
    return False


def hash_repo_files(repo_root: Path) -> dict[str, str]:
    if not repo_root.exists():
        return {}
    result: dict[str, str] = {}
    for path in sorted(repo_root.rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(repo_root)
        if _is_ignored_repo_path(relative_path):
            continue
        result[str(relative_path)] = sha256_file(path)
    return result


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _normalize_text(value: Any) -> str:
    text = str(value).strip().lower()
    return re.sub(r"\s+", " ", text)


def _normalize_path_text(value: Any) -> str:
    text = str(value).strip().replace("\\", "/")
    text = re.sub(r"/+", "/", text)
    if "target_repo/" in text:
        text = text.split("target_repo/", 1)[1]
    if text.startswith("./"):
        text = text[2:]
    return text.strip("/")


def _extract_path_hints(value: Any) -> list[str]:
    text = str(value).replace("\\", "/")
    candidates = re.findall(
        r"(?:target_repo/)?[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+",
        text,
    )
    return [_normalize_path_text(item) for item in candidates if item]


def _flatten_string_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        results: list[str] = []
        for item in value:
            results.extend(_flatten_string_values(item))
        return results
    if isinstance(value, dict):
        results: list[str] = []
        for key, item in value.items():
            results.extend(_flatten_string_values(key))
            results.extend(_flatten_string_values(item))
        return results
    return [str(value)]


def _strip_env_prefix(tokens: list[str]) -> list[str]:
    remaining = list(tokens)
    env_assignment = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*$")
    while remaining and env_assignment.match(remaining[0]):
        remaining = remaining[1:]
    return remaining


def _looks_like_path_token(token: str) -> bool:
    return "/" in token or token.endswith(".py") or token.startswith("./")


def _normalize_command_token(token: str) -> str:
    if token in {"&&", "||", "|", ";"}:
        return token
    normalized = token.strip()
    if _looks_like_path_token(normalized):
        normalized = _normalize_path_text(normalized)
    return _normalize_text(normalized)


def _strip_shell_prefix(tokens: list[str]) -> list[str]:
    if len(tokens) >= 2 and tokens[0] in {"bash", "sh", "zsh"}:
        return tokens[1:]
    return tokens


def _is_target_repo_reference(token: str) -> bool:
    normalized = _normalize_path_text(token)
    return normalized in {"", ".", "target_repo"}


def _strip_cd_target_repo_prefix(tokens: list[str]) -> list[str]:
    if len(tokens) >= 3 and tokens[0] == "cd" and tokens[2] == "&&":
        if _is_target_repo_reference(tokens[1]):
            return tokens[3:]
    return tokens


def _canonicalize_command(command: Any) -> list[str]:
    if not isinstance(command, str):
        return [_normalize_text(command)]
    raw = command.strip()
    if not raw:
        return []
    try:
        tokens = shlex.split(raw)
    except ValueError:
        tokens = raw.split()
    tokens = _strip_env_prefix(tokens)
    tokens = _strip_cd_target_repo_prefix(tokens)
    tokens = _strip_shell_prefix(tokens)
    return [_normalize_command_token(token) for token in tokens if token]


def _contains_contiguous_subsequence(haystack: list[str], needle: list[str]) -> bool:
    if not needle:
        return True
    if len(needle) > len(haystack):
        return False
    for index in range(len(haystack) - len(needle) + 1):
        if haystack[index : index + len(needle)] == needle:
            return True
    return False


def _commands_equivalent(actual: Any, expected: Any) -> bool:
    actual_tokens = _canonicalize_command(actual)
    expected_tokens = _canonicalize_command(expected)
    if actual_tokens == expected_tokens:
        return True
    return _contains_contiguous_subsequence(actual_tokens, expected_tokens) or (
        _contains_contiguous_subsequence(expected_tokens, actual_tokens)
    )


def _looks_like_command(text: str) -> bool:
    normalized = text.strip()
    if not normalized:
        return False
    if " " in normalized or "&&" in normalized:
        return True
    return normalized.startswith(("./", "python", "pytest", "nox", "make", "coverage"))


def _text_requirement_matches(entry: Any, requirement: Any) -> bool:
    entry_text = str(entry)
    required_text = str(requirement)
    normalized_entry = _normalize_text(entry_text)
    normalized_required = _normalize_text(required_text)

    if normalized_required in normalized_entry:
        return True

    stripped_required = re.sub(
        r"^(reviewed|used|parsed|via|from)\s+",
        "",
        normalized_required,
    )
    if stripped_required and stripped_required in normalized_entry:
        return True

    entry_paths = _extract_path_hints(entry_text)
    required_paths = _extract_path_hints(required_text)
    if required_paths:
        for required_path in required_paths:
            if any(required_path in entry_path for entry_path in entry_paths):
                return True
            if required_path in _normalize_path_text(entry_text):
                return True

    if _looks_like_command(required_text) and _looks_like_command(entry_text):
        if _commands_equivalent(entry_text, required_text):
            return True

    return False


def _tool_aliases(tool_name: str) -> list[str]:
    normalized = _normalize_text(tool_name)
    alias_map = {
        "pytest-cov": ["pytest-cov", "--cov", "cov-report"],
        "coverage": ["coverage", "python -m coverage"],
        "pydeps": ["pydeps", "import_graph", "check_cycles"],
        "trivy": ["trivy", "vuln_scan", "security_scan"],
        "semgrep": ["semgrep"],
        "bandit": ["bandit"],
        "poetry": ["poetry"],
        "go": ["go", "go test", "go.mod"],
        "npm": ["npm", "node"],
        "cargo": ["cargo", "cargo test", "cargo tarpaulin"],
    }
    aliases = alias_map.get(normalized, [normalized])
    return [_normalize_text(alias) for alias in aliases]


def _tool_requirement_met(tool_name: str, signals: list[str]) -> bool:
    aliases = _tool_aliases(tool_name)
    return any(
        any(alias in signal for alias in aliases)
        for signal in signals
    )


def _field_gap_aliases(field_path: str) -> list[str]:
    field = _normalize_text(field_path)
    leaf = _normalize_text(field_path.split(".")[-1])
    aliases = {field, leaf}
    if leaf.endswith("_percent"):
        aliases.add(leaf.replace("_percent", ""))
    return sorted(alias for alias in aliases if alias)


def _gap_field_recorded(
    field_path: str,
    gap_fields: set[str],
    gap_signals: list[str],
) -> bool:
    aliases = _field_gap_aliases(field_path)
    normalized_gap_fields = {_normalize_text(field) for field in gap_fields}
    if any(alias in normalized_gap_fields for alias in aliases):
        return True
    return any(
        any(alias in signal for alias in aliases)
        for signal in gap_signals
    )


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        return [str(value)]
    return [str(item) for item in value]


def _get_nested(payload: dict[str, Any], dotted_path: str) -> Any:
    cursor: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return None
        cursor = cursor[part]
    return cursor


def _set_nested(payload: dict[str, Any], dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")
    cursor = payload
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


_BOOL_TO_YESNO = {True: "yes", False: "no"}


def _is_tooling_command_field(field_path: str) -> bool:
    return field_path.startswith("tooling.") and field_path.endswith("_command")


def _is_close(actual: Any, expected: Any, tolerance: float = 0.0) -> bool:
    if expected is None:
        return actual is None

    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual is expected

    if isinstance(expected, (int, float)):
        if actual is None:
            return False
        try:
            actual_float = float(actual)
            expected_float = float(expected)
        except (TypeError, ValueError):
            return False
        return abs(actual_float - expected_float) <= tolerance

    # Normalize booleans to yes/no for string comparison
    normalized_actual = _BOOL_TO_YESNO.get(actual, actual) if isinstance(actual, bool) else actual

    return _normalize_text(normalized_actual) == _normalize_text(expected)


def _evaluate_field_group(
    report: dict[str, Any],
    expected: dict[str, Any],
    fields: list[str],
    tolerances: dict[str, float],
) -> tuple[float, dict[str, Any]]:
    if not fields:
        return 1.0, {}

    mismatches: dict[str, Any] = {}
    correct = 0

    for field_path in fields:
        expected_value = expected.get(field_path)
        actual_value = _get_nested(report, field_path)
        tolerance = float(tolerances.get(field_path, 0.0))
        if _is_tooling_command_field(field_path):
            ok = _commands_equivalent(actual_value, expected_value)
        else:
            ok = _is_close(actual_value, expected_value, tolerance=tolerance)
        if ok:
            correct += 1
            continue
        mismatches[field_path] = {
            "expected": expected_value,
            "actual": actual_value,
            "tolerance": tolerance,
        }

    return correct / len(fields), mismatches


def _evaluate_degradation_handling(
    report: dict[str, Any],
    required_gap_fields: list[str],
    required_blocked_tools: list[str],
    required_fallback_methods: list[str],
) -> tuple[float, dict[str, Any]]:
    checks = 0
    passes = 0
    details: dict[str, Any] = {}

    measurement_gaps = report.get("measurement_gaps")
    gap_fields: set[str] = set()
    gap_signals_raw: list[str] = []
    if isinstance(measurement_gaps, list):
        for entry in measurement_gaps:
            if isinstance(entry, dict):
                field_name = entry.get("field")
                if field_name is not None:
                    gap_fields.add(str(field_name))
            gap_signals_raw.extend(_flatten_string_values(entry))
    elif isinstance(measurement_gaps, dict):
        for key, value in measurement_gaps.items():
            gap_fields.add(str(key))
            gap_signals_raw.extend(_flatten_string_values(value))
    else:
        gap_signals_raw.extend(_flatten_string_values(measurement_gaps))
    gap_signals = [_normalize_text(item) for item in gap_signals_raw]

    tooling_map = report.get("tooling")
    tooling = tooling_map if isinstance(tooling_map, dict) else {}
    blocked_signals_raw = _flatten_string_values(tooling.get("blocked_tools"))
    fallback_signals_raw = _flatten_string_values(tooling.get("fallback_methods"))
    evidence_signals_raw = _flatten_evidence_values(report.get("evidence", {}))

    blocked_signals = [_normalize_text(item) for item in blocked_signals_raw]
    degradation_signals = blocked_signals + gap_signals + [
        _normalize_text(item)
        for item in fallback_signals_raw + evidence_signals_raw
    ]
    fallback_search_space = (
        fallback_signals_raw
        + evidence_signals_raw
        + gap_signals_raw
        + blocked_signals_raw
    )

    for field_path in required_gap_fields:
        checks += 1
        actual_value = _get_nested(report, field_path)
        value_allows_gap = (
            actual_value is None or _normalize_text(actual_value) == "inconclusive"
        )
        has_gap_entry = _gap_field_recorded(field_path, gap_fields, gap_signals)
        if value_allows_gap and has_gap_entry:
            passes += 1
            continue
        details[field_path] = {
            "value_allows_gap": value_allows_gap,
            "has_gap_entry": has_gap_entry,
            "actual": actual_value,
        }

    for tool_name in required_blocked_tools:
        checks += 1
        if _tool_requirement_met(tool_name, degradation_signals):
            passes += 1
            continue
        details[f"blocked_tool::{tool_name}"] = {"missing": True}

    for fallback in required_fallback_methods:
        checks += 1
        if any(
            _text_requirement_matches(entry, fallback)
            for entry in fallback_search_space
        ):
            passes += 1
            continue
        details[f"fallback::{fallback}"] = {"missing": True}

    if checks == 0:
        return 1.0, {}
    return passes / checks, details


def _flatten_evidence_values(obj: Any) -> list[str]:
    """Recursively extract all string values from a nested evidence structure."""
    return _flatten_string_values(obj)


def _evaluate_evidence(
    report: dict[str, Any],
    evidence_requirements: dict[str, list[str]],
) -> tuple[float, dict[str, Any]]:
    if not evidence_requirements:
        return 1.0, {}

    evidence = report.get("evidence")
    evidence_map: dict[str, Any] = evidence if isinstance(evidence, dict) else {}
    checks = 0
    passes = 0
    failures: dict[str, Any] = {}

    # Pre-compute flattened evidence for fallback searches
    all_evidence_strings = _flatten_evidence_values(evidence_map)
    normalized_all_evidence = [_normalize_text(s) for s in all_evidence_strings]

    for key, required_substrings in evidence_requirements.items():
        checks += 1
        entries = _as_string_list(evidence_map.get(key))

        # Primary check: keyed lookup
        if entries:
            all_found = all(
                any(_text_requirement_matches(entry, required) for entry in entries)
                for required in required_substrings
            )
            if all_found:
                passes += 1
                continue

        # Fallback: search all evidence values regardless of key structure
        all_found_fallback = all(
            any(
                _text_requirement_matches(entry, required)
                for entry in all_evidence_strings
            )
            for required in required_substrings
        )
        if all_found_fallback:
            passes += 1
            continue

        failures[key] = {
            "reason": "no_required_substring_match",
            "entries": entries if entries else normalized_all_evidence[:10],
            "required_substrings": required_substrings,
        }

    return passes / checks, failures


def verify_submission(
    workspace_root: Path,
    answer_key_path: Path,
    initial_repo_hashes: dict[str, str] | None = None,
) -> VerificationResult:
    workspace_root = workspace_root.resolve()
    answer_key = _load_json(answer_key_path.resolve())

    report_path = workspace_root / "audit_report.json"
    target_repo = workspace_root / "target_repo"
    final_repo_hashes = hash_repo_files(target_repo)
    repo_untouched = (
        True
        if initial_repo_hashes is None
        else dict(initial_repo_hashes) == final_repo_hashes
    )

    if not report_path.exists():
        return VerificationResult(
            passed=False,
            factual_accuracy=0.0,
            judgment_accuracy=0.0,
            tooling_accuracy=0.0,
            degradation_handling=0.0,
            evidence_grounding=0.0,
            repo_untouched=repo_untouched,
            details={
                "error": "missing audit_report.json",
                "report_path": str(report_path),
                "repo_untouched": repo_untouched,
            },
        )

    try:
        report = _load_json(report_path)
    except json.JSONDecodeError as exc:
        return VerificationResult(
            passed=False,
            factual_accuracy=0.0,
            judgment_accuracy=0.0,
            tooling_accuracy=0.0,
            degradation_handling=0.0,
            evidence_grounding=0.0,
            repo_untouched=repo_untouched,
            details={"error": f"invalid JSON: {exc}"},
        )

    expected = dict(answer_key.get("expected", {}))
    tolerances = {
        str(key): float(value)
        for key, value in dict(answer_key.get("tolerances", {})).items()
    }

    factual_fields = [str(item) for item in answer_key.get("factual_fields", [])]
    judgment_fields = [str(item) for item in answer_key.get("judgment_fields", [])]
    tooling_fields = [str(item) for item in answer_key.get("tooling_fields", [])]

    factual_accuracy, factual_mismatches = _evaluate_field_group(
        report, expected, factual_fields, tolerances
    )
    judgment_accuracy, judgment_mismatches = _evaluate_field_group(
        report, expected, judgment_fields, tolerances
    )
    tooling_accuracy, tooling_mismatches = _evaluate_field_group(
        report, expected, tooling_fields, tolerances
    )

    required_gap_fields = [
        str(item) for item in answer_key.get("required_gap_fields", [])
    ]
    required_blocked_tools = [
        str(item) for item in answer_key.get("required_blocked_tools", [])
    ]
    required_fallback_methods = [
        str(item) for item in answer_key.get("required_fallback_methods", [])
    ]
    degradation_handling, degradation_failures = _evaluate_degradation_handling(
        report,
        required_gap_fields=required_gap_fields,
        required_blocked_tools=required_blocked_tools,
        required_fallback_methods=required_fallback_methods,
    )

    evidence_requirements_raw = dict(answer_key.get("evidence_requirements", {}))
    evidence_requirements = {
        str(key): [str(item) for item in value]
        for key, value in evidence_requirements_raw.items()
        if isinstance(value, list)
    }
    evidence_grounding, evidence_failures = _evaluate_evidence(
        report,
        evidence_requirements=evidence_requirements,
    )

    # Ensure explicit null fields are honored even when not part of field groups.
    null_field_failures: dict[str, Any] = {}
    for field_path in [str(item) for item in answer_key.get("null_fields", [])]:
        actual_value = _get_nested(report, field_path)
        if actual_value is None:
            continue
        null_field_failures[field_path] = {"expected": None, "actual": actual_value}

    thresholds = dict(answer_key.get("thresholds", {}))
    factual_min = float(thresholds.get("factual_accuracy", 1.0))
    judgment_min = float(thresholds.get("judgment_accuracy", 1.0))
    tooling_min = float(thresholds.get("tooling_accuracy", 1.0))
    degradation_min = float(thresholds.get("degradation_handling", 1.0))
    evidence_min = float(thresholds.get("evidence_grounding", 1.0))

    passed = (
        factual_accuracy >= factual_min
        and judgment_accuracy >= judgment_min
        and tooling_accuracy >= tooling_min
        and degradation_handling >= degradation_min
        and evidence_grounding >= evidence_min
        and not null_field_failures
        and repo_untouched
    )

    details: dict[str, Any] = {
        "workspace_root": str(workspace_root),
        "answer_key_path": str(answer_key_path),
        "thresholds": {
            "factual_accuracy": factual_min,
            "judgment_accuracy": judgment_min,
            "tooling_accuracy": tooling_min,
            "degradation_handling": degradation_min,
            "evidence_grounding": evidence_min,
        },
        "mismatches": {
            "factual": factual_mismatches,
            "judgment": judgment_mismatches,
            "tooling": tooling_mismatches,
            "null_fields": null_field_failures,
        },
        "degradation_failures": degradation_failures,
        "evidence_failures": evidence_failures,
        "repo_untouched": repo_untouched,
    }

    return VerificationResult(
        passed=passed,
        factual_accuracy=factual_accuracy,
        judgment_accuracy=judgment_accuracy,
        tooling_accuracy=tooling_accuracy,
        degradation_handling=degradation_handling,
        evidence_grounding=evidence_grounding,
        repo_untouched=repo_untouched,
        details=details,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify a repo-health-audit submission."
    )
    parser.add_argument(
        "workspace_root",
        type=Path,
        help="Path to evaluated workspace (must include target_repo and audit_report.json).",
    )
    parser.add_argument(
        "answer_key_path",
        type=Path,
        help="Path to instance answer_key.json.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional output path for verification JSON.",
    )
    args = parser.parse_args()

    result = verify_submission(args.workspace_root, args.answer_key_path)
    payload = result.to_dict()
    rendered = json.dumps(payload, indent=2, sort_keys=True)

    if args.json_out is not None:
        args.json_out.write_text(rendered)
    print(rendered)
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
