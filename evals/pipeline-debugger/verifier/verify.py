from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class VerificationResult:
    passed: bool
    run_pipeline_exit_zero: bool
    tests_passed: int
    tests_total: int
    schema_valid: bool
    deterministic: bool
    test_files_untouched: bool
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["partial_credit"] = {
            "tests_passed": self.tests_passed,
            "tests_total": self.tests_total,
            "schema_valid": self.schema_valid,
            "deterministic": self.deterministic,
            "test_files_untouched": self.test_files_untouched,
            "run_pipeline_exit_zero": self.run_pipeline_exit_zero,
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


def hash_files(root: Path, pattern: str) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob(pattern))
        if path.is_file()
    }


def normalize_test_hashes(hashes: dict[str, str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for raw_path, digest in hashes.items():
        key = raw_path.replace("\\", "/")
        if key.startswith("tests/"):
            key = key[len("tests/") :]
        normalized[key] = digest
    return normalized


def hash_outputs(output_dir: Path) -> dict[str, str]:
    if not output_dir.exists():
        return {}
    return {
        str(path.relative_to(output_dir)): sha256_file(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file()
    }


def run_command(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )


def collect_total_tests(submission_dir: Path) -> int:
    result = run_command(
        [sys.executable, "-m", "pytest", "tests", "--collect-only", "-q"],
        cwd=submission_dir,
    )
    if result.returncode != 0:
        return 0
    return sum(1 for line in result.stdout.splitlines() if "::" in line)


def parse_tests_passed(pytest_output: str) -> int:
    match = re.search(r"(\d+)\s+passed", pytest_output)
    if not match:
        return 0
    return int(match.group(1))


def dtype_matches(series: pd.Series, expected: str) -> bool:
    if expected == "int64":
        return pd.api.types.is_integer_dtype(series)
    if expected == "float64":
        return pd.api.types.is_float_dtype(series)
    if expected == "bool":
        return pd.api.types.is_bool_dtype(series)
    if expected == "object":
        return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(
            series
        )
    return str(series.dtype) == expected


def validate_schema(submission_dir: Path) -> tuple[bool, list[str]]:
    errors: list[str] = []
    schema_path = submission_dir / "expected_schema.json"
    if not schema_path.exists():
        return False, ["expected_schema.json is missing"]

    schema = json.loads(schema_path.read_text())
    output_file = schema.get("output_file")
    if not output_file:
        return False, ["expected_schema.json missing output_file"]

    output_path = submission_dir / output_file
    if not output_path.exists():
        return False, [f"output file missing: {output_file}"]

    try:
        df = pd.read_csv(output_path)
    except Exception as exc:
        return False, [f"failed to read output file: {exc}"]

    expected_columns = schema.get("columns", [])
    actual_column_names = list(df.columns)
    expected_column_names = [col["name"] for col in expected_columns]
    if actual_column_names != expected_column_names:
        errors.append(
            f"column mismatch: expected {expected_column_names}, got {actual_column_names}"
        )

    for column_spec in expected_columns:
        name = column_spec["name"]
        expected_dtype = column_spec["dtype"]
        nullable = bool(column_spec.get("nullable", True))

        if name not in df.columns:
            errors.append(f"missing column: {name}")
            continue

        series = df[name]
        if not dtype_matches(series, expected_dtype):
            errors.append(
                f"dtype mismatch for {name}: expected {expected_dtype}, got {series.dtype}"
            )

        if not nullable and series.isnull().any():
            errors.append(f"nullability violation for {name}")

    for constraint in schema.get("constraints", []):
        column_name = constraint.get("column")
        if column_name not in df.columns:
            continue
        series = df[column_name]

        if "min" in constraint:
            min_value = float(constraint["min"])
            if (series < min_value).any():
                errors.append(f"constraint violation: {column_name} below {min_value}")

        if "max" in constraint:
            max_value = float(constraint["max"])
            if (series > max_value).any():
                errors.append(f"constraint violation: {column_name} above {max_value}")

        if "allowed_values" in constraint:
            allowed = set(constraint["allowed_values"])
            values = set(series.dropna().astype(str).unique().tolist())
            if not values.issubset(allowed):
                errors.append(
                    f"constraint violation: {column_name} has unexpected values {sorted(values - allowed)}"
                )

    return len(errors) == 0, errors


def check_test_file_integrity(
    submission_dir: Path,
    pre_hashes: dict[str, str],
    post_hashes: dict[str, str],
) -> tuple[bool, dict[str, Any]]:
    normalized_pre = normalize_test_hashes(pre_hashes)
    normalized_post = normalize_test_hashes(post_hashes)
    details: dict[str, Any] = {
        "pre_test_hashes": pre_hashes,
        "post_test_hashes": post_hashes,
        "normalized_pre_test_hashes": normalized_pre,
        "normalized_post_test_hashes": normalized_post,
    }

    expected_hash_path = submission_dir / "tests_hashes.json"
    expected_hashes: dict[str, str] | None = None
    normalized_expected: dict[str, str] | None = None
    if expected_hash_path.exists():
        expected_hashes = json.loads(expected_hash_path.read_text())
        normalized_expected = normalize_test_hashes(expected_hashes)
        details["expected_test_hashes"] = expected_hashes
        details["normalized_expected_test_hashes"] = normalized_expected

    unchanged_during_verifier = normalized_pre == normalized_post
    matches_expected = True
    if expected_hashes is not None:
        assert normalized_expected is not None
        matches_expected = normalized_pre == normalized_expected

    details["unchanged_during_verifier"] = unchanged_during_verifier
    details["matches_expected"] = matches_expected
    return unchanged_during_verifier and matches_expected, details


def verify_submission(submission_dir: Path) -> VerificationResult:
    submission_dir = submission_dir.resolve()
    test_dir = submission_dir / "tests"

    pre_test_hashes = hash_files(test_dir, "*.py") if test_dir.exists() else {}

    run_result_1 = run_command([sys.executable, "run_pipeline.py"], cwd=submission_dir)
    run_pipeline_exit_zero = run_result_1.returncode == 0

    tests_total = collect_total_tests(submission_dir)
    pytest_result = run_command(
        [sys.executable, "-m", "pytest", "tests", "-v"],
        cwd=submission_dir,
    )
    tests_passed = parse_tests_passed(pytest_result.stdout)
    all_tests_pass = pytest_result.returncode == 0 and tests_passed == tests_total

    post_test_hashes = hash_files(test_dir, "*.py") if test_dir.exists() else {}
    test_files_untouched, test_integrity_details = check_test_file_integrity(
        submission_dir,
        pre_test_hashes,
        post_test_hashes,
    )

    schema_valid, schema_errors = validate_schema(submission_dir)

    outputs_dir = submission_dir / "outputs"
    outputs_hash_before = hash_outputs(outputs_dir)
    run_result_2 = run_command([sys.executable, "run_pipeline.py"], cwd=submission_dir)
    outputs_hash_after = hash_outputs(outputs_dir)
    deterministic = (
        run_result_2.returncode == 0
        and bool(outputs_hash_before)
        and outputs_hash_before == outputs_hash_after
    )

    passed = all(
        [
            run_pipeline_exit_zero,
            all_tests_pass,
            test_files_untouched,
            schema_valid,
            deterministic,
        ]
    )

    details: dict[str, Any] = {
        "submission_dir": str(submission_dir),
        "run_pipeline_first": {
            "exit_code": run_result_1.returncode,
            "stdout": run_result_1.stdout,
            "stderr": run_result_1.stderr,
        },
        "pytest": {
            "exit_code": pytest_result.returncode,
            "stdout": pytest_result.stdout,
            "stderr": pytest_result.stderr,
        },
        "run_pipeline_second": {
            "exit_code": run_result_2.returncode,
            "stdout": run_result_2.stdout,
            "stderr": run_result_2.stderr,
        },
        "schema_errors": schema_errors,
        "outputs_hash_before": outputs_hash_before,
        "outputs_hash_after": outputs_hash_after,
        "all_tests_pass": all_tests_pass,
    }
    details.update(test_integrity_details)

    return VerificationResult(
        passed=passed,
        run_pipeline_exit_zero=run_pipeline_exit_zero,
        tests_passed=tests_passed,
        tests_total=tests_total,
        schema_valid=schema_valid,
        deterministic=deterministic,
        test_files_untouched=test_files_untouched,
        details=details,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify a pipeline-debugger submission."
    )
    parser.add_argument(
        "submission_dir",
        type=Path,
        help="Path to a single instance directory to verify.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write JSON result.",
    )
    args = parser.parse_args()

    result = verify_submission(args.submission_dir)
    payload = result.to_dict()

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True))

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
