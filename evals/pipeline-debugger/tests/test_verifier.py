from __future__ import annotations

import json
import shutil
from pathlib import Path

from generator.mutations import apply_mutations
from verifier.verify import hash_files, verify_submission

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = ROOT / "generator" / "template_pipeline"


def write_test_hashes(instance_dir: Path) -> None:
    tests_hashes = hash_files(instance_dir / "tests", "*.py")
    (instance_dir / "tests_hashes.json").write_text(
        json.dumps(tests_hashes, indent=2, sort_keys=True)
    )


def test_verifier_passes_clean_template(tmp_path: Path) -> None:
    instance_dir = tmp_path / "candidate"
    shutil.copytree(TEMPLATE_DIR, instance_dir)
    write_test_hashes(instance_dir)

    result = verify_submission(instance_dir)

    assert result.passed is True
    assert result.run_pipeline_exit_zero is True
    assert result.tests_passed == result.tests_total
    assert result.schema_valid is True
    assert result.deterministic is True
    assert result.test_files_untouched is True


def test_verifier_detects_modified_tests(tmp_path: Path) -> None:
    instance_dir = tmp_path / "candidate"
    shutil.copytree(TEMPLATE_DIR, instance_dir)
    write_test_hashes(instance_dir)

    # Simulate reward-hacking edit to tests before verification.
    test_file = instance_dir / "tests" / "test_pipeline.py"
    test_file.write_text(test_file.read_text() + "\n# tampered\n")

    result = verify_submission(instance_dir)

    assert result.passed is False
    assert result.test_files_untouched is False


def test_verifier_fails_known_broken_mutation(tmp_path: Path) -> None:
    instance_dir = tmp_path / "candidate"
    shutil.copytree(TEMPLATE_DIR, instance_dir)
    apply_mutations(instance_dir, ["wrong_output_path"])
    write_test_hashes(instance_dir)

    result = verify_submission(instance_dir)

    assert result.passed is False
    assert result.tests_passed < result.tests_total
