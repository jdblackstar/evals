from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

from mutations import MUTATIONS, apply_mutations


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8192)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def hash_test_files(instance_dir: Path) -> dict[str, str]:
    tests_dir = instance_dir / "tests"
    return {
        str(path.relative_to(instance_dir)): sha256_file(path)
        for path in sorted(tests_dir.rglob("*.py"))
    }


def run_pytest(instance_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q"],
        cwd=instance_dir,
        check=False,
        capture_output=True,
        text=True,
    )


def choose_mutation_count(rng: random.Random) -> int:
    roll = rng.random()
    if roll < 0.40:
        return 1
    if roll < 0.80:
        return 2
    return 3


def generate_instances(
    template_dir: Path,
    output_dir: Path,
    num_instances: int,
    seed: int,
) -> None:
    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    template_test_result = run_pytest(template_dir)
    if template_test_result.returncode != 0:
        raise RuntimeError(
            "Template pipeline must pass tests before mutation.\n"
            f"stdout:\n{template_test_result.stdout}\n"
            f"stderr:\n{template_test_result.stderr}"
        )
    shutil.rmtree(template_dir / "outputs", ignore_errors=True)

    all_mutation_names = sorted(MUTATIONS)

    for index in range(1, num_instances + 1):
        instance_name = f"instance_{index:03d}"
        instance_dir = output_dir / instance_name
        if instance_dir.exists():
            shutil.rmtree(instance_dir)

        max_attempts = 50
        for attempt in range(1, max_attempts + 1):
            shutil.copytree(template_dir, instance_dir)

            bug_count = choose_mutation_count(rng)
            chosen = sorted(rng.sample(all_mutation_names, bug_count))
            applied = apply_mutations(instance_dir, chosen)

            tests_hashes = hash_test_files(instance_dir)
            (instance_dir / "tests_hashes.json").write_text(
                json.dumps(tests_hashes, indent=2, sort_keys=True)
            )

            (instance_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "instance_id": instance_name,
                        "seed": seed,
                        "attempt": attempt,
                        "bugs": [
                            {
                                "name": bug.name,
                                "category": bug.category,
                                "difficulty": bug.difficulty,
                                "description": bug.description,
                                "file_path": bug.file_path,
                            }
                            for bug in applied
                        ],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )

            result = run_pytest(instance_dir)
            if result.returncode != 0:
                shutil.rmtree(instance_dir / "outputs", ignore_errors=True)
                print(
                    f"Generated {instance_name} with bugs={chosen} (tests failing as expected)."
                )
                break

            shutil.rmtree(instance_dir)

        else:
            raise RuntimeError(
                f"Failed to produce failing instance {instance_name} after {max_attempts} attempts."
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate broken pipeline-debugger instances.")
    parser.add_argument(
        "--template-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "template_pipeline",
        help="Path to clean template pipeline.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "instances",
        help="Directory to write generated instances.",
    )
    parser.add_argument(
        "--num-instances",
        type=int,
        default=15,
        help="Number of instances to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    generate_instances(
        template_dir=args.template_dir,
        output_dir=args.output_dir,
        num_instances=args.num_instances,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
