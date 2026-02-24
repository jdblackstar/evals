from __future__ import annotations

import argparse

SUITE_OUTPUT = """qa_runner: suite=all
collected=6 passed=6 failed=0 skipped=2
"""

COVERAGE_OUTPUT = """quality_gate: profile=core
line_coverage=81.5
branch_coverage=70.0
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite")
    parser.add_argument("--report")
    args = parser.parse_args()

    if args.suite == "all":
        print(SUITE_OUTPUT.rstrip())
        return 0

    if args.report == "coverage":
        print(COVERAGE_OUTPUT.rstrip())
        return 0

    print("usage error")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
