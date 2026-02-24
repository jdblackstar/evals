#!/usr/bin/env bash
set -euo pipefail

echo "Running semgrep..."
semgrep --config=auto services/ lib/ 2>&1 || {
    echo "semgrep: command not found" >&2
    exit 127
}

echo "Running bandit..."
bandit -r services/ lib/ 2>&1 || {
    echo "bandit: command not found" >&2
    exit 127
}
