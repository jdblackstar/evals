#!/usr/bin/env bash
set -euo pipefail

echo "Running plugin verification suite..."
python -m pytest verify/ --tb=short
echo "All checks passed."
