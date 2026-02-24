#!/usr/bin/env bash
set -euo pipefail

echo "Running backend tests..."
pytest backend/tests/ -v
