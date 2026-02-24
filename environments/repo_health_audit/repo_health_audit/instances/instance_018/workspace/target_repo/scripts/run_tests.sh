#!/usr/bin/env bash
set -euo pipefail

echo "Running Python tests..."
pytest services/api/tests/ -v

echo "Running Go tests..."
cd services/worker && go test ./...
