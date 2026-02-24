# Plugin System

An extensible plugin framework using pluggy for data format processing.

## Running Tests

Run the verification suite:

```bash
./run_checks.sh
```

## Coverage

To generate a coverage report:

```bash
python -m pytest verify/ --cov=plugins --cov-report=term
```

A recent quality report snapshot is available in `artifacts/quality_report.txt`.

## Adding Plugins

Create a new module in `plugins/builtins/` that subclasses `BasePlugin` from `plugins.base`.
